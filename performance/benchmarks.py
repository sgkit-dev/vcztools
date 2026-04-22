"""Benchmark runner for vcztools' VczReader iteration paths.

Runs a catalogue of 10 tasks against 4 store backends (local directory,
local zip, local HTTP via ``python -m http.server``, obstore over
``file://``) and writes one JSON-lines row per (task, backend, repeat).
Record counts are checked against exact expected values in
``<dataset>.benchmark_meta.json`` — a silently broken filter cannot pass
as a speed win.

Known platform / Python limits
------------------------------
- ``local-http`` needs ``fsspec[http]`` (pulls ``aiohttp``). The
  ``benchmark`` dependency group installs it; on a fresh env without
  that group the backend will fail. Pass ``--skip-backend local-http``
  to work around.
- ``obstore-file`` needs the ``obstore`` package; same workaround.
"""

import argparse
import dataclasses
import json
import logging
import pathlib
import socket
import subprocess
import sys
import threading
import time

import generate_dataset
import numpy as np
import psutil
import requests

from vcztools import bcftools_filter, retrieval
from vcztools import regions as regions_mod
from vcztools import utils as vcz_utils

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Task:
    """A single benchmark task.

    ``build_reader`` is called with an opened zarr root and the loaded
    metadata dict; it returns a configured :class:`VczReader`. The name
    is used as the key into ``benchmark_meta.json``'s
    ``expected_records`` table.
    """

    name: str
    build_reader: object
    expected_key: str


@dataclasses.dataclass
class Result:
    task: str
    backend: str
    num_samples: int
    num_variants: int
    repeat_index: int
    elapsed_s: float
    records: int
    peak_rss_mb: float
    profile: str
    git_sha: str
    timestamp: str
    hostname: str


# ---------------------------------------------------------------------------
# Backend openers
# ---------------------------------------------------------------------------


class BackendOpener:
    """Open a VCZ at ``vcz_path`` using a specific store backend.

    ``open(vcz_path)`` returns ``(root, cleanup)``. ``cleanup`` is
    called unconditionally after the task loop; for file-backed
    backends it is a no-op, for HTTP it tears down the server.
    """

    name: str = ""

    def open(self, vcz_path: pathlib.Path):
        raise NotImplementedError


class LocalDirOpener(BackendOpener):
    name = "local-dir"

    def open(self, vcz_path):
        root = vcz_utils.open_zarr(str(vcz_path))
        return root, lambda: None


class LocalZipOpener(BackendOpener):
    name = "local-zip"

    def open(self, vcz_path):
        zip_path = vcz_path.parent / (vcz_path.name + ".zip")
        root = vcz_utils.open_zarr(str(zip_path))
        return root, lambda: None


class ObstoreFileOpener(BackendOpener):
    name = "obstore-file"

    def open(self, vcz_path):
        root = vcz_utils.open_zarr(str(vcz_path), zarr_backend_storage="obstore")
        return root, lambda: None


class LocalHttpOpener(BackendOpener):
    name = "local-http"

    def open(self, vcz_path):
        port = _pick_free_port()
        parent = vcz_path.parent
        proc = subprocess.Popen(
            [sys.executable, "-m", "http.server", str(port)],
            cwd=str(parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._wait_for_server(port, vcz_path.name)
        url = f"http://localhost:{port}/{vcz_path.name}/"
        root = vcz_utils.open_zarr(url)

        def cleanup():
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

        return root, cleanup

    def _wait_for_server(self, port: int, vcz_dirname: str) -> None:
        url = f"http://localhost:{port}/{vcz_dirname}/"
        deadline = time.monotonic() + 10.0
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            try:
                r = requests.get(url, timeout=1.0)
                if r.status_code < 500:
                    return
            except requests.RequestException as e:
                last_err = e
            time.sleep(0.05)
        raise RuntimeError(
            f"local-http backend never became ready at {url}: {last_err}"
        )


def _pick_free_port() -> int:
    s = socket.socket()
    s.bind(("localhost", 0))
    port = s.getsockname()[1]
    s.close()
    return port


ALL_BACKENDS: dict[str, BackendOpener] = {
    opener.name: opener
    for opener in (
        LocalDirOpener(),
        LocalZipOpener(),
        LocalHttpOpener(),
        ObstoreFileOpener(),
    )
}


# ---------------------------------------------------------------------------
# Task builders
# ---------------------------------------------------------------------------


def _build_iter_no_fields(root, meta):
    return retrieval.VczReader(root)


def _build_iter_info_only(root, meta):
    return retrieval.VczReader(root)


def _build_iter_info_and_format(root, meta):
    return retrieval.VczReader(root)


def _build_subset_10_samples(root, meta):
    reader = retrieval.VczReader(root)
    rng = np.random.default_rng(12345)
    num_samples = int(root["sample_id"].shape[0])
    # 10 random indexes, deterministic per run.
    indexes = rng.choice(num_samples, size=min(10, num_samples), replace=False)
    reader.set_samples(sorted(int(i) for i in indexes))
    return reader


def _build_subset_half_samples(root, meta):
    reader = retrieval.VczReader(root)
    num_samples = int(root["sample_id"].shape[0])
    reader.set_samples(list(range(num_samples // 2)))
    return reader


def _build_region_10pct(root, meta):
    reader = retrieval.VczReader(root)
    rs = meta["region_spec"]
    region = f"{rs['contig']}:{rs['start']}-{rs['end']}"
    plan = regions_mod.build_chunk_plan(root, regions=region)
    reader.set_variants(plan)
    return reader


def _build_filter_info_dp_gt_80(root, meta):
    reader = retrieval.VczReader(root)
    vf = bcftools_filter.BcftoolsFilter(
        field_names=reader.field_names, include="INFO/DP>80"
    )
    reader.set_variant_filter(vf)
    return reader


def _build_filter_format_gq_gt_50(root, meta):
    reader = retrieval.VczReader(root)
    vf = bcftools_filter.BcftoolsFilter(
        field_names=reader.field_names, include="FMT/GQ>50"
    )
    # Pre-subset semantics — filter must see every sample.
    reader.set_variant_filter(vf, filter_on_subset_samples=False)
    return reader


def _build_iter_genotypes_only(root, meta):
    return retrieval.VczReader(root)


def _build_region_and_sample_subset(root, meta):
    reader = retrieval.VczReader(root)
    num_samples = int(root["sample_id"].shape[0])
    take = max(1, num_samples // 100)
    reader.set_samples(list(range(take)))
    rs = meta["region_spec"]
    region = f"{rs['contig']}:{rs['start']}-{rs['end']}"
    plan = regions_mod.build_chunk_plan(root, regions=region)
    reader.set_variants(plan)
    return reader


TASK_FIELDS: dict[str, list[str] | None] = {
    "iter_no_fields": [],
    "iter_info_only": [
        "variant_position",
        "variant_contig",
        "variant_length",
        "variant_DP",
        "variant_QUAL",
        "variant_IMPACT",
    ],
    "iter_info_and_format": [
        "variant_position",
        "variant_contig",
        "variant_length",
        "variant_DP",
        "variant_QUAL",
        "call_DP",
        "call_GQ",
        "call_genotype",
    ],
    "subset_10_samples": ["call_genotype"],
    "subset_half_samples": ["call_genotype"],
    "region_10pct": ["variant_position"],
    "filter_info_dp_gt_80": ["variant_position", "variant_DP"],
    "filter_format_gq_gt_50": ["variant_position"],
    "iter_genotypes_only": ["call_genotype"],
    "region_and_sample_subset": ["call_genotype"],
}


TASKS: list[Task] = [
    Task("iter_no_fields", _build_iter_no_fields, "iter_no_fields"),
    Task("iter_info_only", _build_iter_info_only, "iter_info_only"),
    Task("iter_info_and_format", _build_iter_info_and_format, "iter_info_and_format"),
    Task("subset_10_samples", _build_subset_10_samples, "subset_10_samples"),
    Task("subset_half_samples", _build_subset_half_samples, "subset_half_samples"),
    Task("region_10pct", _build_region_10pct, "region_10pct"),
    Task("filter_info_dp_gt_80", _build_filter_info_dp_gt_80, "filter_info_dp_gt_80"),
    Task(
        "filter_format_gq_gt_50",
        _build_filter_format_gq_gt_50,
        "filter_format_gq_gt_50",
    ),
    Task("iter_genotypes_only", _build_iter_genotypes_only, "iter_genotypes_only"),
    Task(
        "region_and_sample_subset",
        _build_region_and_sample_subset,
        "region_and_sample_subset",
    ),
]


# ---------------------------------------------------------------------------
# Peak RSS sampler
# ---------------------------------------------------------------------------


class PeakRssSampler:
    """Thread-based peak-RSS tracker (10 ms sample interval)."""

    def __init__(self):
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._peak_bytes = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._peak_bytes = self._proc.memory_info().rss
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join(timeout=1.0)
        return False

    def _run(self):
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss
            if rss > self._peak_bytes:
                self._peak_bytes = rss
            self._stop.wait(0.01)

    @property
    def peak_mb(self) -> float:
        return self._peak_bytes / (1024 * 1024)


# ---------------------------------------------------------------------------
# Task execution
# ---------------------------------------------------------------------------


def _count_variants(reader: retrieval.VczReader, fields: list[str] | None) -> int:
    """Iterate the reader and count rows.

    ``fields=[]`` is special-cased: :meth:`VczReader.variant_chunks`
    short-circuits to an empty generator in that case, so we walk the
    plan directly and count per-chunk variants using ``variants_chunk_size``.
    That preserves the "chunk-scheduling only" intent of the
    ``iter_no_fields`` task — no array reads, but the record count still
    matches ``num_variants``.
    """
    if fields is not None and len(fields) == 0:
        chunk_size = reader.variants_chunk_size
        num_variants = reader.num_variants
        total = 0
        for cr in reader.variant_chunk_plan:
            if cr.selection is not None:
                total += int(cr.selection.size)
                continue
            start = cr.index * chunk_size
            stop = min(start + chunk_size, num_variants)
            total += stop - start
        return total
    return sum(1 for _ in reader.variants(fields=fields))


def run_task(
    task: Task,
    backend: BackendOpener,
    dataset: pathlib.Path,
    meta: dict,
    repeat_index: int,
    profile: str,
    git_sha: str,
    hostname: str,
) -> Result:
    root, cleanup = backend.open(dataset)
    try:
        reader = task.build_reader(root, meta)
        fields = TASK_FIELDS[task.name]
        with PeakRssSampler() as sampler:
            t0 = time.perf_counter()
            records = _count_variants(reader, fields)
            elapsed = time.perf_counter() - t0
    finally:
        cleanup()

    expected = int(meta["expected_records"][task.expected_key])
    if records != expected:
        raise AssertionError(
            f"task {task.name!r} on backend {backend.name!r} emitted "
            f"{records} records, expected exactly {expected}"
        )

    return Result(
        task=task.name,
        backend=backend.name,
        num_samples=int(meta["num_samples"]),
        num_variants=int(meta["num_variants"]),
        repeat_index=repeat_index,
        elapsed_s=elapsed,
        records=records,
        peak_rss_mb=sampler.peak_mb,
        profile=profile,
        git_sha=git_sha,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        hostname=hostname,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _discover_git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _write_jsonl_row(fp, result: Result) -> None:
    fp.write(json.dumps(dataclasses.asdict(result), sort_keys=True))
    fp.write("\n")
    fp.flush()


def _select_tasks(names: list[str] | None) -> list[Task]:
    if names is None or len(names) == 0:
        return list(TASKS)
    task_by_name = {t.name: t for t in TASKS}
    missing = [n for n in names if n not in task_by_name]
    if len(missing) > 0:
        raise SystemExit(f"unknown task(s): {missing}")
    return [task_by_name[n] for n in names]


def _select_backends(
    include: list[str] | None, skip: list[str] | None
) -> list[BackendOpener]:
    names = list(ALL_BACKENDS) if include is None or len(include) == 0 else include
    for n in names:
        if n not in ALL_BACKENDS:
            raise SystemExit(f"unknown backend {n!r}")
    if skip is not None:
        names = [n for n in names if n not in set(skip)]
    return [ALL_BACKENDS[n] for n in names]


def run(args) -> None:
    dataset = args.dataset
    meta = generate_dataset.load_meta(dataset)
    tasks = _select_tasks(args.task)
    backends = _select_backends(args.backend, args.skip_backend)

    git_sha = _discover_git_sha()
    hostname = socket.gethostname()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fp:
        for task in tasks:
            for backend in backends:
                for repeat_index in range(args.repeats):
                    logger.info(
                        "%s / %s [%d/%d]",
                        task.name,
                        backend.name,
                        repeat_index + 1,
                        args.repeats,
                    )
                    result = run_task(
                        task,
                        backend,
                        dataset,
                        meta,
                        repeat_index,
                        args.profile,
                        git_sha,
                        hostname,
                    )
                    _write_jsonl_row(fp, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run the benchmark suite")
    run_p.add_argument("--dataset", type=pathlib.Path, required=True)
    run_p.add_argument("--output", type=pathlib.Path, required=True)
    run_p.add_argument("--repeats", type=int, default=3)
    run_p.add_argument(
        "--backend",
        action="append",
        default=None,
        help="Restrict to these backends; may repeat",
    )
    run_p.add_argument(
        "--task",
        action="append",
        default=None,
        help="Restrict to these tasks; may repeat",
    )
    run_p.add_argument(
        "--skip-backend",
        action="append",
        default=None,
        help="Skip these backends (e.g. local-http on a python build "
        "without aiohttp); may repeat",
    )
    run_p.add_argument(
        "--profile",
        choices=("small", "large"),
        default="small",
        help="Recorded in each JSONL row; does not affect task selection",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.cmd == "run":
        run(args)


if __name__ == "__main__":
    main()
