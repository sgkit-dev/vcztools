"""Benchmark suite for vcztools.

One click entrypoint with three subcommands — ``generate`` (simulate a
dataset), ``run`` (execute the task/backend matrix), and ``compare``
(diff two JSONL runs). The dataset lives at ``performance/data/bench.vcz``
by default so every subcommand can be run without arguments.

The generator produces several on-disk copies of the same logical data
so we can benchmark different storage backends apples-to-apples:

- ``bench.vcz/`` — Zarr v2 directory (bio2zarr default output)
- ``bench.vcz3/`` — Zarr v3 directory (mirrored from the v2 copy)
- ``bench.vcz.zip`` — Zarr v2 zip
- ``bench.vcz.icechunk/`` — icechunk repo mirrored from the v3 copy
- ``bench.vcz.benchmark_meta.json`` — exact expected record counts

Aiohttp / obstore / icechunk are assumed installed via the ``benchmark``
dependency group.
"""

import contextlib
import dataclasses
import json
import logging
import math
import pathlib
import shutil
import socket
import subprocess
import sys
import threading
import time

import bio2zarr.tskit
import bio2zarr.zarr_utils as zarr_utils
import click
import icechunk as ic
import msprime
import numpy as np
import pandas as pd
import psutil
import requests
import tqdm
import zarr
from zarr.codecs import BloscCodec, BloscShuffle

from vcztools import bcftools_filter, retrieval
from vcztools import regions as regions_mod
from vcztools import utils as vcz_utils

logger = logging.getLogger(__name__)


DEFAULT_DATASET = pathlib.Path("performance/data/bench.vcz")

# Simulation constants. Keep pop_size/mutation_rate deterministic across
# runs — the synthetic INFO/FORMAT values are derived from np.arange and
# the expected record counts are closed-form, so neither depends on the
# exact variants produced by msprime. Only seq_length and num_samples
# influence how much data there is.
POPULATION_SIZE = 10_000
MUTATION_RATE = 1e-7
RECOMBINATION_RATE = 1e-8

IMPACT_CYCLE = np.array(
    ["HIGH"] + ["MOD"] * 4 + ["LOW"] * 15,
    dtype="<U8",
)
assert IMPACT_CYCLE.shape == (20,)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    num_samples: int
    seq_length: float
    seed: int


def _simulate(spec: DatasetSpec):
    """Run msprime with deterministic parameters and return a tskit
    TreeSequence. ``spec.num_samples`` is the diploid sample count."""
    ts = msprime.sim_ancestry(
        samples=spec.num_samples,
        sequence_length=spec.seq_length,
        recombination_rate=RECOMBINATION_RATE,
        population_size=POPULATION_SIZE,
        random_seed=spec.seed,
    )
    ts = msprime.sim_mutations(ts, rate=MUTATION_RATE, random_seed=spec.seed)
    return ts


def _build_vcz(
    ts,
    vcz_path: pathlib.Path,
    *,
    variants_chunk_size=None,
    samples_chunk_size=None,
    worker_processes: int = 0,
):
    return bio2zarr.tskit.convert(
        ts,
        str(vcz_path),
        mode="r+",
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
        worker_processes=worker_processes,
        show_progress=True,
    )


def _variant_dp_values(num_variants: int) -> np.ndarray:
    return (np.arange(num_variants) % 101).astype(np.int16)


def _variant_qual_values(num_variants: int) -> np.ndarray:
    return (np.arange(num_variants) % 101).astype(np.float32)


def _variant_impact_values(num_variants: int) -> np.ndarray:
    return IMPACT_CYCLE[np.arange(num_variants) % IMPACT_CYCLE.size]


def _delete_if_present(root, name: str) -> None:
    if name in root:
        del root[name]


def _add_variant_array(root, name, data, *, chunks):
    zarr_utils.create_group_array(
        root,
        name,
        data=data,
        shape=data.shape,
        dtype=data.dtype.str,
        chunks=chunks,
        compressor=zarr_utils.DEFAULT_COMPRESSOR_CONFIG,
        dimension_names=["variants"],
    )


def _call_slab(v0: int, v1: int, num_samples: int, modulo: int, dtype) -> np.ndarray:
    """Compute ``(arange(v0*S, v1*S) % modulo).reshape(v1-v0, S)`` in the
    requested dtype. Works per variant-chunk so peak memory is
    ``(v1-v0) * num_samples * itemsize``."""
    flat = np.arange(v0 * num_samples, v1 * num_samples)
    return (flat % modulo).reshape(v1 - v0, num_samples).astype(dtype)


def _add_chunked_call_array(
    root,
    name: str,
    *,
    num_variants: int,
    num_samples: int,
    variants_chunk: int,
    samples_chunk: int,
    modulo: int,
    dtype,
) -> None:
    """Create a 2-D (variants, samples) array and fill it one variant-chunk
    at a time, never materialising the full (V, S) buffer in memory."""
    zarr_utils.create_empty_group_array(
        root,
        name,
        shape=(num_variants, num_samples),
        dtype=np.dtype(dtype).str,
        chunks=(variants_chunk, samples_chunk),
        compressor=zarr_utils.DEFAULT_COMPRESSOR_CONFIG,
        dimension_names=["variants", "samples"],
    )
    arr = root[name]
    starts = range(0, num_variants, variants_chunk)
    for v0 in tqdm.tqdm(starts, desc=f"augment {name}", unit="chunk"):
        v1 = min(v0 + variants_chunk, num_variants)
        arr[v0:v1, :] = _call_slab(v0, v1, num_samples, modulo, dtype)


def _augment_vcz(root) -> None:
    """Append deterministic synthetic fields to an already-open VCZ."""
    num_variants = int(root["variant_position"].shape[0])
    num_samples = int(root["sample_id"].shape[0])
    variants_chunk = int(root["variant_position"].chunks[0])
    samples_chunk = int(root["sample_id"].chunks[0])

    _add_variant_array(
        root,
        "variant_DP",
        _variant_dp_values(num_variants),
        chunks=(variants_chunk,),
    )

    _delete_if_present(root, "variant_QUAL")
    _add_variant_array(
        root,
        "variant_QUAL",
        _variant_qual_values(num_variants),
        chunks=(variants_chunk,),
    )

    _add_variant_array(
        root,
        "variant_IMPACT",
        _variant_impact_values(num_variants),
        chunks=(variants_chunk,),
    )

    _add_chunked_call_array(
        root,
        "call_DP",
        num_variants=num_variants,
        num_samples=num_samples,
        variants_chunk=variants_chunk,
        samples_chunk=samples_chunk,
        modulo=101,
        dtype=np.int16,
    )
    _add_chunked_call_array(
        root,
        "call_GQ",
        num_variants=num_variants,
        num_samples=num_samples,
        variants_chunk=variants_chunk,
        samples_chunk=samples_chunk,
        modulo=100,
        dtype=np.int8,
    )


# ---------------------------------------------------------------------------
# Mirroring between store formats
# ---------------------------------------------------------------------------


def _copy_array_chunked(src, dst, *, desc: str) -> None:
    """Copy src into dst one variant-chunk (axis-0) slice at a time, keeping
    peak memory bounded to a single chunk row."""
    v_chunk = src.chunks[0]
    starts = range(0, src.shape[0], v_chunk)
    for v0 in tqdm.tqdm(starts, desc=desc, unit="chunk", leave=False):
        v1 = min(v0 + v_chunk, src.shape[0])
        dst[v0:v1] = src[v0:v1]


_SHUFFLE_NAME = {
    0: BloscShuffle.noshuffle,
    1: BloscShuffle.shuffle,
    2: BloscShuffle.bitshuffle,
}


def _v3_compressor_for(src) -> tuple:
    """Translate the source array's numcodecs Blosc compressor (v2) into
    a native Zarr v3 ``BloscCodec``, so on-disk compression behaviour is
    comparable when benchmarking v2 vs v3 directory stores. Fall back to
    the bio2zarr default if the source has no compressor."""
    if src.compressors:
        src_cfg = zarr_utils.get_compressor_config(src)
        return (
            BloscCodec(
                cname=src_cfg["cname"],
                clevel=src_cfg["clevel"],
                shuffle=_SHUFFLE_NAME[src_cfg["shuffle"]],
                blocksize=src_cfg.get("blocksize", 0),
            ),
        )
    default = zarr_utils.DEFAULT_COMPRESSOR_CONFIG
    return (
        BloscCodec(
            cname=default["cname"],
            clevel=default["clevel"],
            shuffle=_SHUFFLE_NAME[default["shuffle"]],
            blocksize=default.get("blocksize", 0),
        ),
    )


def _mirror_group(src_root, dst_root, *, label: str) -> None:
    """Copy every array and attr from ``src_root`` into ``dst_root``.

    The destination is always written as Zarr v3 (that's how the v3
    directory and icechunk destinations are configured), so v2 numcodecs
    compressors on the source are translated to native v3 ``BloscCodec``.
    Filters are dropped on the way across: the only filter in a bio2zarr
    v2 VCZ is ``VLenUTF8`` on string arrays, which v3 handles natively
    via the ``StringDType``/``T`` dtype.

    ``label`` is the outer-stage tag ("mirror_zv3" or "mirror_icechunk")
    used as the tqdm description prefix for each array being copied.
    """
    for k, v in dict(src_root.attrs).items():
        dst_root.attrs[k] = v

    arrays = list(src_root.arrays())
    for name, src in tqdm.tqdm(arrays, desc=label, unit="array"):
        dim_names = vcz_utils.array_dims(src)
        dst = dst_root.create_array(
            name=name,
            shape=src.shape,
            chunks=src.chunks,
            dtype=src.dtype,
            dimension_names=tuple(dim_names) if dim_names is not None else None,
            compressors=_v3_compressor_for(src),
        )
        for k, v in dict(src.attrs).items():
            dst.attrs[k] = v
        _copy_array_chunked(src, dst, desc=f"  {name}")


def _mirror_to_v3_dir(src_root, dest_path: pathlib.Path) -> None:
    if dest_path.exists():
        shutil.rmtree(dest_path)
    store = zarr.storage.LocalStore(dest_path)
    dst_root = zarr.create_group(store, zarr_format=3, overwrite=True)
    _mirror_group(src_root, dst_root, label="mirror_zv3")


def _mirror_to_icechunk(src_root, dest_path: pathlib.Path) -> None:
    if dest_path.exists():
        shutil.rmtree(dest_path)
    dest_path.mkdir(parents=True, exist_ok=True)
    storage = ic.Storage.new_local_filesystem(str(dest_path))
    repo = ic.Repository.create(storage)
    session = repo.writable_session("main")
    dst_root = zarr.create_group(session.store, zarr_format=3, overwrite=True)
    _mirror_group(src_root, dst_root, label="mirror_icechunk")
    session.commit("benchmark snapshot")


# ---------------------------------------------------------------------------
# Exact expected-record-count helpers
# ---------------------------------------------------------------------------


def _count_by_arange_modulo(num_variants: int, modulo: int, threshold: int) -> int:
    full_cycles = num_variants // modulo
    tail = num_variants % modulo
    per_cycle = max(0, modulo - 1 - threshold)
    tail_hits = max(0, tail - 1 - threshold)
    return full_cycles * per_cycle + tail_hits


def _count_call_threshold(
    variant_indices, num_samples: int, modulo: int, threshold: int
) -> int:
    """Variant-level count for ``value = (v*num_samples + s) % modulo > threshold``
    (bcftools semantics: variant kept if any sample passes).

    Iterates over an explicit list of variant ordinals so the same helper
    serves both full-scan filters and region-restricted filters.
    """
    if num_samples >= modulo:
        return len(variant_indices)
    count = 0
    for v in variant_indices:
        start = (int(v) * num_samples) % modulo
        if start + num_samples <= modulo:
            if start + num_samples - 1 > threshold:
                count += 1
        else:
            count += 1
    return count


def _default_region_spec(root, fraction: float):
    """Return a ``(contig, start, end)`` tuple covering ``fraction`` of the
    variants on the first contig, centred on the median."""
    contig_ids = root["contig_id"][:].tolist()
    contig = contig_ids[0]
    variant_contig = root["variant_contig"][:]
    variant_position = root["variant_position"][:]
    positions = np.sort(variant_position[variant_contig == 0])
    half = fraction / 2.0
    lo_idx = int(len(positions) * (0.5 - half))
    hi_idx = max(lo_idx + 1, int(len(positions) * (0.5 + half)))
    start = int(positions[lo_idx])
    end = int(positions[min(hi_idx, len(positions)) - 1])
    return (contig, start, end)


def _region_variant_indices(root, region_spec) -> np.ndarray:
    contig_name, start, end = region_spec
    contig_ids = root["contig_id"][:].tolist()
    contig_idx = contig_ids.index(contig_name)
    variant_contig = root["variant_contig"][:]
    variant_position = root["variant_position"][:]
    mask = (
        (variant_contig == contig_idx)
        & (variant_position >= start)
        & (variant_position <= end)
    )
    return np.flatnonzero(mask)


def _build_meta(root, spec: DatasetSpec, region_spec) -> dict:
    num_variants = int(root["variant_position"].shape[0])
    num_samples = int(root["sample_id"].shape[0])
    region_indices = _region_variant_indices(root, region_spec)
    full_indices = range(num_variants)

    dp_gt_80 = _count_by_arange_modulo(num_variants, 101, 80)
    region_gq_gt_50 = _count_call_threshold(region_indices, num_samples, 100, 50)

    variants_chunk = int(root["variant_position"].chunks[0])
    first_variant_chunks_records = min(
        FIRST_VARIANT_CHUNKS_COUNT * variants_chunk, num_variants
    )

    return {
        "num_samples": num_samples,
        "num_variants": num_variants,
        "seed": spec.seed,
        "seq_length": spec.seq_length,
        "region_spec": {
            "contig": region_spec[0],
            "start": region_spec[1],
            "end": region_spec[2],
        },
        "region_variant_count": int(region_indices.size),
        "expected_records": {
            "iter_no_fields": num_variants,
            "iter_info_only": num_variants,
            "region_info_and_format": int(region_indices.size),
            "first_samples_chunk": num_variants,
            "first_variant_chunks": first_variant_chunks_records,
            "region_variant_position": int(region_indices.size),
            "filter_info_dp_gt_80": dp_gt_80,
            "region_filter_format_gq_gt_50": region_gq_gt_50,
            "region_and_sample_subset": int(region_indices.size),
            # Sanity check: the full-scan equivalents, computed once so a
            # user curious about the "what if we did scan everything" cost
            # can enable them manually via a task-list edit without
            # regenerating the dataset.
            "full_filter_format_gq_gt_50": _count_call_threshold(
                full_indices, num_samples, 100, 50
            ),
        },
    }


def _meta_path(vcz_path: pathlib.Path) -> pathlib.Path:
    return vcz_path.parent / (vcz_path.name + ".benchmark_meta.json")


def _zip_path(vcz_path: pathlib.Path) -> pathlib.Path:
    return vcz_path.parent / (vcz_path.name + ".zip")


def _v3_path(vcz_path: pathlib.Path) -> pathlib.Path:
    return vcz_path.parent / (vcz_path.name + "3")


def _icechunk_path(vcz_path: pathlib.Path) -> pathlib.Path:
    return vcz_path.parent / (vcz_path.name + ".icechunk")


@contextlib.contextmanager
def _stage(name: str):
    t0 = time.perf_counter()
    logger.info("stage %s: start", name)
    yield
    logger.info("stage %s: done in %.2fs", name, time.perf_counter() - t0)


def _generate(
    *,
    num_samples: int,
    seq_length: float,
    seed: int,
    output: pathlib.Path,
    variants_chunk_size: int | None,
    samples_chunk_size: int | None,
    region_fraction: float,
    worker_processes: int,
) -> dict:
    spec = DatasetSpec(num_samples=num_samples, seq_length=seq_length, seed=seed)

    with _stage("simulate"):
        ts = _simulate(spec)
    if ts.num_sites == 0:
        raise RuntimeError(
            f"simulation produced 0 variants for spec={spec}; "
            f"increase seq_length or sample count"
        )

    if output.exists():
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with _stage("convert"):
        root = _build_vcz(
            ts,
            output,
            variants_chunk_size=variants_chunk_size,
            samples_chunk_size=samples_chunk_size,
            worker_processes=worker_processes,
        )
    with _stage("augment"):
        _augment_vcz(root)

    with _stage("mirror_zv3"):
        _mirror_to_v3_dir(root, _v3_path(output))

    # Reopen the v3 dir read-only as the source for the icechunk mirror
    # so we're writing data that is byte-stable on disk.
    with _stage("zip"):
        zip_path = _zip_path(output)
        if zip_path.exists():
            zip_path.unlink()
        zarr_utils.zip_zarr(str(output), str(zip_path))

    with _stage("mirror_icechunk"):
        v3_root = zarr.open(_v3_path(output), mode="r")
        _mirror_to_icechunk(v3_root, _icechunk_path(output))

    region_spec = _default_region_spec(root, region_fraction)
    meta = _build_meta(root, spec, region_spec)
    _meta_path(output).write_text(json.dumps(meta, indent=2, sort_keys=True))
    logger.info("wrote %s", _meta_path(output))
    return meta


def _load_meta(vcz_path: pathlib.Path) -> dict:
    return json.loads(_meta_path(vcz_path).read_text())


# ---------------------------------------------------------------------------
# Backend openers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class OpenedStore:
    root: zarr.Group
    cleanup: object  # callable(); no-op for file backends, tears down HTTP


class BackendOpener:
    name: str = ""

    def open(self, vcz_path: pathlib.Path) -> OpenedStore:
        raise NotImplementedError


class LocalDirOpener(BackendOpener):
    name = "local-dir"

    def open(self, vcz_path):
        root = vcz_utils.open_zarr(str(vcz_path))
        return OpenedStore(root, lambda: None)


class LocalDirZv3Opener(BackendOpener):
    name = "local-dir-zv3"

    def open(self, vcz_path):
        root = vcz_utils.open_zarr(str(_v3_path(vcz_path)))
        return OpenedStore(root, lambda: None)


class LocalZipOpener(BackendOpener):
    name = "local-zip"

    def open(self, vcz_path):
        root = vcz_utils.open_zarr(str(_zip_path(vcz_path)))
        return OpenedStore(root, lambda: None)


class ObstoreFileOpener(BackendOpener):
    name = "obstore-file"

    def open(self, vcz_path):
        root = vcz_utils.open_zarr(str(vcz_path), zarr_backend_storage="obstore")
        return OpenedStore(root, lambda: None)


class IcechunkOpener(BackendOpener):
    name = "icechunk"

    def open(self, vcz_path):
        root = vcz_utils.open_zarr(
            str(_icechunk_path(vcz_path)), zarr_backend_storage="icechunk"
        )
        return OpenedStore(root, lambda: None)


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

        return OpenedStore(root, cleanup)

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
        LocalDirZv3Opener(),
        LocalZipOpener(),
        LocalHttpOpener(),
        ObstoreFileOpener(),
        IcechunkOpener(),
    )
}


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Task:
    name: str
    build_reader: object
    fields: list[str] | None


def _build_iter_no_fields(root, meta):
    return retrieval.VczReader(root)


def _build_iter_info_only(root, meta):
    return retrieval.VczReader(root)


def _build_region_info_and_format(root, meta):
    reader = retrieval.VczReader(root)
    rs = meta["region_spec"]
    region = f"{rs['contig']}:{rs['start']}-{rs['end']}"
    plan = regions_mod.build_chunk_plan(root, regions=region)
    reader.set_variants(plan)
    return reader


def _build_first_samples_chunk(root, meta):
    """Read the first 1000 samples across every variant. At the default
    ``samples_chunk_size=10000`` this lands entirely inside one
    samples-chunk, so the task measures the backend's throughput for
    pulling a complete samples-chunk across the full variants axis."""
    reader = retrieval.VczReader(root)
    num_samples = int(root["sample_id"].shape[0])
    take = min(1000, num_samples)
    reader.set_samples(list(range(take)))
    return reader


FIRST_VARIANT_CHUNKS_COUNT = 5


def _build_first_variant_chunks(root, meta):
    """Read the first ``FIRST_VARIANT_CHUNKS_COUNT`` variant-chunks
    across every sample. The chunk count is tuned so the returned
    volume roughly matches ``first_samples_chunk`` at the default
    dataset shape, giving two directly comparable data-rate readings
    for different I/O patterns — contiguous variants region vs
    contiguous samples column."""
    reader = retrieval.VczReader(root)
    num_variants = int(root["variant_position"].shape[0])
    variants_chunk = int(root["variant_position"].chunks[0])
    num_variant_chunks = math.ceil(num_variants / variants_chunk)
    take = min(FIRST_VARIANT_CHUNKS_COUNT, num_variant_chunks)
    plan = [vcz_utils.ChunkRead(index=i) for i in range(take)]
    reader.set_variants(plan)
    return reader


def _build_region_variant_position(root, meta):
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


def _build_region_filter_format_gq_gt_50(root, meta):
    reader = retrieval.VczReader(root)
    rs = meta["region_spec"]
    region = f"{rs['contig']}:{rs['start']}-{rs['end']}"
    plan = regions_mod.build_chunk_plan(root, regions=region)
    reader.set_variants(plan)
    vf = bcftools_filter.BcftoolsFilter(
        field_names=reader.field_names, include="FMT/GQ>50"
    )
    # Force the filter to see every sample's GQ, even though we'll never
    # emit them — this exercises the "full GQ scan, region only" path.
    reader.set_variant_filter(vf, filter_on_subset_samples=False)
    return reader


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


TASKS: list[Task] = [
    Task("iter_no_fields", _build_iter_no_fields, []),
    Task(
        "iter_info_only",
        _build_iter_info_only,
        [
            "variant_position",
            "variant_contig",
            "variant_length",
            "variant_DP",
            "variant_QUAL",
            "variant_IMPACT",
        ],
    ),
    Task(
        "region_info_and_format",
        _build_region_info_and_format,
        [
            "variant_position",
            "variant_contig",
            "variant_length",
            "variant_DP",
            "variant_QUAL",
            "call_DP",
            "call_GQ",
            "call_genotype",
        ],
    ),
    Task("first_samples_chunk", _build_first_samples_chunk, ["call_genotype"]),
    Task("first_variant_chunks", _build_first_variant_chunks, ["call_genotype"]),
    Task(
        "region_variant_position",
        _build_region_variant_position,
        ["variant_position"],
    ),
    Task(
        "filter_info_dp_gt_80",
        _build_filter_info_dp_gt_80,
        ["variant_position", "variant_DP"],
    ),
    Task(
        "region_filter_format_gq_gt_50",
        _build_region_filter_format_gq_gt_50,
        ["variant_position"],
    ),
    Task(
        "region_and_sample_subset",
        _build_region_and_sample_subset,
        ["call_genotype"],
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Result:
    task: str
    backend: str
    num_samples: int
    num_variants: int
    repeat_index: int
    elapsed_s: float
    cpu_s: float
    records: int
    bytes_retrieved: int
    data_rate_mib_s: float
    peak_rss_mb: float
    profile: str
    git_sha: str
    timestamp: str
    hostname: str


@dataclasses.dataclass
class ReadStats:
    records: int
    bytes_retrieved: int


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


def _iterate_reader(reader: retrieval.VczReader, fields: list[str] | None) -> ReadStats:
    """Iterate the reader's variant chunks, counting records and summing
    the byte volume of every returned array.

    ``fields=[]`` is the chunk-scheduler-only probe: ``variant_chunks``
    short-circuits to an empty generator, so we walk the plan directly
    and count per-chunk variants — no array reads, zero bytes retrieved,
    but the count still matches ``num_variants``.

    For non-empty ``fields`` we iterate ``variant_chunks`` and sum
    ``arr.nbytes`` over every key. A sample-scope filter can add a
    ``sample_filter_pass`` entry (see ``retrieval.py:614-616``); it's a
    real materialised array, so its bytes are part of the retrieved
    volume.
    """
    if fields is not None and len(fields) == 0:
        chunk_size = reader.variants_chunk_size
        num_variants = reader.num_variants
        records = 0
        for cr in reader.variant_chunk_plan:
            if cr.selection is not None:
                records += int(cr.selection.size)
                continue
            start = cr.index * chunk_size
            stop = min(start + chunk_size, num_variants)
            records += stop - start
        return ReadStats(records=records, bytes_retrieved=0)

    records = 0
    total_bytes = 0
    for chunk in reader.variant_chunks(fields=fields):
        first = next(iter(chunk.values()))
        records += int(first.shape[0])
        for arr in chunk.values():
            total_bytes += int(arr.nbytes)
    return ReadStats(records=records, bytes_retrieved=total_bytes)


def _run_task(
    task: Task,
    backend: BackendOpener,
    dataset: pathlib.Path,
    meta: dict,
    repeat_index: int,
    profile: str,
    git_sha: str,
    hostname: str,
) -> Result:
    opened = backend.open(dataset)
    try:
        reader = task.build_reader(opened.root, meta)
        with PeakRssSampler() as sampler:
            wall_t0 = time.perf_counter()
            cpu_t0 = time.process_time()
            stats = _iterate_reader(reader, task.fields)
            elapsed = time.perf_counter() - wall_t0
            cpu = time.process_time() - cpu_t0
    finally:
        opened.cleanup()

    expected = int(meta["expected_records"][task.name])
    if stats.records != expected:
        raise AssertionError(
            f"task {task.name!r} on backend {backend.name!r} emitted "
            f"{stats.records} records, expected exactly {expected}"
        )

    data_rate_mib_s = (
        stats.bytes_retrieved / (1024 * 1024) / elapsed if elapsed > 0 else 0.0
    )

    return Result(
        task=task.name,
        backend=backend.name,
        num_samples=int(meta["num_samples"]),
        num_variants=int(meta["num_variants"]),
        repeat_index=repeat_index,
        elapsed_s=elapsed,
        cpu_s=cpu,
        records=stats.records,
        bytes_retrieved=stats.bytes_retrieved,
        data_rate_mib_s=data_rate_mib_s,
        peak_rss_mb=sampler.peak_mb,
        profile=profile,
        git_sha=git_sha,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        hostname=hostname,
    )


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


def _select_tasks(names: tuple[str, ...]) -> list[Task]:
    if len(names) == 0:
        return list(TASKS)
    task_by_name = {t.name: t for t in TASKS}
    missing = [n for n in names if n not in task_by_name]
    if len(missing) > 0:
        raise click.ClickException(f"unknown task(s): {missing}")
    return [task_by_name[n] for n in names]


def _select_backends(
    include: tuple[str, ...], skip: tuple[str, ...]
) -> list[BackendOpener]:
    names = list(ALL_BACKENDS) if len(include) == 0 else list(include)
    for n in names:
        if n not in ALL_BACKENDS:
            raise click.ClickException(f"unknown backend {n!r}")
    skip_set = set(skip)
    return [ALL_BACKENDS[n] for n in names if n not in skip_set]


# ---------------------------------------------------------------------------
# Compare (pandas)
# ---------------------------------------------------------------------------


def _group_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeats into one median row per (task, backend), carrying a
    single ``records`` value forward. Raises if a group disagrees on
    records internally."""
    records = df.groupby(["task", "backend"])["records"].nunique()
    bad = records[records != 1]
    if len(bad) > 0:
        raise click.ClickException(
            f"inconsistent records within a single run for: {list(bad.index)}"
        )
    med = df.groupby(["task", "backend"], as_index=False)["elapsed_s"].median()
    recs = df.groupby(["task", "backend"], as_index=False)["records"].first()
    return med.merge(recs, on=["task", "backend"])


def _compare_runs(a_path: pathlib.Path, b_path: pathlib.Path) -> int:
    a = pd.read_json(a_path, lines=True)
    b = pd.read_json(b_path, lines=True)
    a_stats = _group_medians(a)
    b_stats = _group_medians(b)

    merged = a_stats.merge(
        b_stats, on=["task", "backend"], suffixes=("_a", "_b"), how="outer"
    )
    only_a = merged[merged["elapsed_s_b"].isna()][["task", "backend"]]
    only_b = merged[merged["elapsed_s_a"].isna()][["task", "backend"]]
    if len(only_a) > 0 or len(only_b) > 0:
        parts = []
        if len(only_a) > 0:
            parts.append(f"only in A: {only_a.to_records(index=False).tolist()}")
        if len(only_b) > 0:
            parts.append(f"only in B: {only_b.to_records(index=False).tolist()}")
        click.echo("(task, backend) sets differ — " + "; ".join(parts), err=True)
        return 1

    rec_mismatch = merged[merged["records_a"] != merged["records_b"]]
    if len(rec_mismatch) > 0:
        lines = [
            f"  {r.task}/{r.backend}: A={r.records_a} B={r.records_b}"
            for r in rec_mismatch.itertuples()
        ]
        click.echo("record counts differ:\n" + "\n".join(lines), err=True)
        return 1

    merged = merged.sort_values(["task", "backend"]).reset_index(drop=True)
    merged["ratio"] = merged["elapsed_s_b"] / merged["elapsed_s_a"]

    click.echo(
        merged[["task", "backend", "elapsed_s_a", "elapsed_s_b", "ratio"]].to_string(
            index=False, float_format="%.4f"
        )
    )

    positive = (
        merged.loc[merged["ratio"] > 0, "ratio"].replace([np.inf], np.nan).dropna()
    )
    if len(positive) > 0:
        geomean = math.exp(np.log(positive).mean())
        click.echo("")
        click.echo(f"geomean ratio B/A: {geomean:.4f}x")
    return 0


# ---------------------------------------------------------------------------
# Click CLI
# ---------------------------------------------------------------------------


@click.group()
def cli():
    """vcztools benchmark suite."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@cli.command("generate")
@click.option("--num-samples", type=int, default=100_000, show_default=True)
@click.option("--seq-length", type=float, default=1e7, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option(
    "--output",
    type=click.Path(path_type=pathlib.Path),
    default=DEFAULT_DATASET,
    show_default=True,
)
@click.option("--variants-chunk-size", type=int, default=None)
@click.option("--samples-chunk-size", type=int, default=None)
@click.option(
    "--region-fraction",
    type=float,
    default=0.002,
    show_default=True,
    help="Fraction of the first contig's variants covered by the region "
    "task's region_spec (centred on the median). Default 0.2% is tuned "
    "so region_info_and_format lands inside the target band on a "
    "100k-sample dataset.",
)
@click.option(
    "--worker-processes",
    type=int,
    default=0,
    show_default=True,
    help="Worker processes for bio2zarr.tskit.convert. "
    "0 uses the main process only. Parallel workers speed up the "
    "convert stage but each pickles the tree sequence, so memory "
    "scales with the worker count.",
)
def generate_cmd(
    num_samples,
    seq_length,
    seed,
    output,
    variants_chunk_size,
    samples_chunk_size,
    region_fraction,
    worker_processes,
):
    """Simulate and write the benchmark dataset (v2 dir, v3 dir, zip, icechunk)."""
    meta = _generate(
        num_samples=num_samples,
        seq_length=seq_length,
        seed=seed,
        output=output,
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
        region_fraction=region_fraction,
        worker_processes=worker_processes,
    )
    click.echo(
        f"generated {meta['num_variants']} variants x {meta['num_samples']} samples"
    )


@cli.command("run")
@click.option(
    "--dataset",
    type=click.Path(path_type=pathlib.Path),
    default=DEFAULT_DATASET,
    show_default=True,
    help="Base path used to locate per-backend siblings "
    "(<dataset>.zip, <dataset>3/, <dataset>.icechunk/, "
    "<dataset>.benchmark_meta.json). The directory itself does not "
    "need to exist as long as the backends you request are satisfied "
    "by the sibling files.",
)
@click.option(
    "--output",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    help="JSONL output path; one row per (task, backend, repeat).",
)
@click.option("--repeats", type=int, default=3, show_default=True)
@click.option(
    "--task",
    "tasks",
    multiple=True,
    help="Restrict to these task names; may repeat. Default: all.",
)
@click.option(
    "--backend",
    "backends",
    multiple=True,
    help="Restrict to these backend names; may repeat. Default: all.",
)
@click.option(
    "--skip-backend",
    "skip_backends",
    multiple=True,
    help="Skip these backend names; may repeat.",
)
@click.option(
    "--profile",
    type=click.Choice(["small", "large"]),
    default="large",
    show_default=True,
    help="Tag recorded in each JSONL row; does not affect task selection.",
)
def run_cmd(dataset, output, repeats, tasks, backends, skip_backends, profile):
    """Execute the task x backend matrix and write a JSONL row per run."""
    meta = _load_meta(dataset)
    selected_tasks = _select_tasks(tasks)
    selected_backends = _select_backends(backends, skip_backends)

    git_sha = _discover_git_sha()
    hostname = socket.gethostname()

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as fp:
        for task in selected_tasks:
            for backend in selected_backends:
                for repeat_index in range(repeats):
                    logger.info(
                        "%s / %s [%d/%d]",
                        task.name,
                        backend.name,
                        repeat_index + 1,
                        repeats,
                    )
                    result = _run_task(
                        task,
                        backend,
                        dataset,
                        meta,
                        repeat_index,
                        profile,
                        git_sha,
                        hostname,
                    )
                    fp.write(json.dumps(dataclasses.asdict(result), sort_keys=True))
                    fp.write("\n")
                    fp.flush()


@cli.command("compare")
@click.argument("a", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("b", type=click.Path(exists=True, path_type=pathlib.Path))
def compare_cmd(a, b):
    """Diff two JSONL runs; prints a table + geomean of B/A ratios."""
    sys.exit(_compare_runs(a, b))


if __name__ == "__main__":
    cli()
