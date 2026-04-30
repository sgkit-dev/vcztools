"""Benchmark suite for vcztools.

One click entrypoint with four subcommands — ``generate`` (simulate a
dataset), ``run`` (execute the task/backend matrix against the generated
sibling layout), ``run-one`` (execute one task against an arbitrary Zarr
path on a single storage backend), and ``compare`` (diff two JSONL runs).
The dataset lives at ``performance/data/bench.vcz`` by default so
``generate`` and ``run`` can be invoked without arguments.

The generator produces several on-disk copies of the same logical data
so we can benchmark different storage backends apples-to-apples:

- ``bench.vcz/`` — Zarr v2 directory (bio2zarr default output)
- ``bench.vcz3/`` — Zarr v3 directory (mirrored from the v2 copy)
- ``bench.vcz.zip`` — Zarr v2 zip
- ``bench.vcz.icechunk/`` — icechunk repo mirrored from the v3 copy

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
# Region selection
# ---------------------------------------------------------------------------


def _default_region_spec(reader: retrieval.VczReader, fraction: float):
    """Return a ``(contig, start, end)`` tuple covering ``fraction`` of the
    variants on the contig of the first variant, centred on the median."""
    variant_contig = reader.root["variant_contig"][:]
    variant_position = reader.root["variant_position"][:]
    contig_idx = int(variant_contig[0])
    contig = reader.contig_ids[contig_idx]
    positions = np.sort(variant_position[variant_contig == contig_idx])
    half = fraction / 2.0
    lo_idx = int(len(positions) * (0.5 - half))
    hi_idx = max(lo_idx + 1, int(len(positions) * (0.5 + half)))
    start = int(positions[lo_idx])
    end = int(positions[min(hi_idx, len(positions)) - 1])
    return (contig, start, end)


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
    worker_processes: int,
) -> zarr.Group:
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

    with _stage("zip"):
        zip_path = _zip_path(output)
        if zip_path.exists():
            zip_path.unlink()
        zarr_utils.zip_zarr(str(output), str(zip_path))

    with _stage("mirror_icechunk"):
        v3_root = zarr.open(_v3_path(output), mode="r")
        _mirror_to_icechunk(v3_root, _icechunk_path(output))

    return root


# ---------------------------------------------------------------------------
# Backend openers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class OpenedStore:
    root: zarr.Group
    cleanup: object  # callable(); tears down any per-open resources


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
        root = vcz_utils.open_zarr(str(vcz_path), backend_storage="obstore")
        return OpenedStore(root, lambda: None)


class IcechunkOpener(BackendOpener):
    name = "icechunk"

    def open(self, vcz_path):
        root = vcz_utils.open_zarr(
            str(_icechunk_path(vcz_path)), backend_storage="icechunk"
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


# Storage-layer openers consumed by ``run-one``. Unlike the matrix
# openers above, these do not derive sibling paths — they open whatever
# string path or URL the user supplies via Zarr's chosen storage backend.


class AutoOpener(BackendOpener):
    name = "auto"

    def open(self, dataset):
        root = vcz_utils.open_zarr(str(dataset))
        return OpenedStore(root, lambda: None)


class ObstoreOpener(BackendOpener):
    name = "obstore"

    def open(self, dataset):
        root = vcz_utils.open_zarr(str(dataset), backend_storage="obstore")
        return OpenedStore(root, lambda: None)


class IcechunkUrlOpener(BackendOpener):
    name = "icechunk"

    def open(self, dataset):
        root = vcz_utils.open_zarr(str(dataset), backend_storage="icechunk")
        return OpenedStore(root, lambda: None)


RUN_ONE_BACKENDS: dict[str, BackendOpener] = {
    opener.name: opener
    for opener in (AutoOpener(), ObstoreOpener(), IcechunkUrlOpener())
}


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RunContext:
    """Per-invocation context for a benchmark run.

    Built once per ``run`` / ``run-one`` invocation by opening the dataset,
    reading sample/variant counts, and computing a region spec from the
    given fraction. Threaded through to every task builder."""

    num_samples: int
    num_variants: int
    region_spec: tuple[str, int, int]
    readahead_bytes: int | None = None
    readahead_workers: int | None = None


@dataclasses.dataclass(frozen=True)
class Task:
    name: str
    build_reader: object
    fields: list[str] | None


def _build_iter_no_fields(root, ctx):
    return retrieval.VczReader(root)


def _build_iter_info_only(root, ctx):
    return retrieval.VczReader(root)


def _build_region_info_and_format(root, ctx):
    reader = retrieval.VczReader(root)
    contig, start, end = ctx.region_spec
    region = f"{contig}:{start}-{end}"
    plan = regions_mod.build_chunk_plan(root, regions=region)
    reader.set_variants(plan)
    return reader


def _build_first_samples_chunk(root, ctx):
    """Read the full first samples-chunk across every variant. The
    selection matches the on-disk sample chunk size, so
    ``build_chunk_plan`` emits ``selection=None`` and no per-chunk
    slice is applied — the task reports the reader's delivered
    bandwidth on an intact sample chunk."""
    reader = retrieval.VczReader(root)
    take = min(reader.samples_chunk_size, reader.num_samples)
    reader.set_samples(np.arange(take))
    return reader


FMT_FIELDS_VARIANT_CHUNKS_COUNT = 60


def _fmt_fields_reader(root):
    """Shared setup for :func:`_build_fmt_fields` and
    :func:`_build_fmt_fields_filtered`: the first samples-chunk wide
    by the first ``FMT_FIELDS_VARIANT_CHUNKS_COUNT`` variant-chunks
    tall. Sized so the filter variant lands in the ~10s band on the
    slower local-zip backend (~0.17s per chunk from
    ``performance/reports/bulk-filter.md``) — full-dataset coverage
    isn't needed to capture the per-chunk filter-eval overhead, which
    is linear in chunk count."""
    reader = retrieval.VczReader(root)
    take_samples = min(reader.samples_chunk_size, reader.num_samples)
    reader.set_samples(np.arange(take_samples))

    num_variant_chunks = math.ceil(reader.num_variants / reader.variants_chunk_size)
    take_chunks = min(FMT_FIELDS_VARIANT_CHUNKS_COUNT, num_variant_chunks)
    plan = [vcz_utils.ChunkRead(index=i) for i in range(take_chunks)]
    reader.set_variants(plan)
    return reader


def _build_fmt_fields(root, ctx):
    """Read three FMT fields (``call_genotype``, ``call_GQ``,
    ``call_DP``) on the shared fmt_fields slice. Apples-to-apples
    baseline for :func:`_build_fmt_fields_filtered`: same fields,
    same coverage, no filter — the wall-time delta between the two
    is filter-eval overhead (plus the per-chunk
    ``sample_filter_pass`` mask the filter path publishes into
    ``chunk_data``)."""
    return _fmt_fields_reader(root)


def _build_fmt_fields_filtered(root, ctx):
    """Same slice as :func:`_build_fmt_fields` plus a FMT-scope filter
    that evaluates to ``True`` on every cell of the bench dataset, so
    every variant in scope survives and the output volume matches the
    unfiltered baseline. Exercises the per-genotype filter-evaluation
    path end-to-end: reads two extra call_* fields (``call_GQ``,
    ``call_DP``), evaluates four element-wise predicates combined
    with ``&&``, and materialises a per-chunk
    ``(variants_chunk, num_samples)`` pass mask."""
    reader = _fmt_fields_reader(root)
    vf = bcftools_filter.BcftoolsFilter(
        field_names=reader.field_names,
        include="FMT/GQ >= 0 && FMT/GQ <= 100 && FMT/DP >= 0 && FMT/DP <= 101",
    )
    reader.set_variant_filter(vf)
    return reader


def _build_first_samples_chunk_slice(root, ctx):
    """Same coverage as :func:`_build_first_samples_chunk` but drops
    the last sample of the chunk, routing the selection through
    ``samples.build_chunk_plan``'s ``slice(0, N-1)`` path. Throughput
    is expected within measurement noise of the full-chunk baseline;
    a regression would mean the Phase 0.5 slice optimisation is
    carrying a cost we haven't noticed."""
    reader = retrieval.VczReader(root)
    take = min(reader.samples_chunk_size, reader.num_samples)
    reader.set_samples(np.arange(take - 1))
    return reader


FIRST_VARIANT_CHUNKS_COUNT = 5


def _build_first_variant_chunks(root, ctx):
    """Read the first ``FIRST_VARIANT_CHUNKS_COUNT`` variant-chunks
    across every sample. The chunk count is tuned so the returned
    volume roughly matches ``first_samples_chunk`` at the default
    dataset shape, giving two directly comparable data-rate readings
    for different I/O patterns — contiguous variants region vs
    contiguous samples column."""
    reader = retrieval.VczReader(root)
    num_variant_chunks = math.ceil(reader.num_variants / reader.variants_chunk_size)
    take = min(FIRST_VARIANT_CHUNKS_COUNT, num_variant_chunks)
    plan = [vcz_utils.ChunkRead(index=i) for i in range(take)]
    reader.set_variants(plan)
    return reader


def _build_region_variant_position(root, ctx):
    reader = retrieval.VczReader(root)
    contig, start, end = ctx.region_spec
    region = f"{contig}:{start}-{end}"
    plan = regions_mod.build_chunk_plan(root, regions=region)
    reader.set_variants(plan)
    return reader


def _build_filter_info_dp_gt_80(root, ctx):
    reader = retrieval.VczReader(root)
    vf = bcftools_filter.BcftoolsFilter(
        field_names=reader.field_names, include="INFO/DP>80"
    )
    reader.set_variant_filter(vf)
    return reader


def _build_filter_info_dp_gt_80_genotypes(root, ctx):
    """Filter on ``INFO/DP > 80`` and emit genotypes for the first 1000
    samples. Tests the path where a variant filter gates a large
    call_* output — the shape the readahead pipeline previously
    opted out of."""
    reader = retrieval.VczReader(root)
    take = min(1000, reader.num_samples)
    reader.set_samples(np.arange(take))
    vf = bcftools_filter.BcftoolsFilter(
        field_names=reader.field_names, include="INFO/DP>80"
    )
    reader.set_variant_filter(vf)
    return reader


def _build_region_filter_format_gq_gt_50(root, ctx):
    reader = retrieval.VczReader(root)
    contig, start, end = ctx.region_spec
    region = f"{contig}:{start}-{end}"
    plan = regions_mod.build_chunk_plan(root, regions=region)
    reader.set_variants(plan)
    vf = bcftools_filter.BcftoolsFilter(
        field_names=reader.field_names, include="FMT/GQ>50"
    )
    reader.set_variant_filter(vf)
    # Force the filter to evaluate against every sample's GQ, even though
    # we'll never emit them — this exercises the "full GQ scan, region
    # only" path (bcftools-view pre-subset semantics).
    reader.set_filter_samples(reader.non_null_sample_indices)
    return reader


def _build_region_and_sample_subset(root, ctx):
    reader = retrieval.VczReader(root)
    take = max(1, reader.num_samples // 100)
    reader.set_samples(np.arange(take))
    contig, start, end = ctx.region_spec
    region = f"{contig}:{start}-{end}"
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
    Task(
        "fmt_fields",
        _build_fmt_fields,
        ["call_genotype", "call_GQ", "call_DP"],
    ),
    Task(
        "fmt_fields_filtered",
        _build_fmt_fields_filtered,
        ["call_genotype"],
    ),
    Task(
        "first_samples_chunk_slice",
        _build_first_samples_chunk_slice,
        ["call_genotype"],
    ),
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
        "filter_info_dp_gt_80_genotypes",
        _build_filter_info_dp_gt_80_genotypes,
        ["variant_position", "variant_DP", "call_genotype"],
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
    dataset,
    ctx: RunContext,
    repeat_index: int,
    profile: str,
    git_sha: str,
    hostname: str,
) -> Result:
    opened = backend.open(dataset)
    try:
        reader = task.build_reader(opened.root, ctx)
        if ctx.readahead_bytes is not None:
            logger.info(f"readahead_bytes overridden to {ctx.readahead_bytes}")
            reader.readahead_bytes = ctx.readahead_bytes
        if ctx.readahead_workers is not None:
            logger.info(f"readahead_workers overridden to {ctx.readahead_workers}")
            reader.readahead_workers = ctx.readahead_workers
        with PeakRssSampler() as sampler:
            wall_t0 = time.perf_counter()
            cpu_t0 = time.process_time()
            stats = _iterate_reader(reader, task.fields)
            elapsed = time.perf_counter() - wall_t0
            cpu = time.process_time() - cpu_t0
    finally:
        opened.cleanup()

    data_rate_mib_s = (
        stats.bytes_retrieved / (1024 * 1024) / elapsed if elapsed > 0 else 0.0
    )

    return Result(
        task=task.name,
        backend=backend.name,
        num_samples=ctx.num_samples,
        num_variants=ctx.num_variants,
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


def _build_run_context(
    dataset,
    opener: BackendOpener,
    region_fraction: float,
    *,
    readahead_bytes: int | None = None,
    readahead_workers: int | None = None,
) -> RunContext:
    """Open ``dataset`` with ``opener`` once to read sample/variant counts
    and compute a region spec at the requested fraction. Returns a frozen
    :class:`RunContext` that is then threaded through every per-(task,
    repeat) reopen."""
    # Setup-only reader: this VczReader exists to materialize counts and
    # the region spec. The timed reader is built fresh per repeat in
    # `_run_task`, so the open here doesn't warm the timing-path cache.
    opened = opener.open(dataset)
    try:
        reader = retrieval.VczReader(opened.root)
        return RunContext(
            num_samples=reader.num_samples,
            num_variants=reader.num_variants,
            region_spec=_default_region_spec(reader, region_fraction),
            readahead_bytes=readahead_bytes,
            readahead_workers=readahead_workers,
        )
    finally:
        opened.cleanup()


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


def _configure_logging(level: int) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s.%(msecs)03d %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def _verbosity_callback(ctx, param, value):
    level = {0: logging.WARNING, 1: logging.INFO}.get(value, logging.DEBUG)
    _configure_logging(level)
    return value


verbosity_option = click.option(
    "-v",
    "--verbosity",
    count=True,
    callback=_verbosity_callback,
    expose_value=False,
    help="Increase log verbosity (-v for INFO, -vv for DEBUG).",
)


@click.group()
def cli():
    """vcztools benchmark suite."""


@cli.command("generate")
@verbosity_option
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
    worker_processes,
):
    """Simulate and write the benchmark dataset (v2 dir, v3 dir, zip, icechunk)."""
    root = _generate(
        num_samples=num_samples,
        seq_length=seq_length,
        seed=seed,
        output=output,
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
        worker_processes=worker_processes,
    )
    num_variants = int(root["variant_position"].shape[0])
    num_samples = int(root["sample_id"].shape[0])
    click.echo(f"generated {num_variants} variants x {num_samples} samples")


@cli.command("run")
@verbosity_option
@click.option(
    "--dataset",
    type=click.Path(path_type=pathlib.Path),
    default=DEFAULT_DATASET,
    show_default=True,
    help="Base path used to locate per-backend siblings "
    "(<dataset>.zip, <dataset>3/, <dataset>.icechunk/). "
    "The directory itself does not need to exist as long as the "
    "backends you request are satisfied by the sibling files.",
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
def run_cmd(
    dataset,
    output,
    repeats,
    tasks,
    backends,
    skip_backends,
    profile,
    region_fraction,
):
    """Execute the task x backend matrix and write a JSONL row per run."""
    selected_tasks = _select_tasks(tasks)
    selected_backends = _select_backends(backends, skip_backends)

    ctx = _build_run_context(dataset, ALL_BACKENDS["local-dir"], region_fraction)

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
                        ctx,
                        repeat_index,
                        profile,
                        git_sha,
                        hostname,
                    )
                    fp.write(json.dumps(dataclasses.asdict(result), sort_keys=True))
                    fp.write("\n")
                    fp.flush()


_TASK_NAMES_HELP = ", ".join(t.name for t in TASKS)


@cli.command("run-one")
@verbosity_option
@click.argument("dataset", type=click.Path(path_type=str))
@click.option(
    "--task",
    "task_name",
    required=True,
    type=str,
    help=f"Task to run. Available: {_TASK_NAMES_HELP}.",
)
@click.option(
    "--output",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    help="JSONL output path; one row per repeat.",
)
@click.option(
    "--backend",
    type=click.Choice(list(RUN_ONE_BACKENDS)),
    default="auto",
    show_default=True,
    help="Storage layer Zarr uses to open the dataset.",
)
@click.option("--repeats", type=int, default=3, show_default=True)
@click.option(
    "--region-fraction",
    type=float,
    default=0.002,
    show_default=True,
    help="Fraction of the first contig's variants used by region tasks.",
)
@click.option(
    "--profile",
    type=click.Choice(["small", "large"]),
    default="large",
    show_default=True,
    help="Tag recorded in each JSONL row; does not affect task selection.",
)
@click.option(
    "--readahead-bytes",
    type=int,
    default=None,
    show_default=True,
    help="Override the readahead pipeline's byte budget. None uses the "
    "VczReader default.",
)
@click.option(
    "--readahead-workers",
    type=int,
    default=None,
    show_default=True,
    help="Override the readahead pipeline's worker count. None uses the "
    "VczReader default.",
)
def run_one_cmd(
    dataset,
    task_name,
    output,
    backend,
    repeats,
    region_fraction,
    profile,
    readahead_bytes,
    readahead_workers,
):
    """Run a single task against a single Zarr store on a single backend.

    ``dataset`` may be a local path, a directory store, a ``.zip``, or any
    URL the chosen backend can open (e.g. ``http://``, ``s3://``, an
    icechunk repo path)."""
    [task] = _select_tasks((task_name,))
    opener = RUN_ONE_BACKENDS[backend]

    ctx = _build_run_context(
        dataset,
        opener,
        region_fraction,
        readahead_bytes=readahead_bytes,
        readahead_workers=readahead_workers,
    )

    git_sha = _discover_git_sha()
    hostname = socket.gethostname()

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as fp:
        for repeat_index in range(repeats):
            logger.info("%s [%d/%d]", task.name, repeat_index + 1, repeats)
            result = _run_task(
                task,
                opener,
                dataset,
                ctx,
                repeat_index,
                profile,
                git_sha,
                hostname,
            )
            fp.write(json.dumps(dataclasses.asdict(result), sort_keys=True))
            fp.write("\n")
            fp.flush()


@cli.command("compare")
@verbosity_option
@click.argument("a", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("b", type=click.Path(exists=True, path_type=pathlib.Path))
def compare_cmd(a, b):
    """Diff two JSONL runs; prints a table + geomean of B/A ratios."""
    sys.exit(_compare_runs(a, b))


if __name__ == "__main__":
    cli()
