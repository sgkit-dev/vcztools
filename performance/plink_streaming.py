"""Micro-benchmarks for :class:`vcztools.PlinkStreamingSource`.

Exercises the four read patterns we expect FUSE / range-HTTP /
preview consumers to drive:

- ``stream``: forward-stream throughput (bytes/sec, time to drain
  ``stream_bed`` over the full store).
- ``tail``: ``read_tail(4096)`` p50 latency.
- ``random``: single-chunk-aligned ``read_bed`` random reads
  (one ``bytes_per_variant`` row at a uniformly-sampled offset).
- ``sparse``: ``read_variants`` over a sparse selection of N variants
  spanning at most M chunks.

Pass a path that :func:`vcztools.open_zarr` can resolve. For local
runs this is the zarr-v2 directory (``performance/bench.vcz``) or the
zarr-v3 directory (``performance/bench.vcz3``). Pass
``--backend-storage icechunk`` to drive the icechunk store.
"""

import dataclasses
import json
import logging
import pathlib
import statistics
import time

import click
import numpy as np

import vcztools

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Result:
    name: str
    repeats: int
    median_seconds: float
    min_seconds: float
    max_seconds: float
    extra: dict


def _bench_stream(src, *, chunk_size: int, repeats: int) -> Result:
    """Drain ``stream_bed`` end-to-end and report wall time per run."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        total = 0
        for fragment in src.stream_bed(chunk_size=chunk_size):
            total += len(fragment)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        assert total == src.bed_size
    bytes_per_sec = src.bed_size / statistics.median(times)
    return Result(
        name="stream",
        repeats=repeats,
        median_seconds=statistics.median(times),
        min_seconds=min(times),
        max_seconds=max(times),
        extra={
            "bed_size": src.bed_size,
            "chunk_size": chunk_size,
            "bytes_per_sec": bytes_per_sec,
        },
    )


def _bench_tail(src, *, repeats: int, nbytes: int) -> Result:
    """Measure individual ``read_tail`` calls — what a header probe
    over an HTTP range read looks like."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = src.read_tail(nbytes)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        assert len(out) == min(nbytes, src.bed_size)
    return Result(
        name="tail",
        repeats=repeats,
        median_seconds=statistics.median(times),
        min_seconds=min(times),
        max_seconds=max(times),
        extra={"nbytes": nbytes},
    )


def _bench_random_reads(src, *, repeats: int, seed: int) -> Result:
    """Single-row ``read_bed`` calls at uniformly-sampled offsets."""
    rng = np.random.default_rng(seed)
    bpv = src.bytes_per_variant
    times = []
    for _ in range(repeats):
        v = int(rng.integers(0, src.num_variants))
        offset = 3 + v * bpv
        t0 = time.perf_counter()
        out = src.read_bed(offset, bpv)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        assert len(out) == bpv
    return Result(
        name="random",
        repeats=repeats,
        median_seconds=statistics.median(times),
        min_seconds=min(times),
        max_seconds=max(times),
        extra={"bytes_per_variant": bpv},
    )


def _bench_sparse(
    src,
    *,
    repeats: int,
    n_variants: int,
    n_chunks: int,
    chunk_size_hint: int,
    seed: int,
) -> Result:
    """Sparse ``read_variants`` over ``n_variants`` indexes spread
    across ``n_chunks`` variant chunks. ``chunk_size_hint`` is the
    on-disk variant-chunk size used to pick chunk anchors.
    """
    rng = np.random.default_rng(seed)
    times = []
    for _ in range(repeats):
        max_chunk = max(0, src.num_variants // chunk_size_hint - 1)
        anchors = rng.choice(
            max_chunk + 1, size=min(n_chunks, max_chunk + 1), replace=False
        )
        per_chunk = max(1, n_variants // len(anchors))
        idx = []
        for a in anchors:
            base = int(a) * chunk_size_hint
            for i in range(per_chunk):
                idx.append(base + i)
        idx = np.array(sorted(set(idx)), dtype=np.int64)[:n_variants]
        t0 = time.perf_counter()
        out = src.read_variants(idx, strategy="sparse")
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        assert len(out) == idx.size * src.bytes_per_variant
    return Result(
        name="sparse",
        repeats=repeats,
        median_seconds=statistics.median(times),
        min_seconds=min(times),
        max_seconds=max(times),
        extra={"n_variants": n_variants, "n_chunks": n_chunks},
    )


@click.command()
@click.argument("dataset", type=click.Path(path_type=pathlib.Path))
@click.option(
    "--backend-storage",
    type=click.Choice(["fsspec", "obstore", "icechunk"]),
    default=None,
    help="Forwarded to vcztools.open_zarr.",
)
@click.option(
    "--repeats",
    type=int,
    default=5,
    show_default=True,
    help="Number of repeats for each benchmark.",
)
@click.option(
    "--stream-chunk-size",
    type=int,
    default=1 << 20,
    show_default=True,
)
@click.option("--tail-bytes", type=int, default=4096, show_default=True)
@click.option("--sparse-n-variants", type=int, default=10, show_default=True)
@click.option("--sparse-n-chunks", type=int, default=10, show_default=True)
@click.option("--sparse-chunk-size-hint", type=int, default=10000, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--output", type=click.Path(path_type=pathlib.Path), default=None)
def main(
    dataset,
    backend_storage,
    repeats,
    stream_chunk_size,
    tail_bytes,
    sparse_n_variants,
    sparse_n_chunks,
    sparse_chunk_size_hint,
    seed,
    output,
):
    """Benchmark PlinkStreamingSource against DATASET."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = vcztools.open_zarr(dataset, backend_storage=backend_storage)
    with vcztools.PlinkStreamingSource(root) as src:
        results = []
        results.append(_bench_stream(src, chunk_size=stream_chunk_size, repeats=1))
        results.append(_bench_tail(src, repeats=repeats, nbytes=tail_bytes))
        results.append(_bench_random_reads(src, repeats=repeats, seed=seed))
        results.append(
            _bench_sparse(
                src,
                repeats=repeats,
                n_variants=sparse_n_variants,
                n_chunks=sparse_n_chunks,
                chunk_size_hint=sparse_chunk_size_hint,
                seed=seed,
            )
        )

        report = {
            "dataset": str(dataset),
            "backend_storage": backend_storage,
            "num_variants": src.num_variants,
            "num_samples": src.num_samples,
            "bed_size": src.bed_size,
            "results": [dataclasses.asdict(r) for r in results],
        }

    payload = json.dumps(report, indent=2)
    if output is not None:
        output.write_text(payload + "\n")
    click.echo(payload)


if __name__ == "__main__":
    main()
