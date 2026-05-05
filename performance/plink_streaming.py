"""Micro-benchmarks for :class:`vcztools.BedEncoder`.

Exercises three patterns we expect FUSE / range-HTTP / sequential
consumers to drive:

- ``sequential``: forward drain via fixed-size ``read(off, size)`` calls
  on one encoder. Primary throughput number; one chunk iterator runs
  across the whole drain so per-call setup is amortised to zero.
- ``random``: uniform-random ``read(off, size)`` calls on one encoder;
  every read is non-sequential, so each pays one restart's worth of
  decode. Quantifies the restart cost.
- ``fanout``: N encoders sharing one ``VczReader``, each draining
  concurrently in its own thread. Tests reader-side contention; should
  scale near-linearly up to CPU contention.

Pass a path that :func:`vcztools.open_zarr` can resolve. For local
runs this is the zarr-v2 directory (``performance/bench.vcz``) or the
zarr-v3 directory (``performance/bench.vcz3``). Pass
``--backend-storage icechunk`` to drive the icechunk store.
"""

import concurrent.futures as cf
import dataclasses
import json
import logging
import pathlib
import statistics
import time

import click
import numpy as np

import vcztools
from vcztools import retrieval

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Result:
    name: str
    repeats: int
    median_seconds: float
    min_seconds: float
    max_seconds: float
    extra: dict


def _drain(enc, read_size: int) -> int:
    """Read the full BED in fixed-size fragments; return total bytes."""
    off = 0
    total = 0
    while off < enc.bed_size:
        chunk = enc.read(off, read_size)
        total += len(chunk)
        off += len(chunk)
    return total


def _bench_sequential(reader, *, read_size: int, repeats: int) -> Result:
    """Forward drain throughput: end-to-end ``read(off, size)``."""
    times = []
    bed_size = 0
    for _ in range(repeats):
        with vcztools.BedEncoder(reader) as enc:
            bed_size = enc.bed_size
            t0 = time.perf_counter()
            total = _drain(enc, read_size)
            elapsed = time.perf_counter() - t0
        times.append(elapsed)
        assert total == bed_size
    median = statistics.median(times)
    return Result(
        name="sequential",
        repeats=repeats,
        median_seconds=median,
        min_seconds=min(times),
        max_seconds=max(times),
        extra={
            "bed_size": bed_size,
            "read_size": read_size,
            "bytes_per_sec": bed_size / median,
        },
    )


def _bench_random(reader, *, read_size: int, repeats: int, seed: int) -> Result:
    """Uniform-random ``read(off, size)`` calls on one encoder.

    Every read is non-sequential, so each pays one restart's worth of
    decode (one chunk's bytes pulled and the prefix discarded).
    """
    rng = np.random.default_rng(seed)
    times = []
    with vcztools.BedEncoder(reader) as enc:
        bed_size = enc.bed_size
        for _ in range(repeats):
            high = max(1, bed_size - read_size)
            off = int(rng.integers(0, high))
            t0 = time.perf_counter()
            out = enc.read(off, read_size)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            assert len(out) == min(read_size, bed_size - off)
    return Result(
        name="random",
        repeats=repeats,
        median_seconds=statistics.median(times),
        min_seconds=min(times),
        max_seconds=max(times),
        extra={"read_size": read_size, "bed_size": bed_size},
    )


def _bench_fanout(reader, *, read_size: int, n_encoders: int, repeats: int) -> Result:
    """N encoders sharing one ``VczReader``, each draining concurrently.

    Aggregate throughput should scale near-linearly with ``n_encoders``
    until CPU contention kicks in — there's no per-encoder reader
    construction cost to bottleneck on.
    """
    times = []
    bed_size = 0
    for _ in range(repeats):
        encoders = [vcztools.BedEncoder(reader) for _ in range(n_encoders)]
        bed_size = encoders[0].bed_size
        try:
            t0 = time.perf_counter()
            with cf.ThreadPoolExecutor(max_workers=n_encoders) as pool:
                totals = list(pool.map(lambda e: _drain(e, read_size), encoders))
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            for total in totals:
                assert total == bed_size
        finally:
            for enc in encoders:
                enc.close()
    median = statistics.median(times)
    aggregate_bytes = bed_size * n_encoders
    return Result(
        name="fanout",
        repeats=repeats,
        median_seconds=median,
        min_seconds=min(times),
        max_seconds=max(times),
        extra={
            "n_encoders": n_encoders,
            "read_size": read_size,
            "bed_size": bed_size,
            "aggregate_bytes_per_sec": aggregate_bytes / median,
        },
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
    help="Repeats for the random benchmark.",
)
@click.option(
    "--read-size",
    type=int,
    default=1 << 17,
    show_default=True,
    help="Per-call read size in bytes (default 128 KiB).",
)
@click.option(
    "--fanout",
    type=int,
    default=4,
    show_default=True,
    help="Number of encoders sharing one reader in the fanout benchmark.",
)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--output", type=click.Path(path_type=pathlib.Path), default=None)
def main(dataset, backend_storage, repeats, read_size, fanout, seed, output):
    """Benchmark BedEncoder against DATASET."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = vcztools.open_zarr(dataset, backend_storage=backend_storage)
    with retrieval.VczReader(root) as reader:
        num_samples = int(reader.sample_ids.size)
        bed_size = 3 + reader.num_variants * ((num_samples + 3) // 4)
        results = [
            _bench_sequential(reader, read_size=read_size, repeats=1),
            _bench_random(reader, read_size=read_size, repeats=repeats, seed=seed),
            _bench_fanout(reader, read_size=read_size, n_encoders=fanout, repeats=1),
        ]
        report = {
            "dataset": str(dataset),
            "backend_storage": backend_storage,
            "num_variants": reader.num_variants,
            "num_samples": num_samples,
            "bed_size": bed_size,
            "results": [dataclasses.asdict(r) for r in results],
        }
    payload = json.dumps(report, indent=2)
    if output is not None:
        output.write_text(payload + "\n")
    click.echo(payload)


if __name__ == "__main__":
    main()
