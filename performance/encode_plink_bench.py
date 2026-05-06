"""Microbenchmark for the ``vcz_encode_plink`` C kernel.

Allocates a synthetic ``call_genotype``-shaped int8 buffer in memory
and times repeated calls to ``_vcztools.encode_plink`` in isolation,
with no I/O or readahead pipeline involved. Gives the upper bound
the matrix benchmarks (``output_plink`` / ``output_bed_stream``)
can asymptotically approach.

Run from the repo root::

    uv run python performance/encode_plink_bench.py

Defaults to one variant-chunk's worth of work on the standard bench
shape (1000 variants x 100000 samples). Override with ``--variants``
and ``--samples``; ``--repeats`` controls the number of timed calls.

``--threads`` and ``--block-bytes`` exercise the same variant-axis
slicing strategy used by :class:`vcztools.plink.BedEncoder` for
parallel encoding. ``--sweep`` runs the canonical ``(threads,
block_bytes)`` matrix at the chosen sample count and prints a
markdown table of MiB/s.
"""

import concurrent.futures as cf
import statistics
import time

import click
import numpy as np

from vcztools import _vcztools, plink

# Mix of values that exercises every branch in encode_diploid_fixed:
# 0/1 (REF/ALT calls), 2 (out-of-range diploid), -1 (missing
# sentinel), -2 (haploid sentinel paired with 0/1/2 in the b slot).
_GENOTYPE_VALUES = np.array([0, 1, 0, 0, 1, 1, 2, -1, -2, 0], dtype=np.int8)

_SWEEP_THREADS = (1, 2, 4, 8)
_SWEEP_BLOCK_BYTES = (1 << 20, 4 << 20, 10 << 20, 40 << 20)


def _make_genotypes(num_variants: int, num_samples: int, seed: int) -> np.ndarray:
    """Return a (V, S, 2) int8 array with a realistic value mix.

    Uses ``np.tile`` over a fixed pattern, with a per-row roll keyed
    by the seed so consecutive variants don't share the exact same
    layout (avoids the inner branch becoming trivially predictable
    in a way real data never is)."""
    rng = np.random.default_rng(seed)
    flat = np.empty(num_variants * num_samples * 2, dtype=np.int8)
    pattern = _GENOTYPE_VALUES
    for j in range(num_variants):
        offset = int(rng.integers(0, pattern.size))
        rolled = np.roll(pattern, offset)
        row_size = num_samples * 2
        repeats = (row_size + pattern.size - 1) // pattern.size
        row = np.tile(rolled, repeats)[:row_size]
        flat[j * row_size : (j + 1) * row_size] = row
    return flat.reshape(num_variants, num_samples, 2)


def _encode_parallel(
    G: np.ndarray,
    executor: cf.ThreadPoolExecutor,
    block_variants: int,
) -> bytes:
    futures = [
        executor.submit(plink._encode_genotypes_sync, G[start : start + block_variants])
        for start in range(0, G.shape[0], block_variants)
    ]
    out = bytearray()
    for f in futures:
        out.extend(f.result())
    return bytes(out)


def _time_one_sync(G: np.ndarray) -> float:
    t0 = time.perf_counter()
    _vcztools.encode_plink(G)
    return time.perf_counter() - t0


def _time_one_parallel(
    G: np.ndarray,
    executor: cf.ThreadPoolExecutor,
    block_variants: int,
) -> float:
    t0 = time.perf_counter()
    _encode_parallel(G, executor, block_variants)
    return time.perf_counter() - t0


def _median_rate(times: list[float], in_mib: float) -> float:
    return in_mib / statistics.median(times)


def _sweep(num_variants: int, num_samples: int, repeats: int, seed: int) -> None:
    G = _make_genotypes(num_variants, num_samples, seed)
    in_mib = G.nbytes / (1024 * 1024)
    out_mib = ((num_samples + 3) // 4) * num_variants / (1024 * 1024)

    click.echo("# encode_plink parallel sweep")
    click.echo(
        f"# shape: {num_variants} variants x {num_samples} samples "
        f"(in {in_mib:.1f} MiB, out {out_mib:.1f} MiB), repeats={repeats}"
    )
    click.echo("")

    headers = ["threads"] + [f"{b // (1 << 20)} MiB" for b in _SWEEP_BLOCK_BYTES]
    click.echo("| " + " | ".join(headers) + " |")
    click.echo("|" + "|".join(["---"] * len(headers)) + "|")

    # Untimed warmup.
    _vcztools.encode_plink(G)

    for threads in _SWEEP_THREADS:
        row = [str(threads)]
        if threads == 1:
            times = [_time_one_sync(G) for _ in range(repeats)]
            rate = _median_rate(times, in_mib)
            row.extend([f"{rate:.0f}"] * len(_SWEEP_BLOCK_BYTES))
        else:
            with cf.ThreadPoolExecutor(
                max_workers=threads,
                thread_name_prefix="vcztools-encode-plink-bench",
            ) as executor:
                for block_bytes in _SWEEP_BLOCK_BYTES:
                    block_variants = max(1, block_bytes // (num_samples * 2))
                    times = [
                        _time_one_parallel(G, executor, block_variants)
                        for _ in range(repeats)
                    ]
                    rate = _median_rate(times, in_mib)
                    row.append(f"{rate:.0f}")
        click.echo("| " + " | ".join(row) + " |")
    click.echo("")
    click.echo("# Cell values are input-MiB/s (median of repeats).")


@click.command()
@click.option("--variants", type=int, default=1000, show_default=True)
@click.option("--samples", type=int, default=100_000, show_default=True)
@click.option("--repeats", type=int, default=10, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option(
    "--threads",
    type=int,
    default=1,
    show_default=True,
    help="Worker threads. 1 = synchronous (existing single-thread bench).",
)
@click.option(
    "--block-bytes",
    type=int,
    default=10 * 1024 * 1024,
    show_default=True,
    help="Input-bytes target per sub-block. Ignored when --threads=1.",
)
@click.option(
    "--sweep",
    is_flag=True,
    help="Run the canonical (threads, block_bytes) matrix and emit a "
    "markdown table of MiB/s.",
)
def main(
    variants: int,
    samples: int,
    repeats: int,
    seed: int,
    threads: int,
    block_bytes: int,
    sweep: bool,
) -> None:
    """Time vcz_encode_plink over an in-memory genotype array."""
    if sweep:
        _sweep(variants, samples, repeats, seed)
        return

    genotypes = _make_genotypes(variants, samples, seed)
    bytes_in = genotypes.nbytes
    bytes_out = ((samples + 3) // 4) * variants

    # One untimed call to warm caches and (on first ever call) the
    # numpy/C extension boundary.
    _vcztools.encode_plink(genotypes)

    if threads <= 1:
        times = [_time_one_sync(genotypes) for _ in range(repeats)]
    else:
        block_variants = max(1, block_bytes // (samples * 2))
        with cf.ThreadPoolExecutor(
            max_workers=threads,
            thread_name_prefix="vcztools-encode-plink-bench",
        ) as executor:
            times = [
                _time_one_parallel(genotypes, executor, block_variants)
                for _ in range(repeats)
            ]

    median = statistics.median(times)
    fastest = min(times)
    slowest = max(times)

    in_mib = bytes_in / (1024 * 1024)
    out_mib = bytes_out / (1024 * 1024)

    click.echo(f"shape:        {variants} variants x {samples} samples")
    click.echo(f"bytes_in:     {in_mib:.1f} MiB")
    click.echo(f"bytes_out:    {out_mib:.1f} MiB")
    click.echo(f"threads:      {threads}")
    if threads > 1:
        block_variants = max(1, block_bytes // (samples * 2))
        click.echo(
            f"block_bytes:  {block_bytes / (1024 * 1024):.1f} MiB "
            f"({block_variants} variants/block)"
        )
    click.echo(f"repeats:      {repeats}")
    click.echo(
        f"time:         median {median * 1000:.1f} ms "
        f"(min {fastest * 1000:.1f}, max {slowest * 1000:.1f})"
    )
    click.echo(f"input rate:   {in_mib / median:.0f} MiB/s")
    click.echo(f"output rate:  {out_mib / median:.0f} MiB/s")


if __name__ == "__main__":
    main()
