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
"""

import statistics
import time

import click
import numpy as np

from vcztools import _vcztools

# Mix of values that exercises every branch in encode_diploid_fixed:
# 0/1 (REF/ALT calls), 2 (out-of-range diploid), -1 (missing
# sentinel), -2 (haploid sentinel paired with 0/1/2 in the b slot).
_GENOTYPE_VALUES = np.array([0, 1, 0, 0, 1, 1, 2, -1, -2, 0], dtype=np.int8)


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


def _time_one(genotypes: np.ndarray) -> float:
    t0 = time.perf_counter()
    _vcztools.encode_plink(genotypes)
    return time.perf_counter() - t0


@click.command()
@click.option("--variants", type=int, default=1000, show_default=True)
@click.option("--samples", type=int, default=100_000, show_default=True)
@click.option("--repeats", type=int, default=10, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
def main(variants: int, samples: int, repeats: int, seed: int) -> None:
    """Time vcz_encode_plink over an in-memory genotype array."""
    genotypes = _make_genotypes(variants, samples, seed)
    bytes_in = genotypes.nbytes
    bytes_out = ((samples + 3) // 4) * variants

    # One untimed call to warm caches and (on first ever call) the
    # numpy/C extension boundary.
    _vcztools.encode_plink(genotypes)

    times = [_time_one(genotypes) for _ in range(repeats)]
    median = statistics.median(times)
    fastest = min(times)
    slowest = max(times)

    in_mib = bytes_in / (1024 * 1024)
    out_mib = bytes_out / (1024 * 1024)

    click.echo(f"shape:        {variants} variants x {samples} samples")
    click.echo(f"bytes_in:     {in_mib:.1f} MiB")
    click.echo(f"bytes_out:    {out_mib:.1f} MiB")
    click.echo(f"repeats:      {repeats}")
    click.echo(
        f"time:         median {median * 1000:.1f} ms "
        f"(min {fastest * 1000:.1f}, max {slowest * 1000:.1f})"
    )
    click.echo(f"input rate:   {in_mib / median:.0f} MiB/s")
    click.echo(f"output rate:  {out_mib / median:.0f} MiB/s")


if __name__ == "__main__":
    main()
