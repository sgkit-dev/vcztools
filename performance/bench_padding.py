"""Microbenchmark for BgenEncoder's per-variant padding-bytes
construction.

Per variant the padding field is ``b"." + pad_byte * (slack - 1)``,
where ``slack`` is the leftover budget after the four actual-length
string fields. The current implementation builds these via a Python
list comprehension; this script measures alternative numpy-vectorised
constructions on representative chunk sizes.

Run: ``uv run python performance/bench_padding.py``.
"""

from __future__ import annotations

import time

import numpy as np


def pad_bytes_loop(n: int, slack: np.ndarray, pad_byte: bytes) -> np.ndarray:
    """Current implementation: build per-variant ``bytes`` in Python,
    wrap as object array, cast to ``S{max_slack}``."""
    if n == 0:
        return np.zeros(0, dtype="S1")
    padding_list = [b"." + pad_byte * (int(s) - 1) for s in slack]
    obj = np.array(padding_list, dtype=object)
    max_slack = max(int(slack.max()), 1)
    return obj.astype(f"S{max_slack}")


def pad_bytes_numpy_fill_then_zero(
    n: int, slack: np.ndarray, pad_byte: bytes
) -> np.ndarray:
    """Allocate ``(n, max_slack)`` uint8 prefilled with ``pad_byte``,
    NUL out the tail beyond each row's ``slack``, set column 0 to ``"."``,
    then view as ``S{max_slack}``."""
    if n == 0:
        return np.zeros(0, dtype="S1")
    max_slack = max(int(slack.max()), 1)
    out = np.full((n, max_slack), pad_byte[0], dtype=np.uint8)
    col = np.arange(max_slack)
    # NUL out positions beyond each row's slack.
    out[col[None, :] >= slack[:, None]] = 0
    out[:, 0] = ord(".")
    return np.ascontiguousarray(out).view(f"S{max_slack}").reshape(n)


def pad_bytes_numpy_zero_then_fill(
    n: int, slack: np.ndarray, pad_byte: bytes
) -> np.ndarray:
    """Allocate ``(n, max_slack)`` uint8 of zeros, set in-bounds
    positions to ``pad_byte``, then set column 0 to ``"."``."""
    if n == 0:
        return np.zeros(0, dtype="S1")
    max_slack = max(int(slack.max()), 1)
    out = np.zeros((n, max_slack), dtype=np.uint8)
    col = np.arange(max_slack)
    in_bounds = col[None, :] < slack[:, None]
    out[in_bounds] = pad_byte[0]
    out[:, 0] = ord(".")
    return np.ascontiguousarray(out).view(f"S{max_slack}").reshape(n)


def pad_bytes_numpy_where(n: int, slack: np.ndarray, pad_byte: bytes) -> np.ndarray:
    """``np.where`` on the in-bounds mask, then patch column 0."""
    if n == 0:
        return np.zeros(0, dtype="S1")
    max_slack = max(int(slack.max()), 1)
    col = np.arange(max_slack)
    in_bounds = col[None, :] < slack[:, None]
    out = np.where(in_bounds, pad_byte[0], np.uint8(0)).astype(np.uint8)
    out[:, 0] = ord(".")
    return np.ascontiguousarray(out).view(f"S{max_slack}").reshape(n)


# --- correctness check -----------------------------------------------------


def _reference(n: int, slack: np.ndarray, pad_byte: bytes) -> list[bytes]:
    """Pure-Python ground truth: each variant's expected bytes."""
    return [b"." + pad_byte * (int(s) - 1) for s in slack]


def check(impl, label: str, n: int, slack: np.ndarray, pad_byte: bytes) -> None:
    got = impl(n, slack, pad_byte)
    expected = _reference(n, slack, pad_byte)
    for i in range(n):
        assert bytes(got[i]) == expected[i], (
            f"{label}: variant {i} got {bytes(got[i])!r} expected {expected[i]!r}"
        )


# --- benchmark driver ------------------------------------------------------


def bench(impl, n: int, slack: np.ndarray, pad_byte: bytes, repeats: int) -> float:
    """Return median seconds per call over ``repeats`` invocations."""
    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        impl(n, slack, pad_byte)
        timings.append(time.perf_counter() - t0)
    timings.sort()
    return timings[len(timings) // 2]


def make_slack(
    n: int, total_string_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Synthetic slack distribution: simulate a chunk where chrom is
    1-2 bytes, alleles are SNPs (1 byte each), and variant_id is rs +
    a 4-9 digit number. used = chrom + a1 + a2 + variant_id is roughly
    in [6, 15], so slack = total - used is in [49, 58] for the default
    total=64."""
    chrom_len = rng.integers(1, 3, size=n)
    a1_len = np.ones(n, dtype=np.int64)
    a2_len = np.ones(n, dtype=np.int64)
    variant_id_len = rng.integers(4, 10, size=n)  # e.g. rs1234..rs123456789
    used = chrom_len + a1_len + a2_len + variant_id_len
    slack = total_string_length - used
    return slack.astype(np.int64)


def main() -> None:
    rng = np.random.default_rng(0)
    pad_byte = b"."
    impls = [
        ("loop", pad_bytes_loop),
        ("numpy_fill_then_zero", pad_bytes_numpy_fill_then_zero),
        ("numpy_zero_then_fill", pad_bytes_numpy_zero_then_fill),
        ("numpy_where", pad_bytes_numpy_where),
    ]
    # Cross-check against the reference once on a small input.
    slack_small = make_slack(50, 64, rng)
    for label, impl in impls:
        check(impl, label, 50, slack_small, pad_byte)
    print("correctness: all implementations agree on a 50-row sample\n")

    sizes = [1_000, 10_000, 100_000]
    total_string_length = 64
    print(f"{'n':>10} | " + " | ".join(f"{label:>22}" for label, _ in impls))
    print("-" * (12 + len(impls) * 25))
    for n in sizes:
        slack = make_slack(n, total_string_length, rng)
        row = []
        for _, impl in impls:
            # Warm one call before timing.
            impl(n, slack, pad_byte)
            t = bench(impl, n, slack, pad_byte, repeats=11)
            row.append(f"{t * 1e3:>20.3f} ms")
        print(f"{n:>10} | " + " | ".join(row))


if __name__ == "__main__":
    main()
