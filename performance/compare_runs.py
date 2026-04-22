"""Diff two benchmark JSONL runs.

Groups rows by ``(task, backend)``, takes the median ``elapsed_s`` per
group in each run, and prints a table of ``A``, ``B`` and ``B/A`` along
with a geometric-mean ratio across all rows.

Exits non-zero if the two runs disagree on:
- the set of ``(task, backend)`` pairs, or
- the ``records`` emitted for any shared pair.
"""

import argparse
import collections
import dataclasses
import json
import math
import pathlib
import sys


@dataclasses.dataclass
class GroupStats:
    median_elapsed: float
    records: int


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    rows = []
    with path.open() as fp:
        for line in fp:
            line = line.strip()
            if len(line) == 0:
                continue
            rows.append(json.loads(line))
    return rows


def _median(values: list[float]) -> float:
    sorted_values = sorted(values)
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 1:
        return sorted_values[mid]
    return 0.5 * (sorted_values[mid - 1] + sorted_values[mid])


def _group_rows(rows: list[dict]) -> dict[tuple[str, str], GroupStats]:
    by_key: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_key[(row["task"], row["backend"])].append(row)
    stats = {}
    for key, group in by_key.items():
        elapsed = [float(r["elapsed_s"]) for r in group]
        records_set = {int(r["records"]) for r in group}
        if len(records_set) != 1:
            raise SystemExit(
                f"inconsistent records within a single run for {key}: "
                f"{sorted(records_set)}"
            )
        stats[key] = GroupStats(
            median_elapsed=_median(elapsed),
            records=next(iter(records_set)),
        )
    return stats


def compare(a_path: pathlib.Path, b_path: pathlib.Path) -> int:
    a_rows = _load_jsonl(a_path)
    b_rows = _load_jsonl(b_path)
    a_stats = _group_rows(a_rows)
    b_stats = _group_rows(b_rows)

    a_keys = set(a_stats)
    b_keys = set(b_stats)
    if a_keys != b_keys:
        only_a = sorted(a_keys - b_keys)
        only_b = sorted(b_keys - a_keys)
        msg_parts = []
        if len(only_a) > 0:
            msg_parts.append(f"only in A: {only_a}")
        if len(only_b) > 0:
            msg_parts.append(f"only in B: {only_b}")
        print("(task, backend) sets differ — " + "; ".join(msg_parts), file=sys.stderr)
        return 1

    mismatches = []
    for key in sorted(a_keys):
        if a_stats[key].records != b_stats[key].records:
            mismatches.append(
                f"{key}: A={a_stats[key].records} B={b_stats[key].records}"
            )
    if len(mismatches) > 0:
        print("record counts differ:\n  " + "\n  ".join(mismatches), file=sys.stderr)
        return 1

    print(f"{'task':<30} {'backend':<15} {'A (s)':>10} {'B (s)':>10} {'B/A':>8}")
    ratios = []
    for key in sorted(a_keys):
        task, backend = key
        a = a_stats[key].median_elapsed
        b = b_stats[key].median_elapsed
        ratio = b / a if a > 0 else float("inf")
        ratios.append(ratio)
        print(f"{task:<30} {backend:<15} {a:>10.4f} {b:>10.4f} {ratio:>7.2f}x")

    finite_ratios = [r for r in ratios if math.isfinite(r) and r > 0]
    if len(finite_ratios) > 0:
        log_mean = sum(math.log(r) for r in finite_ratios) / len(finite_ratios)
        geomean = math.exp(log_mean)
        print()
        print(f"{'geomean':<30} {'':<15} {'':>10} {'':>10} {geomean:>7.2f}x")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=pathlib.Path, help="reference run (baseline)")
    parser.add_argument("b", type=pathlib.Path, help="new run (candidate)")
    args = parser.parse_args()
    sys.exit(compare(args.a, args.b))


if __name__ == "__main__":
    main()
