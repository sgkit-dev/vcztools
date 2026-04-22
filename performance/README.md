# vcztools performance benchmark suite

Drives `VczReader` through a catalogue of 10 iteration, subset, region
and filter tasks across 4 store backends (local directory, local zip,
local HTTP, obstore-over-file). Designed for "diff this branch against
that branch on my laptop" — not for longitudinal tracking.

## Install

The benchmark extras are in a dedicated dependency group so CI-test
users are not forced to install `psutil` / `aiohttp` / `requests`:

```
uv sync --group benchmark
```

## Generate a dataset

```
uv run python performance/generate_dataset.py \
    --num-samples 10000 --seq-length 1e6 --seed 42 \
    --output /tmp/bench.vcz
```

This runs msprime → `bio2zarr.tskit.convert` (direct, no VCF round
trip), augments the resulting store with deterministic synthetic fields
(`variant_DP`, `variant_QUAL`, `variant_IMPACT`, `call_DP`, `call_GQ`)
built from `np.arange` patterns, and writes a sibling `.zip` archive
plus a `benchmark_meta.json` file holding exact expected record counts
per filter task.

For a tiny smoke test use `--num-samples 1000 --seq-length 200000`.

## Run the benchmarks

```
uv run python performance/benchmarks.py run \
    --dataset /tmp/bench.vcz \
    --repeats 3 \
    --output performance/results/run-a.jsonl
```

Optional flags:

- `--task NAME` (repeatable) — restrict to specific tasks.
- `--backend NAME` (repeatable) — restrict to specific backends
  (`local-dir`, `local-zip`, `local-http`, `obstore-file`).
- `--skip-backend NAME` (repeatable) — exclude a backend, e.g. on a
  Python build without `aiohttp` the `local-http` backend will fail and
  can be skipped.
- `--profile {small,large}` — tag recorded in each JSONL row; does not
  change task behaviour.

Every row has its `records` field checked against the exact value in
`benchmark_meta.json`; a silently broken filter fails loudly rather
than passing as a speed win.

## Compare two runs

```
uv run python performance/compare_runs.py \
    performance/results/run-a.jsonl \
    performance/results/run-b.jsonl
```

Prints a table of `A (s)`, `B (s)`, `B/A` ratio per `(task, backend)`
and a geometric mean across all rows. Exits non-zero if the
`(task, backend)` sets differ between the two runs, or if any shared
pair disagrees on `records`.

## Known platform limits

- `local-http` relies on `fsspec[http]` (which pulls `aiohttp`). The
  benchmark dependency group installs it; on a fresh env without that
  group the HTTP backend fails at open time.
- `obstore-file` requires the `obstore` package; same workaround.

## Files

- `generate_dataset.py` — simulate + augment + write meta.
- `benchmarks.py` — task catalogue and runner.
- `compare_runs.py` — diff two JSONL runs.
- `results/` — JSONL outputs (gitignored).
