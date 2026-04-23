# vcztools performance benchmark suite

Drives `VczReader` through nine iteration, subset, region, and filter
tasks across six storage backends (Zarr v2 local directory, Zarr v3
local directory, Zarr v2 zip, local HTTP, obstore over `file://`,
icechunk). Designed for "diff this branch against that branch on my
laptop" — not for longitudinal tracking.

## Install

```
uv sync --group benchmark
```

Pulls in `msprime`, `bio2zarr`, `tskit`, `fsspec[http]`, `obstore`,
`icechunk`, `pandas`, `psutil`, `requests`.

## Generate the dataset

```
uv run python performance/benchmarks.py generate
```

Defaults: 100 000 diploid samples, sequence length 1e7 (~80–160k
sites), seed 42, written to `performance/data/bench.vcz`. Expect ~10
minutes wall time on a modern laptop, dominated by
`bio2zarr.tskit.convert` and the Zarr v3 mirror. Each stage
(`simulate`, `convert`, `augment`, `mirror_zv3`, `zip`,
`mirror_icechunk`) is timed in the log so you can bisect if it
overshoots.

The command writes five artefacts under `performance/data/`, all
holding the same logical data:

- `bench.vcz/` — Zarr v2 directory store (bio2zarr's default on-disk
  format).
- `bench.vcz3/` — Zarr v3 directory store, mirrored from the v2 copy.
- `bench.vcz.zip` — Zarr v2 zipped via `bio2zarr.zarr_utils.zip_zarr`.
- `bench.vcz.icechunk/` — icechunk repo mirrored from the v3 copy.
- `bench.vcz.benchmark_meta.json` — exact expected record counts per
  task, computed from the deterministic `np.arange`-based synthetic
  fields (`variant_DP`, `variant_QUAL`, `variant_IMPACT`, `call_DP`,
  `call_GQ`). The runner asserts every task's output against these
  values, so a silently broken filter fails loudly rather than
  passing as a speed win.

For a tiny smoke run: `generate --num-samples 1000 --seq-length 200000`.

Pass `--worker-processes N` to parallelise the `bio2zarr.tskit.convert`
stage. It defaults to 0 (main process only) because bio2zarr currently
pickles the tree sequence to each worker, so peak memory scales with
the worker count.

## Run the matrix

```
uv run python performance/benchmarks.py run \
    --output performance/results/run-a.jsonl
```

Writes one JSONL row per `(task, backend, repeat)` with `elapsed_s`,
`peak_rss_mb`, `records`, `git_sha`, `timestamp`, `hostname`, and the
`profile` tag. Defaults to `--repeats 3` across all nine tasks and all
six backends.

Optional flags:

- `--dataset PATH` — default `performance/data/bench.vcz`.
- `--task NAME` (repeatable) — restrict to specific tasks.
- `--backend NAME` (repeatable) — restrict to specific backends
  (`local-dir`, `local-dir-zv3`, `local-zip`, `local-http`,
  `obstore-file`, `icechunk`).
- `--skip-backend NAME` (repeatable) — exclude a backend.
- `--profile {small,large}` — free-form tag recorded in each row.

## Compare two runs

```
uv run python performance/benchmarks.py compare \
    performance/results/run-a.jsonl \
    performance/results/run-b.jsonl
```

Groups rows by `(task, backend)`, takes the median `elapsed_s` per
group, and prints a pandas table of `A`, `B`, and `B/A` ratio along
with a geometric-mean ratio across all rows. Exits non-zero if the
two runs disagree on:

- the set of `(task, backend)` pairs, or
- the `records` emitted for any shared pair.

## Task catalogue

Tuned for the 100k-sample default so no task iterates the full
genotype matrix. The `--region-fraction` setting on `generate` /
`prepare` (default 0.2%) sets the common region for all `region_*`
tasks.

| # | Task | Fields | Selection |
|---|------|--------|-----------|
| 1 | `iter_no_fields` | — | chunk scheduler only |
| 2 | `iter_info_only` | INFO scalars | full |
| 3 | `region_info_and_format` | INFO + FORMAT + genotypes | region |
| 4 | `subset_10_samples` | `call_genotype` | 10 samples × all variants |
| 5 | `region_variant_position` | `variant_position` | region |
| 6 | `filter_info_dp_gt_80` | INFO | full (`INFO/DP>80`) |
| 7 | `region_filter_format_gq_gt_50` | — | region + `FMT/GQ>50` with filter seeing *all* samples |
| 8 | `region_and_sample_subset` | `call_genotype` | 1% samples × region |

## Files

- `benchmarks.py` — everything above.
- `compare.py` — legacy bcftools-vs-vcztools ad-hoc script, kept for
  reference.
- `data/` — home for generated datasets. Gitignored except for the
  existing Makefile and requirements.txt.
- `results/` — JSONL outputs (gitignored).
