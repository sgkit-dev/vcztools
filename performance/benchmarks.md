# vcztools — current benchmark numbers

Snapshot of the post-perf-work benchmark matrix.

## Setup

- **Source run:** `performance/reports/run-readerkwargs.jsonl`, recorded 2026-04-27 against the per-pipeline-executor implementation (final state of the `perf` branch).
- **Hardware:** 4-CPU host `claude-worker1`.
- **Dataset:** 100 000 diploid samples × 496 414 variants, chunk shape `(1000, 10000)`. Synthetic deterministic fields (`np.arange`- derived `variant_DP`, `variant_QUAL`, `variant_IMPACT`, `call_DP`, `call_GQ`); record counts asserted against `bench.vcz.benchmark_meta.json`.
- **Matrix:** 11 tasks × 6 backends × 3 repeats = 198 runs. All record-count assertions passed.
- **Reproduce:**
  ```
  uv run python performance/benchmarks.py run \
      --dataset performance/data/bench.vcz \
      --output performance/results/run.jsonl
  uv run python performance/benchmarks.py compare \
      performance/reports/run-phase0-baseline.jsonl \
      performance/results/run.jsonl
  ```

## Headline gains vs the 2026-04-23 baseline

| Task / Backend | Baseline | Current | Speedup |
|---|---:|---:|---:|
| `first_variant_chunks` / icechunk | 62 MiB/s | **1280 MiB/s** | **20.6×** |
| `first_variant_chunks` / local-zip | 62 MiB/s | **1211 MiB/s** | **19.4×** |
| `first_samples_chunk` / icechunk¹ | 17.2 s | **4.06 s** | **4.2× wall, ~42× per byte** |
| `first_samples_chunk` / local-zip¹ | 19.4 s | **6.05 s** | **3.2× wall, ~32× per byte** |

¹ `first_samples_chunk` was reshaped during the perf round: the baseline took 1 000 samples (~947 MB output); the current task takes the full 10 000-sample chunk (~9.47 GiB). Per-byte throughput is the fair comparison; wall-time is given alongside for reference.

## Median elapsed time (seconds)

| Task | icechunk | local-dir | local-dir-zv3 | local-http | local-zip | obstore-file |
|---|---:|---:|---:|---:|---:|---:|
| `iter_no_fields` | **0.002** | 0.007 | 0.006 | 0.015 | 0.006 | 0.007 |
| `iter_info_only` | **5.16** | 13.36 | 13.25 | 26.80 | 25.54 | 13.37 |
| `region_info_and_format` | 1.27 | 1.68 | 1.47 | 1.49 | 1.59 | **1.35** |
| `region_variant_position` | **0.045** | 0.057 | 0.052 | 0.082 | 0.053 | 0.056 |
| `region_and_sample_subset` | 0.040 | 0.040 | 0.038 | 0.050 | 0.042 | **0.037** |
| `region_filter_format_gq_gt_50` | **0.611** | 0.799 | 0.727 | 0.787 | 0.793 | 0.733 |
| `first_samples_chunk` | 4.06 | 4.01 | **3.76** | 6.21 | 6.05 | 4.57 |
| `first_samples_chunk_slice` | 4.17 | 4.05 | **3.94** | 6.36 | 6.29 | 4.51 |
| `first_variant_chunks` | **0.745** | 0.787 | 0.749 | 0.849 | 0.788 | 0.765 |
| `filter_info_dp_gt_80` | **2.51** | 6.11 | 5.33 | 12.54 | 10.61 | 6.55 |
| `filter_info_dp_gt_80_genotypes` | **5.80** | 11.28 | 9.73 | 22.55 | 24.72 | 10.61 |

## Median throughput (MiB/s of returned data)

Tasks whose output is below ~1 MB are omitted (throughput is dominated by metadata overhead there).

| Task | Output | icechunk | local-dir | local-dir-zv3 | local-http | local-zip | obstore-file |
|---|---:|---:|---:|---:|---:|---:|---:|
| `first_samples_chunk` | 9.47 GiB | 2330 | 2362 | **2520** | 1526 | 1565 | 2073 |
| `first_samples_chunk_slice` | 9.47 GiB | 2270 | 2338 | **2401** | 1489 | 1505 | 2099 |
| `first_variant_chunks` | 953 MiB | **1280** | 1212 | 1273 | 1123 | 1211 | 1247 |
| `region_info_and_format` | 473 MiB | **374** | 281 | 322 | 317 | 298 | 351 |
| `region_filter_format_gq_gt_50` | 95 MiB | **155** | 119 | 130 | 120 | 119 | 129 |
| `region_and_sample_subset` | 1.9 MiB | 48 | 48 | 50 | 38 | 45 | **51** |
| `filter_info_dp_gt_80_genotypes` | 188 MiB | **32** | 17 | 19 | 8 | 8 | 18 |

## CPU/wall ratio (4 cores available; ceiling ≈ 4)

| Task | icechunk | local-dir | local-dir-zv3 | local-http | local-zip | obstore-file |
|---|---:|---:|---:|---:|---:|---:|
| `first_samples_chunk` | **3.06** | 2.85 | 2.86 | 2.08 | 1.74 | 2.82 |
| `first_samples_chunk_slice` | **3.04** | 2.82 | 2.78 | 2.05 | 1.72 | 2.73 |
| `filter_info_dp_gt_80_genotypes` | **2.69** | 1.62 | 1.62 | 1.16 | 0.77 | 1.83 |
| `first_variant_chunks` | **2.46** | 2.17 | 2.24 | 2.02 | 2.08 | 2.20 |
| `region_info_and_format` | **1.38** | 1.20 | 1.24 | 1.13 | 1.12 | 1.25 |
| `region_and_sample_subset` | **1.46** | 1.36 | 1.35 | 1.19 | 1.17 | **1.46** |
| `region_filter_format_gq_gt_50` | **1.12** | 1.08 | 1.07 | 1.06 | 1.02 | 1.10 |
| `filter_info_dp_gt_80` | **1.21** | 0.90 | 0.70 | 0.74 | 0.41 | 1.03 |
| `iter_info_only` | **1.20** | 0.95 | 0.77 | 0.75 | 0.43 | 1.09 |
icechunk saturates ~76 % of 4 cores on the headline bulk task (`first_samples_chunk`) and ~67 % on the bulk-filter task. **`local-zip` caps at ~1.7×** on bulk tasks — the zip file's single-handle random-access cap. Re-packaging as `local-dir-zv3` or `icechunk` recovers the full 4-core utilisation on the same hardware.

## Peak RSS (MB)

Bulk tasks land in the ~600 MB – 2 GB range. Readahead budget is 256 MiB of in-flight prefetched blocks; the rest is assembled chunks held briefly plus interpreter / Zarr overhead. Well within single-machine budgets.

| Task | icechunk | local-dir | local-dir-zv3 | local-http | local-zip | obstore-file |
|---|---:|---:|---:|---:|---:|---:|
| `region_info_and_format` | 2153 | 2033 | 2069 | 2120 | 2084 | 2113 |
| `region_filter_format_gq_gt_50` | 1137 | 1116 | 1122 | 1133 | 1127 | 1142 |
| `first_variant_chunks` | 1113 | 1192 | 1164 | 1179 | 1156 | 1122 |
| `region_variant_position` | 935 | 935 | 935 | 935 | 935 | 935 |
| `region_and_sample_subset` | 737 | 736 | 736 | 737 | 737 | 737 |
| `iter_info_only` / `iter_no_fields` | 735 | 734 | 734 | 735 | 736 | 735 |
| `first_samples_chunk` | 634 | 654 | 540 | 644 | 651 | 647 |
| `first_samples_chunk_slice` | 653 | 657 | 639 | 649 | 650 | 653 |
| `filter_info_dp_gt_80_genotypes` | 626 | 612 | 616 | 610 | 603 | 615 |
| `filter_info_dp_gt_80` | 164 | 124 | 126 | 141 | 153 | 147 |

## Backend ranking (geomean median elapsed across all 11 tasks)

| Rank | Backend | Geomean elapsed (s) |
|:---:|---|---:|
| 1 | **icechunk** | 0.589 |
| 2 | local-dir-zv3 | 0.803 |
| 3 | obstore-file | 0.868 |
| 4 | local-dir | 0.877 |
| 5 | local-zip | 1.124 |
| 6 | local-http | 1.299 |

icechunk's lead over the other directory-store backends comes from its async-friendly storage layout; `local-zip`'s last-place finish on the geomean is dominated by the small-read tasks (`iter_info_only`, `filter_info_dp_gt_80`) where the single file handle serialises everything.

## Output benchmarks (write_vcf / write_plink / BedEncoder)

These three tasks measure how fast vcztools produces output bytes —
headline metric is `output_rate_mib_s = bytes_written / elapsed_s`.
They run on the `local-dir` backend only because the encoder is the
bottleneck and the input store has negligible influence at these
rates. Per-task chunk counts are sized so each task lands ~6–10 s
wall on the standard 100k-sample dataset; the three encoder rates
differ by ~2× so a shared chunk count would over- or under-spend.

- **Source run:** `performance/benchmarks.py run --task output_vcf
  --task output_plink --task output_bed_stream` against
  `performance/bench.vcz`, recorded 2026-05-06.
- **Hardware:** 4-CPU host `claude-worker1`.
- **Dataset:** 100 000 diploid samples, ~98 % biallelic (the rest
  dropped via per-chunk materialisation in
  `_biallelic_first_chunks_plan` so all three tasks see the same
  variant set).
- **Reproduce:**
  ```
  uv run python performance/benchmarks.py run \
      --dataset performance/bench.vcz \
      --task output_vcf --task output_plink --task output_bed_stream \
      --output /tmp/output.jsonl
  ```

| Task | Scope | bytes_written | elapsed | **output rate** |
|---|---|---:|---:|---:|
| `output_vcf` | 1 variant-chunk → /dev/null | 4.82 GiB | 8.3 s | **110 MiB/s** |
| `output_plink` | 20 variant-chunks → tempdir | 472 MiB | 6.6 s | **72 MiB/s** |
| `output_bed_stream` | 20 variant-chunks, sequential drain | 469 MiB | 6.2 s | **76 MiB/s** |

### `vcz_encode_plink` kernel ceiling

The pure C kernel, exercised in isolation by
`performance/encode_plink_bench.py` on an in-memory genotype array
(no I/O, no readahead, single chunk of 1000 variants × 100 000
samples × 2):

| | rate |
|---|---:|
| Input bytes processed | 2265 MiB/s |
| Output bytes produced | **283 MiB/s** |

So the kernel has ~3.7× more headroom than the pipeline currently
uses; further wall-time gains on `output_plink` /
`output_bed_stream` come from the read pipeline (Zarr decompression,
`_assemble_call`), not from the encoder.

### Optimisation history

Two passes against the 2026-05-06 baseline took the headline
PLINK output rate from 41 MiB/s to ~75 MiB/s (1.8×):

| Stage | output_plink | output_bed_stream | Change |
|---|---:|---:|---|
| Original (RMW kernel) | 41 MiB/s | 42 MiB/s | — |
| Unrolled kernel | 61 MiB/s | 60 MiB/s | rewrote `vcz_encode_plink` to pack 4 codes per byte and write once; deleted the initial `memset`. C tottime 4.39 s → 2.08 s. |
| Fused assembly | **72 MiB/s** | **76 MiB/s** | Folded the variant-axis fancy-index slice into `_assemble_call` so the call_genotype path materialises the assembled output once instead of concat-then-slice. Saved ~64 ms per chunk. |
