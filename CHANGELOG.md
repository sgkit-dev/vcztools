# Changelog

## [0.2.0] - 2026-XX-XX

New output formats:

- Add `view-plink` command for PLINK 1 binary output (.bed/.bim/.fam),
  following the semantics of `plink2 --vcf X --make-bed`. Use
  `--no-bim`/`--no-fam` to suppress sidecars.
- Add `view-bgen` command for Oxford BGEN output, streamed to stdout or
  written with `.bgen.bgi`/`.sample` sidecars (`-o STEM`). Supports
  haploid and mixed-ploidy data.

Cloud and remote storage:

- Read VCZ from cloud and remote stores via the optional `[obstore]` and
  `[icechunk]` extras (includes Azure); fsspec and HTTP stores are also
  supported.
- Add support for `.vcz.zip` files (#280).
- Add `vcztools.open_zarr` for opening local, remote, and zipped stores.

Filtering and CLI:

- Support `-R/--regions-file` and `-T/--targets-file` (#268).
- Add `-v/--types`, `-V/--exclude-types`, `-m/--min-alleles`, and
  `-M/--max-alleles` to `view` and `view-plink`, matching bcftools view,
  along with a new `N_ALT` filter identifier.
- Add `N_MISSING` and `F_MISSING` filter variables, e.g.
  `-i 'F_MISSING < 0.05'`.
- Add `--fill-tags` to recompute `INFO/AC` and `INFO/AN`.
- Add `--log-level` and `--log-file` options; report throughput in MiB/s.

Data types:

- Add support for float16 and float64 data stored in Zarr (#413, #414).
- Add support for 64-bit integer POS and INFO/FORMAT fields.

Python API:

- Add a public Python API, documented under {ref}`sec-python-api`:
  - `VczReader` for variant data access, with sample selection by ID via
    `set_samples(sample_ids, complement=, ignore_missing_samples=)` (or by
    raw index via `set_sample_indexes`).
  - One-shot writers `write_plink` / `write_bgen` and streaming
    `BedEncoder` / `BgenEncoder` byte encoders.
  - Sidecar writers `write_bim` / `write_fam` / `write_sample` /
    `write_bgi`.
  - Click option bundles (`ViewBgenOptions`, `ViewPlinkOptions`,
    `SelectionOptions`, `ZarrStoreOptions`, `ReaderOptions`, `LogOptions`)
    and `GroupedCommand` for downstream CLI reuse.
  - Array-sentinel helpers `is_missing` / `is_fill` / `trim_fill` for
    interpreting missing and end-of-vector (fill) values in iterated arrays.

Platform and packaging:

- Add Windows support.
- Provide prebuilt wheels for Linux, macOS, and Windows (CPython 3.11-3.13).

Documentation:

- Add a documentation website covering installation, storage backends, the
  CLI, PLINK/BGEN output, and the Python API.

Deprecations:

- `--zarr-backend-storage` is deprecated in favour of `--backend-storage`;
  the old name still works and emits a warning.

Bug fixes:

- vcztools query silently truncated output on multi-chunk stores (#283)
- vcztools view crash on FILTER expressions on multi-chunk stores (#282)
- Incorrect output on vcztools query on FORMAT fields (#286, #287)
- Per-allele INFO field + -s crashes (#295)
- VCF header emitted even when filter expression is invalid (#221)
- Incorrect behaviour for bcftools query with FORMAT scoped filters
  and sample subsetting (#297)
- Incorrect output for filtering with missing Number=A INFO field + numeric (#299)
- Noise on stderr when performing arithmetic with missing values (#301)
- Null samples included in plink output (#310)
- Fill values handled incorrectly by query (#415)

## [0.1.2] - 2026-03-02

- Fix a dependency declaration in pyproject.

## [0.1.1] - 2026-03-02

Minor maintenance and bugfix release.

Features:

- Add -N/--disable-automatic-newline option (#261)
- Support -S/--samples-file in query (#264)
- Ignore missing samples (#258)

Bug fixes:

- Fix region edge cases and improve test coverage (#262). Region queries or views were in
some cases omitting variants that should have been returned.

Breaking:

- Require NumPy 2 (#249)
- Require Zarr Python version 3.1 or greater (#259)

## [0.1.0] - 2025-05-29

Improvements:

- Support filtering by FILTER (#217), CHROM (#223) and general string values (#220)
- Support regions (-r/-t), filter expressions (-i/-e) and samples (-s) in query command  (#205)
- Various improvements to support VCZ datasets produced from tskit and plink files by bio2zarr.
- Use a fully dynamically generated header via ``vcf_meta_information`` attributes
(#208). Requires vcf-zarr version >= 0.4 (bio2zarr >= 0.1.6) to fully recover the original
header.
- Add --version (#197)

Breaking:

- Update minimum Click version to 8.2.0 (#206)

## [0.0.2] - 2025-04-04

Important bugfixes for filtering language and sample subsetting.

- Clarify the implementation status of the filtering mini-lanuage in
  view/query. Version 0.0.1 contained several data-corrupting bugs,
  including incorrect missing data handling (#163), incorrect
  matching on FILTER (#164) and CHROM (#178) columns, and
  incorrect per-sample filtering in query (#179). These issues
  have been resolved by raising informative errors on aspects
  of the query language that are not implemented correctly.

- The filtering mini-language now consists of arbitrary arithmetic
  expressions on 1-dimensional fields.

- Add support for specifying samples via -s/-S options

## [0.0.1] - 2025-02-05

Initial release of vcztools
