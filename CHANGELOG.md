# Changelog

## [0.1.X] - 2026-XX-XX

Features:

- Add support for .vcz.zip files (#280).
- Add `view-plink` command for PLINK 1 binary output (.bed/.bim/.fam)
  following the semantics of `plink2 --vcf X --make-bed`.
- Add `-v/--types`, `-V/--exclude-types`, `-m/--min-alleles`, and
  `-M/--max-alleles` to `view` and `view-plink`, matching bcftools
  view. The filter language exposes a new `N_ALT` identifier
  (count of non-empty ALT slots), so e.g. `-i 'N_ALT >= 2'` is now
  valid. On `view-plink`, `--max-alleles` also gains a `-M` short flag.
- Filter language exposes `N_MISSING` and `F_MISSING` calculated
  variables (count and fraction of samples with all-missing GTs per
  variant), so e.g. `-i 'F_MISSING < 0.05'` and `-i 'N_MISSING == 0'`
  are now valid. Both default to 0 on datasets without `call_genotype`.
  Other bcftools-style calculated variables (`AC`, `AF`, `MAC`,
  `MAF`, `AN`, `ILEN`, `N_SAMPLES`) now raise a dedicated
  `UnsupportedCalculatedVariableError` pointing at issue #171.
- `view`'s `--max-alleles` and `-v/-V/-m` now compose with
  sample-scope `-i 'FMT/...'` filters via axis-1 broadcasting in
  the AND combinator (previously rejected). `view-plink` still
  rejects sample-scope filters, since the .bed format has no
  per-sample channel to mask into.
- `view-bgen` now supports haploid (e.g. chrX in males, mtDNA) and
  mixed-ploidy stores. Per-sample BGEN ploidy bytes follow the
  spec: haploid calls get `0x01` and one probability byte; diploid
  calls stay at `0x02` and two probability bytes. VCZ stores haploid
  data either as `call_genotype.shape[2] == 1` or as
  `call_genotype.shape[2] == 2` with `-2` haploid-padding in slot 1;
  both forms are accepted. The fixed-size `BgenEncoder` class
  requires uniform ploidy (all haploid or all diploid) across the
  store; mixed-ploidy stores must go through `write_bgen`.
- `view-bgen` now streams the `.bgen` payload to stdout by default
  (mirroring `vcztools view`). Pass `-o STEM` to write files plus the
  `.bgen.bgi` and `.sample` sidecars; suppress sidecars individually
  with `--no-bgi` / `--no-sample-file`. New `--no-header-samples`
  clears the BGEN `SAMPLE_IDS_PRESENT` flag and omits the in-header
  sample-ID block.
- `view-plink` gains `--no-bim` / `--no-fam` to suppress those
  sidecars (`.bed` is always written).
- `view-bgen` default `--compression-level` is now `1` (was `-1` =
  zlib default ≈ 6). Hard-call BGEN payloads are low-entropy enough
  that level 1 captures most of the size win at a fraction of the CPU
  cost; level 6 only buys ~10-30% smaller files for several times
  the encode time. Pass `--compression-level 9` for archival output
  or `--compression-level 0` for stored / fastest.
- Pin `write_bim`, `write_fam`, `write_sample`, and `write_bgi` on
  the public `vcztools` API. All four take `(reader, output, ...)`
  and write to a path or file-like (`write_bgi` is SQLite-backed and
  remains path-only). They replace the internal `generate_bim`,
  `generate_fam`, `generate_sample`, `write_bgen_samples`, and
  `write_bgen_index` helpers; `write_plink`'s `write_bim`/`write_fam`
  boolean kwargs are renamed to `bim`/`fam`.
- `VczReader` now has a simple default field-resolution model
  (programmatic use): a virtual field (`AC`/`AN`/`AF`/`NS`,
  `N_MISSING`, `F_MISSING`, `N_ALT`) resolves the same way in a filter
  and in the output, so `-i 'AC>1'` and the emitted `INFO/AC` always
  agree. A sample subset makes the sample-dependent virtual fields
  recompute to reflect the subset (the stored full-file value would be
  stale). The bcftools CLI semantics — filters evaluate against the
  stored `INFO`, and `INFO` output is recomputed only on request — are
  opt-in via the new `VczReader.set_bcftools_semantics()`; the `view`
  and `query` CLI commands enable it, so their output is unchanged.
- `vcztools view-plink` and `view-bgen` evaluate `-i/-e` filters over the
  selected sample subset (native `VczReader` semantics): with a `-s`/`-S`
  subset, `AC`/`AN`/`AF`/`NS` in a filter expression are recomputed for
  that subset instead of read from the file's stored (full-cohort) INFO.
  This differs from `view`/`query` (which follow bcftools and filter on
  stored INFO) and suits the export-to-analysis workflow — filters reflect
  the analysis cohort, and only the selected samples are read. See the
  PLINK / BGEN output documentation pages.
- Add support for float16 and float64 data stored in Zarr (#413, #414).

Breaking changes:

- `VczReader.set_filter_samples()` is removed. The only behaviour it
  was used for — bcftools `view`'s pre-subset sample-scope filter axis
  — is now reached via `set_bcftools_semantics(full_sample_filter=True)`.
  Direct `VczReader` callers that filtered or read virtual fields under
  a sample subset get the new default model described above unless they
  call `set_bcftools_semantics()`.
- `view-bgen` and `view-plink`: the `--out PREFIX` flag has been
  replaced by `-o/--output STEM`. The stem is now taken **verbatim**
  rather than going through `pathlib.with_suffix`, so `-o sample.bgen`
  produces `sample.bgen.bgen` etc.; pass an unsuffixed stem.
  `view-bgen` no longer requires `-o` (default = stream to stdout);
  `view-plink` still requires it.
- The six exported option bundles (`ViewBgenOptions`,
  `ViewPlinkOptions`, `SelectionOptions`, `ZarrStoreOptions`,
  `ReaderOptions`, `LogOptions`) replace `pop_from_click_kwargs`
  with non-destructive `from_click_kwargs`. Construction no longer
  mutates the Click `**kwargs` dict, so downstream consumers (e.g.
  biofuse) can read the dict multiple times.
- Export `GroupedCommand` on the public `vcztools` API. Downstream
  apps mounting `ViewBgenOptions.decorator` / `ViewPlinkOptions.decorator`
  can pass `cls=GroupedCommand` to `@click.command(...)` and get the
  same grouped `--help` sections as `vcztools view-bgen` / `view-plink`.
- `--compression-level` is no longer part of `ViewBgenOptions`. It is
  now attached directly to the `view-bgen` command (alongside `-o`),
  matching `-o`'s treatment as a vcztools-CLI-only flag. Downstream
  consumers re-exposing `ViewBgenOptions.decorator` who want a
  compression knob must add their own Click option.

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
