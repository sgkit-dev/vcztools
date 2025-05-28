# Changelog

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
