(sec-vcf)=
# VCF

The output of {ref}`vcztools view<cmd-vcztools-view>`
and {func}``vcztools.write_vcf` aims to be 100\% compatible with
bcftools. There are a few minor caveats, which this page documents.

## Format details

- **Uncompressed only.** The output is plain text VCF; compressed or binary
  suffixes (`.gz`, `.bgz`, `.bcf`) are rejected. Pipe through `bgzip` or
  `bcftools` if you need bgzipped VCF or BCF.

- **Missing data handling.**

## See also

- {ref}`The view flag reference<cmd-vcztools-view>`.
- [VCF specification](https://samtools.github.io/hts-specs/VCFv4.3.pdf).
