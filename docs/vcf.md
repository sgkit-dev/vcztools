(sec-vcf)=
# VCF

The {ref}`vcztools view<cmd-vcztools-view>` command writes **uncompressed VCF**
from a VCZ store, streaming to stdout by default or to a path with
`-o`/`--output`. It is a drop-in replacement for `bcftools view`.

```bash
# Streaming: pipe straight into another tool, or redirect to a file.
vcztools view sample.vcz

# File output.
vcztools view sample.vcz -o out.vcf
```

Sample, region and filter selection mirrors the bcftools flags
(`-s`/`-S`/`-r`/`-R`/`-t`/`-T`/`-i`/`-e`); the per-flag reference is in
{ref}`sec-cli-ref`.

## Format details

- **Uncompressed only.** The output is plain text VCF; compressed or binary
  suffixes (`.gz`, `.bgz`, `.bcf`) are rejected. Pipe through `bgzip` or
  `bcftools` if you need bgzipped VCF or BCF.
- **Header control.** `-h`/`--header-only` writes just the header,
  `-H`/`--no-header` suppresses it, and `--no-version` omits the appended
  version / command-line lines.
- **INFO recomputation.** With a sample subset, INFO fields are recomputed by
  default; `-I`/`--no-update` keeps the stored values, and `--fill-tags`
  (re)computes a chosen set of tags.
- **Dropping genotypes.** `-G`/`--drop-genotypes` emits a sites-only VCF.

## See also

- {ref}`The view flag reference<cmd-vcztools-view>`.
- [VCF specification](https://samtools.github.io/hts-specs/VCFv4.3.pdf).
