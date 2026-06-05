# Introduction

`vcztools` provides
[bcftools](https://samtools.github.io/bcftools/bcftools.html)-compatible
querying and format conversion for VCF Zarr (VCZ).
Please see the [GigaScience paper](https://doi.org/10.1093/gigascience/giaf049)
and the [specification repo](https://github.com/sgkit-dev/vcf-zarr-spec/)
for more information about the VCF Zarr format.
To create VCZ files from VCF and other formats, see the
[bio2zarr](https://sgkit-dev.github.io/bio2zarr/) project.

Vcztools provides Python APIs to efficiently {ref}`read VCZ data
<sec-reading-vcz>` from local files and
remote {ref}`object stores <sec-storage-backends>` using
bcftools-compatible region and variant filtering syntax.
It aims to provide a drop-in replacement for a subset of bcftools functionality.
Currently supported are the {ref}`view<cmd-vcztools-view>`,
{ref}`query<cmd-vcztools-query>` and {ref}`index<cmd-vcztools-index>` `-s/-n`
commands; see {ref}`sec-bcftools-emulation` for worked examples.
We aim for 100% compatibility — if you notice a difference between the
output of vcztools and bcftools please
[open an issue](https://github.com/sgkit-dev/vcztools/issues).

In addition to the bcftools-shaped commands, `vcztools`
provides format exporters for common downstream tools:

- {ref}`view-plink<cmd-vcztools-view-plink>` — PLINK 1 binary
  (`.bed`/`.bim`/`.fam`); see the {ref}`PLINK 1 file format page<sec-plink>` and
  the {ref}`conversion walkthrough<sec-plink-conversion>`.
- {ref}`view-bgen<cmd-vcztools-view-bgen>` — Oxford BGEN
  (`.bgen`/`.sample`/`.bgen.bgi`); see the {ref}`BGEN file format page<sec-bgen>`
  and the {ref}`conversion walkthrough<sec-bgen-conversion>`.

## Development status

`vcztools` is under active development and contributions are warmly welcomed
at the [GitHub repository](https://github.com/sgkit-dev/vcztools).
