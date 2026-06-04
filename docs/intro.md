# vcztools

`vcztools` is a partial reimplementation of
[bcftools](https://samtools.github.io/bcftools/bcftools.html) for
[VCF Zarr](https://github.com/sgkit-dev/vcf-zarr-spec/).

It aims to be a drop-in replacement for a subset of bcftools functionality.
Currently supported are the {ref}`view<cmd-vcztools-view>`,
{ref}`query<cmd-vcztools-query>` and {ref}`index<cmd-vcztools-index>` `-s/-n`
commands.
We aim for 100% compatibility — if you notice a difference between the
output of vcztools and bcftools please
[open an issue](https://github.com/sgkit-dev/vcztools/issues).

In addition to the bcftools-shaped commands, `vcztools` ships
non-bcftools format exporters for common downstream tools:

- {ref}`view-plink<cmd-vcztools-view-plink>` — PLINK 1 binary
  (`.bed`/`.bim`/`.fam`); see {ref}`sec-plink`.
- {ref}`view-bgen<cmd-vcztools-view-bgen>` — Oxford BGEN
  (`.bgen`/`.sample`/`.bgen.bgi`); see {ref}`sec-bgen`.

To create VCZ files from VCF, see the
[bio2zarr](https://sgkit-dev.github.io/bio2zarr/) project.

Please see the [GigaScience paper](https://doi.org/10.1093/gigascience/giaf049)
and the [specification repo](https://github.com/sgkit-dev/vcf-zarr-spec/)
for more information about the VCF Zarr format.

## Development status

`vcztools` is under active development and contributions are warmly welcomed
at the [GitHub repository](https://github.com/sgkit-dev/vcztools).
