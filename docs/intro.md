# vcztools

`vcztools` is a partial reimplementation of
[bcftools](https://samtools.github.io/bcftools/bcftools.html) for
[VCF Zarr](https://github.com/sgkit-dev/vcf-zarr-spec/).

It aims to be a drop-in replacement for a subset of bcftools functionality.
Currently supported are the `view`, `query` and `index -s/-n` commands.
We aim for 100% compatibility — if you notice a difference between the
output of vcztools and bcftools please
[open an issue](https://github.com/sgkit-dev/vcztools/issues).

To create VCZ files from VCF, see the
[bio2zarr](https://sgkit-dev.github.io/bio2zarr/) project.

Please see the [preprint](https://www.biorxiv.org/content/10.1101/2024.06.11.598241)
for more information about the VCF Zarr format.

## Development status

`vcztools` is under active development and contributions are warmly welcomed
at the [GitHub repository](https://github.com/sgkit-dev/vcztools).
