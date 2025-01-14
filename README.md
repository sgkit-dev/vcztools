[![CI](https://github.com/sgkit-dev/vcztools/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sgkit-dev/vcztools/actions/workflows/ci.yml)

# vcztools
Partial reimplementation of bcftools for [VCF Zarr](https://github.com/sgkit-dev/vcf-zarr-spec/)

Please see the [preprint](https://www.biorxiv.org/content/10.1101/2024.06.11.598241) for more information.


## Installation

```
python3 -m pip install vcztools
```

## Usage

```
vcztools view <path.vcz>
```
or 
```
python -m vcztools view <path.vcz>
```
should be equivalent to running 
```
bcftools view <path.vcf.gz>
```

See the [bio2zarr](https://sgkit-dev.github.io/bio2zarr/) project for help in 
converting VCF files to Zarr.

## Goals

Vcztools aims to be a drop-in replacement for a subset of bcftools functionality.
Currently supported are the ``view``, ``query`` and ``index -s/-n`` commands.

We aim for 100% compatibility so if you notice a difference between the output of 
vcztools and bcftools please do open an issue.

## Development

Vcztools is under active development and contributions are warmly welcomed. Please 
see the project on [GitHub](https://github.com/sgkit-dev/vcztools).

