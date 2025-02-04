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

## Cloud stores

Vcztools can read vcz files from cloud stores using [fsspec](https://filesystem-spec.readthedocs.io/en/latest/).

For example, to read from Amazon S3, first install the `s3fs` fsspec library:

```
python3 -m pip install s3fs
```

Then provide your AWS credentials as described in the [`s3fs` documentation](https://s3fs.readthedocs.io/en/latest/#credentials), for example by setting environment variables:

```
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

You can then run vcztools using an `s3://` URL:

```
python -m vcztools view s3://<bucket-name>/path/to.vcz
```

## Development

Vcztools is under active development and contributions are warmly welcomed. Please 
see the project on [GitHub](https://github.com/sgkit-dev/vcztools).

