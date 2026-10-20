# Contributing to vcztools

## Development setup

We use [uv](https://docs.astral.sh/uv/) for dependency management.

Clone the repository and install all development dependencies:

```bash
git clone https://github.com/sgkit-dev/vcztools.git
cd vcztools
uv sync --group dev
```

Build the C extension module:

```bash
uv run python setup.py build_ext --inplace
```

## Running tests

```bash
uv run pytest
```

Note: some tests require `bcftools` and `plink` to be installed
(available via conda-forge and bioconda).

All BGEN tests use the maintained `bgen-limix>=4.14,<5` backend through
its public `BgenFile` and `BgenMetafile` APIs. It is included in the test
dependencies; no separate environment is needed.

```bash
uv run pytest tests/test_bgen.py tests/test_bgen_validation.py tests/test_bgen_probability.py
```

The native tests also include real VCZ -> BGEN -> bio2zarr roundtrips
for both Zarr formats, selected samples, GP/HP and missing/fill sentinels.
These run when a bio2zarr build providing `bio2zarr.bgen.convert` and
`sample_file` support is installed in that environment; otherwise only
those integration cases are skipped.

General encoding tests now require the matching bgen-limix 4.15 development
wheel exposing `encode_layout2` and `encode_layout2_hardcalls`. Released
4.14 still covers reading/indexing and the capability-gate tests, but is
not sufficient for the complete suite. This staged integration is not yet
public-install-ready, and CI using only the released lockfile cannot run
the full general-export suite successfully until the encoder is published.
Do not add machine-local wheel paths to `pyproject.toml` or `uv.lock`.

After installing the matching development wheel with `uv pip install`
into the project environment, use `uv run --no-sync pytest` to preserve it:

```bash
uv run --no-sync pytest tests/test_bgen_native_encoding.py tests/test_bgen_probability.py
```

## Linting

We use [prek](https://github.com/prek-dev/prek) for pre-commit linting,
configured in `prek.toml`. Install it as a pre-commit hook:

```bash
uv run prek install
```

Run all checks manually:

```bash
uv run --only-group=lint prek -c prek.toml run --all-files
```

If local results differ from CI, run `uv run prek cache clean`.

## C code

C source files are in `lib/`. C code is formatted with `clang-format`
(enforced by prek). Run the C unit tests:

```bash
cd lib
meson setup -Db_coverage=true build
ninja -C build test
```

## Pull requests

- Create a branch from `main`
- Ensure all CI checks pass
- Add a changelog entry if appropriate
