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

All BGEN tests use the maintained `bgen-limix>=4.15,<5` backend through
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

The locked public bgen-limix backend provides both general and specialized
legacy layout-2 encoding interfaces. No development wheel is needed.
Capability-gate tests also cover backends missing the required interfaces;
encoding must fail before creating a payload or sidecars.

```bash
uv run pytest tests/test_bgen_native_encoding.py tests/test_bgen_native_interface.py tests/test_bgen_probability.py
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
