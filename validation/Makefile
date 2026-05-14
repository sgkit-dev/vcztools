# vcztools downstream-tool validation suite.
#
# Targets:
#   make tools       - install every external tool into tools/<name>/
#   make data        - generate synthetic VCZ fixtures into data/
#   make test        - run the pytest suite (skips tools that aren't installed)
#   make all         - tools + data + test
#   make clean-tools - remove all installed tools
#   make clean-data  - remove generated test data
#   make clean       - clean-tools + clean-data
#
# Each tool install runs independently; `make -j` builds them in parallel.

SHELL := /usr/bin/env bash
HERE  := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

TOOLS := qctool regenie plink19 bgenix bolt_lmm

.PHONY: all tools data test clean clean-tools clean-data $(TOOLS)

# `make -j all` parallelises the tool installs (their only mutual
# dependency-free phase) while keeping `test` ordered after both
# `tools` and `data` so it doesn't run against half-installed binaries.
all: test

tools: $(TOOLS)

$(TOOLS):
	bash $(HERE)install/install_$@.sh

data: data/small.vcz.ready data/large.vcz.ready data/haploid.vcz.ready data/mixed_ploidy.vcz.ready

# Marker file sits next to the store (not inside it) so zarr doesn't
# warn about unrecognised hierarchy components on every open.
data/small.vcz.ready: generate_data.py
	uv run --group validation python generate_data.py --size small

data/large.vcz.ready: generate_data.py
	uv run --group validation python generate_data.py --size large

data/haploid.vcz.ready: generate_data.py
	uv run --group validation python generate_data.py --kind haploid

data/mixed_ploidy.vcz.ready: generate_data.py
	uv run --group validation python generate_data.py --kind mixed_ploidy

# The repo's pyproject.toml sets `addopts = "-n auto"` for xdist
# parallelism. The validation suite runs heavy external binaries
# (BOLT-LMM in particular) on multi-thousand-sample fixtures and OOMs
# under parallel workers, so override addopts to run serially.
# `--group validation` resolves msprime / tstrait / bio2zarr /
# bgen-reader ad-hoc; the `dev` group does not pull them in.
test: tools data
	uv run --group validation pytest -v -o "addopts=" $(HERE)

clean-tools:
	rm -rf $(HERE)tools

clean-data:
	rm -rf $(HERE)data

clean: clean-tools clean-data
