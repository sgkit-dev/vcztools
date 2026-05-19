# vcztools downstream-tool validation suite.
#
# Targets:
#   make tools       - install every external tool into tools/<name>/
#   make data        - generate synthetic VCZ fixtures into $(DATA_DIR)
#   make test        - run the pytest suite (skips tools that aren't installed)
#   make all         - tools + data + test
#   make clean-tools - remove all installed tools
#   make clean-data  - remove generated test data
#   make clean       - clean-tools + clean-data
#
# Each tool install runs independently; `make -j` builds them in parallel.

SHELL := /usr/bin/env bash
HERE  := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
# Fixtures live outside the repo. Default location is
# ../datasets/validation/ (relative to the vcztools repo root) so
# recursive searches over the working tree don't trip on them.
# Matches DATA_DIR in generate_data.py / conftest.py.
DATA_DIR := $(abspath $(HERE)../../datasets/validation)

TOOLS := qctool regenie plink19 bgenix bolt_lmm
TOOL_MARKERS := $(addprefix $(HERE)tools/,$(addsuffix /.installed,$(TOOLS)))

.PHONY: all tools data test clean clean-tools clean-data $(TOOLS)

# `make -j all` parallelises the tool installs (their only mutual
# dependency-free phase) while keeping `test` ordered after both
# `tools` and `data` so it doesn't run against half-installed binaries.
all: test

# `tools` depends on the per-tool .installed marker files, not on the
# phony per-tool target names, so a no-op rebuild walks the marker
# mtimes and skips the install scripts entirely (the scripts
# self-short-circuit when the marker is already present, but the
# shell invocation alone added noise on every `make all`).
tools: $(TOOL_MARKERS)

# Pattern rule: tools/<name>/.installed is rebuilt when its install
# script changes (or the marker is missing). The install script
# creates the marker as its last step.
$(HERE)tools/%/.installed: $(HERE)install/install_%.sh
	bash $<

# Convenience phony aliases (`make qctool`, `make bgenix`, etc.):
# trigger the marker rule above but do nothing themselves.
$(TOOLS): %: $(HERE)tools/%/.installed

data: $(DATA_DIR)/small.vcz.ready $(DATA_DIR)/large.vcz.ready $(DATA_DIR)/haploid.vcz.ready $(DATA_DIR)/mixed_ploidy.vcz.ready $(DATA_DIR)/varied_strings.vcz.ready

# Marker file sits next to the store (not inside it) so zarr doesn't
# warn about unrecognised hierarchy components on every open.
$(DATA_DIR)/small.vcz.ready: generate_data.py
	uv run --group validation python generate_data.py --size small

$(DATA_DIR)/large.vcz.ready: generate_data.py
	uv run --group validation python generate_data.py --size large

$(DATA_DIR)/haploid.vcz.ready: generate_data.py
	uv run --group validation python generate_data.py --kind haploid

$(DATA_DIR)/mixed_ploidy.vcz.ready: generate_data.py
	uv run --group validation python generate_data.py --kind mixed_ploidy

$(DATA_DIR)/varied_strings.vcz.ready: generate_data.py
	uv run --group validation python generate_data.py --kind varied_strings

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
	rm -rf $(DATA_DIR)

clean: clean-tools clean-data
