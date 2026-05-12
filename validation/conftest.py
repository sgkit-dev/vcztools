"""Pytest configuration for the downstream-tool validation suite.

Provides:

- Per-tool ``<name>_bin`` fixtures that resolve the installed binary
  under ``validation/tools/<name>/bin/``. If the marker file is
  missing, the fixture skips the test, mirroring the ``shutil.which``
  pattern from ``tests/test_plink_validation.py``.
- Session-scoped fixtures for the two VCZ sizes plus their
  pre-converted PLINK and BGEN outputs (compression level -1 and 0).
"""

from __future__ import annotations

import dataclasses
import pathlib

import pytest

from . import helpers

HERE = pathlib.Path(__file__).parent
TOOLS_DIR = HERE / "tools"
DATA_DIR = HERE / "data"


# ---------------------------------------------------------------------------
# Tool discovery — one fixture per tool. Skipped if not installed.
# ---------------------------------------------------------------------------


def _resolve_tool(name: str, binary: str) -> pathlib.Path | None:
    """Return the path to <name>'s binary if its .installed marker is
    present and the binary is executable; otherwise None."""
    marker = TOOLS_DIR / name / ".installed"
    candidate = TOOLS_DIR / name / "bin" / binary
    if marker.exists() and candidate.exists():
        return candidate
    return None


def _skip_if_missing(path: pathlib.Path | None, name: str) -> pathlib.Path:
    if path is None:
        pytest.skip(
            f"{name} not installed — run `make -C validation {name}` to install it"
        )
    return path


@pytest.fixture(scope="session")
def qctool_bin() -> pathlib.Path:
    return _skip_if_missing(_resolve_tool("qctool", "qctool"), "qctool")


@pytest.fixture(scope="session")
def regenie_bin() -> pathlib.Path:
    return _skip_if_missing(_resolve_tool("regenie", "regenie"), "regenie")


@pytest.fixture(scope="session")
def plink19_bin() -> pathlib.Path:
    return _skip_if_missing(_resolve_tool("plink19", "plink"), "plink19")


@pytest.fixture(scope="session")
def bgenix_bin() -> pathlib.Path:
    return _skip_if_missing(_resolve_tool("bgenix", "bgenix"), "bgenix")


@pytest.fixture(scope="session")
def bolt_lmm_bin() -> pathlib.Path:
    return _skip_if_missing(_resolve_tool("bolt_lmm", "bolt"), "bolt_lmm")


# ---------------------------------------------------------------------------
# VCZ fixtures + converted outputs.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FixtureOutputs:
    """Paths to converted artefacts for one VCZ fixture.

    ``vcz_path`` is the source store. ``bgen_minus1`` is BGEN written
    at the zlib default level (-1); ``bgen_stored`` is BGEN written
    with compression level 0 (the "fixed length / stored" variant we
    want exercised). ``plink_prefix`` is the bed/bim/fam prefix from
    ``view-plink``. ``sample_path`` is the .sample file produced
    alongside the BGEN outputs (qctool / bgenix need it).
    """

    size: str
    vcz_path: pathlib.Path
    pheno_path: pathlib.Path
    plink_prefix: pathlib.Path
    bgen_minus1: pathlib.Path
    bgen_stored: pathlib.Path
    sample_path: pathlib.Path


def _require_fixture(size: str) -> pathlib.Path:
    vcz = DATA_DIR / f"{size}.vcz"
    if not (DATA_DIR / f"{size}.vcz.ready").exists():
        pytest.skip(
            f"{size}.vcz fixture missing — run `make -C validation data` to generate it"
        )
    return vcz


def _build_outputs(size: str, work: pathlib.Path) -> FixtureOutputs:
    vcz = _require_fixture(size)
    pheno = DATA_DIR / f"{size}.pheno.tsv"

    plink_prefix = work / "plink"
    bgen_minus1 = work / "bgen_lvl_minus1"
    bgen_stored = work / "bgen_lvl_stored"

    # The msprime simulation produces occasional recurrent mutations at
    # the same site, so a small fraction of variants are tri-allelic.
    # PLINK 1 .bed and BGEN layout 2 (as we write it) are biallelic-
    # only; --max-alleles 2 drops those rows, matching what every
    # downstream tool expects.
    extra = "--max-alleles 2"
    helpers.run_view_plink(vcz, plink_prefix, extra_args=extra)
    helpers.run_view_bgen(vcz, bgen_minus1, compression_level=-1, extra_args=extra)
    helpers.run_view_bgen(vcz, bgen_stored, compression_level=0, extra_args=extra)

    # vcztools writes a `.sample` next to the BGEN; both BGENs share
    # the same sample list. Use the one from bgen_minus1.
    sample_path = bgen_minus1.with_suffix(".sample")
    if not sample_path.exists():
        raise AssertionError(f"missing .sample file at {sample_path}")

    return FixtureOutputs(
        size=size,
        vcz_path=vcz,
        pheno_path=pheno,
        plink_prefix=plink_prefix,
        bgen_minus1=bgen_minus1.with_suffix(".bgen"),
        bgen_stored=bgen_stored.with_suffix(".bgen"),
        sample_path=sample_path,
    )


@pytest.fixture(scope="session")
def small_fixture(tmp_path_factory) -> FixtureOutputs:
    work = tmp_path_factory.mktemp("small_outputs")
    return _build_outputs("small", work)


@pytest.fixture(scope="session")
def large_fixture(tmp_path_factory) -> FixtureOutputs:
    work = tmp_path_factory.mktemp("large_outputs")
    return _build_outputs("large", work)
