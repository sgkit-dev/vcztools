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

    ``vcz_path`` is the source store. ``bgen_minus1`` and ``bgen_stored``
    come from the ``view-bgen`` CLI at compression levels ``-1`` (zlib
    default) and ``0`` (stored) respectively. ``bgen_encoder`` is the
    fixed-size, byte-equal-width-per-variant output from the
    Python-API-only :class:`vcztools.bgen.BgenEncoder`. The encoder
    writes its own ``.sample`` sidecar (``bgen_encoder_sample``);
    ``sample_path`` is the one shared by both ``view-bgen`` outputs.
    ``plink_prefix`` is the bed/bim/fam prefix from ``view-plink``.
    """

    size: str
    vcz_path: pathlib.Path
    pheno_path: pathlib.Path
    plink_prefix: pathlib.Path
    bgen_minus1: pathlib.Path
    bgen_stored: pathlib.Path
    bgen_encoder: pathlib.Path
    sample_path: pathlib.Path
    bgen_encoder_sample: pathlib.Path


BGEN_LEVELS = ["lvl=-1", "lvl=0", "encoder"]


def bgen_for_level(fx: FixtureOutputs, level: str) -> tuple[pathlib.Path, pathlib.Path]:
    """Return ``(bgen_path, sample_path)`` for one of the three BGEN
    flavours produced by the fixture. The encoder needs its own .sample
    because it does not share the CLI's filter/sample plumbing."""
    if level == "encoder":
        return fx.bgen_encoder, fx.bgen_encoder_sample
    if level == "lvl=-1":
        return fx.bgen_minus1, fx.sample_path
    if level == "lvl=0":
        return fx.bgen_stored, fx.sample_path
    raise ValueError(f"unknown BGEN level identifier {level!r}")


def _require_fixture(size: str, variant: str) -> pathlib.Path:
    """Resolve ``data/<size>_<variant>.vcz``; skip if the size's marker
    file (which signals *both* variants are built) is missing."""
    vcz = DATA_DIR / f"{size}_{variant}.vcz"
    if not (DATA_DIR / f"{size}.vcz.ready").exists():
        pytest.skip(
            f"{size}.vcz fixture missing — run `make -C validation data` to generate it"
        )
    return vcz


def _build_outputs(size: str, variant: str, work: pathlib.Path) -> FixtureOutputs:
    vcz = _require_fixture(size, variant)
    pheno = DATA_DIR / f"{size}.pheno.tsv"

    plink_prefix = work / "plink"
    bgen_minus1 = work / "bgen_lvl_minus1"
    bgen_stored = work / "bgen_lvl_stored"
    bgen_encoder = work / "bgen_encoder"

    # The msprime simulation produces occasional recurrent mutations at
    # the same site, so a small fraction of variants are tri-allelic.
    # PLINK 1 .bed and BGEN layout 2 (as we write it) are biallelic-
    # only; --max-alleles 2 drops those rows, matching what every
    # downstream tool expects. run_bgen_encoder applies the same
    # biallelic-only filter internally.
    extra = "--max-alleles 2"
    helpers.run_view_plink(vcz, plink_prefix, extra_args=extra)
    helpers.run_view_bgen(vcz, bgen_minus1, compression_level=-1, extra_args=extra)
    helpers.run_view_bgen(vcz, bgen_stored, compression_level=0, extra_args=extra)
    helpers.run_bgen_encoder(vcz, bgen_encoder)

    # view-bgen writes a `.sample` next to each BGEN; the two view-bgen
    # outputs share a sample list, the encoder writes its own.
    sample_path = bgen_minus1.with_suffix(".sample")
    encoder_sample_path = bgen_encoder.with_suffix(".sample")
    for p in (sample_path, encoder_sample_path):
        if not p.exists():
            raise AssertionError(f"missing .sample file at {p}")

    return FixtureOutputs(
        size=size,
        vcz_path=vcz,
        pheno_path=pheno,
        plink_prefix=plink_prefix,
        bgen_minus1=bgen_minus1.with_suffix(".bgen"),
        bgen_stored=bgen_stored.with_suffix(".bgen"),
        bgen_encoder=bgen_encoder.with_suffix(".bgen"),
        sample_path=sample_path,
        bgen_encoder_sample=encoder_sample_path,
    )


@pytest.fixture(scope="session")
def small_unphased_fixture(tmp_path_factory) -> FixtureOutputs:
    work = tmp_path_factory.mktemp("small_unphased_outputs")
    return _build_outputs("small", "unphased", work)


@pytest.fixture(scope="session")
def small_phased_fixture(tmp_path_factory) -> FixtureOutputs:
    work = tmp_path_factory.mktemp("small_phased_outputs")
    return _build_outputs("small", "phased", work)


@pytest.fixture(scope="session")
def large_unphased_fixture(tmp_path_factory) -> FixtureOutputs:
    work = tmp_path_factory.mktemp("large_unphased_outputs")
    return _build_outputs("large", "unphased", work)


@pytest.fixture(scope="session")
def large_phased_fixture(tmp_path_factory) -> FixtureOutputs:
    work = tmp_path_factory.mktemp("large_phased_outputs")
    return _build_outputs("large", "phased", work)
