"""
Session-scoped VCZ fixtures for the vcztools test suite.

Every committed ``.vcz.zip`` under ``tests/data/vcf/`` is exposed as a
``VczFixture`` loaded once per session. Tests access the fixture's
``group`` attribute (a read-only ``zarr.Group``) directly — the
production APIs in ``vcztools`` short-circuit zarr groups at
``vcztools.utils.open_zarr``, so there's no extra plumbing.

Tests that need to mutate the data MUST take a copy via
``tests.vcz_builder.copy_vcz`` before writing; the session groups are
shared across the whole run.

Fixture naming convention
-------------------------
All pytest fixtures in this test suite — both the session-scoped ones
defined here and the class-scoped ones defined inside test modules —
start with the ``fx_`` prefix. This makes fixture parameters instantly
recognisable at call sites and avoids collisions with plain names
used in production code (``vcz``, ``root``, ``source``, ...).
"""

import pathlib
import zipfile
from dataclasses import dataclass

import pytest
import zarr
import zarr.storage

_VCF_DIR = pathlib.Path(__file__).parent / "data" / "vcf"


@dataclass(frozen=True)
class VczFixture:
    """A loaded VCZ test fixture, including pointers to source artefacts."""

    name: str
    vcf_path: pathlib.Path
    zip_path: pathlib.Path
    directory_path: pathlib.Path
    group: zarr.Group

    def path(self, backend: str) -> pathlib.Path:
        """Return the VCZ path for the given backend ("zip" or "directory")."""
        if backend == "zip":
            return self.zip_path
        if backend == "directory":
            return self.directory_path
        raise ValueError(f"unknown backend: {backend!r}")


def _load(name: str, vcf_filename: str, tmp_path_factory) -> VczFixture:
    vcf_path = _VCF_DIR / vcf_filename
    zip_path = _VCF_DIR / f"{name}.vcz.zip"
    store = zarr.storage.ZipStore(zip_path, mode="r")
    group = zarr.open(store, mode="r")
    directory_path = tmp_path_factory.mktemp(f"{name}_vcz_dir") / f"{name}.vcz"
    directory_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, mode="r") as zf:
        zf.extractall(directory_path)
    return VczFixture(
        name=name,
        vcf_path=vcf_path,
        zip_path=zip_path,
        directory_path=directory_path,
        group=group,
    )


# ---------------------------------------------------------------------------
# Per-VCF session fixtures. Registry keys are the VCF filename (matching
# what existing parametrizations pass around as strings).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fx_sample_vcz(tmp_path_factory) -> VczFixture:
    return _load("sample", "sample.vcf.gz", tmp_path_factory)


@pytest.fixture(scope="session")
def fx_sample_zv3_vcz(tmp_path_factory) -> VczFixture:
    return _load("sample_zv3", "sample.vcf.gz", tmp_path_factory)


@pytest.fixture(scope="session")
def fx_sample_split_alleles_vcz(tmp_path_factory) -> VczFixture:
    return _load(
        "sample-split-alleles", "sample-split-alleles.vcf.gz", tmp_path_factory
    )


@pytest.fixture(scope="session")
def fx_msprime_diploid_vcz(tmp_path_factory) -> VczFixture:
    return _load("msprime_diploid", "msprime_diploid.vcf.gz", tmp_path_factory)


@pytest.fixture(scope="session")
def fx_chr22_vcz(tmp_path_factory) -> VczFixture:
    return _load("chr22", "chr22.vcf.gz", tmp_path_factory)


@pytest.fixture(scope="session")
def fx_chrM_vcz(tmp_path_factory) -> VczFixture:
    return _load("1kg_2020_chrM", "1kg_2020_chrM.vcf.gz", tmp_path_factory)


@pytest.fixture(scope="session")
def fx_field_type_combos_vcz(tmp_path_factory) -> VczFixture:
    return _load("field_type_combos", "field_type_combos.vcf.gz", tmp_path_factory)


@pytest.fixture(scope="session")
def fx_chr20_annotations_vcz(tmp_path_factory) -> VczFixture:
    return _load(
        "1kg_2020_chr20_annotations",
        "1kg_2020_chr20_annotations.bcf",
        tmp_path_factory,
    )


@pytest.fixture(scope="session")
def fx_all_vcz(
    fx_sample_vcz,
    fx_sample_split_alleles_vcz,
    fx_msprime_diploid_vcz,
    fx_chr22_vcz,
    fx_chrM_vcz,
    fx_field_type_combos_vcz,
    fx_chr20_annotations_vcz,
) -> dict[str, VczFixture]:
    """
    Registry of every VCZ fixture keyed by its VCF filename. Use this in
    parametrized tests that select a fixture by string (e.g. bcftools
    parity tests that iterate over ``("view ...", "sample.vcf.gz")``).
    """
    fixtures = (
        fx_sample_vcz,
        fx_sample_split_alleles_vcz,
        fx_msprime_diploid_vcz,
        fx_chr22_vcz,
        fx_chrM_vcz,
        fx_field_type_combos_vcz,
        fx_chr20_annotations_vcz,
    )
    return {fx.vcf_path.name: fx for fx in fixtures}
