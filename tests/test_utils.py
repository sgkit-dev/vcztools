import pathlib

import numpy as np
import pytest
import zarr
from numpy.testing import assert_array_equal

from vcztools import utils
from vcztools.constants import (
    FLOAT32_FILL,
    FLOAT32_MISSING,
    INT_FILL,
    INT_MISSING,
    STR_FILL,
    STR_MISSING,
)
from vcztools.utils import (
    _as_fixed_length_string,
    _as_fixed_length_unicode,
    missing,
    open_zarr,
    search,
    vcf_name_to_vcz_names,
)

FIXTURE_VCZ_ZIP = pathlib.Path("tests/data/vcf/sample.vcz.zip")


@pytest.mark.parametrize(
    ("a", "v", "expected_ind"),
    [
        (["a", "b", "c", "d"], ["b", "a", "c"], [1, 0, 2]),
        (["a", "c", "d", "b"], ["b", "a", "c"], [3, 0, 1]),
        (["a", "c", "d", "b"], ["b", "a", "a", "c"], [3, 0, 0, 1]),
        (["a", "c", "d", "b"], [], []),
    ],
)
def test_search(a, v, expected_ind):
    assert_array_equal(search(a, v), expected_ind)


class TestChunkRead:
    """Axis-agnostic chunk-read descriptor used by both the variants
    plan (:mod:`vcztools.regions`) and the samples plan
    (:mod:`vcztools.samples`)."""

    def test_defaults(self):
        cr = utils.ChunkRead(index=3)
        assert cr.index == 3
        assert cr.selection is None

    def test_with_selection(self):
        sel = np.array([0, 2], dtype=np.int64)
        cr = utils.ChunkRead(index=1, selection=sel)
        assert cr.index == 1
        assert_array_equal(cr.selection, [0, 2])


@pytest.mark.parametrize(
    ("vczs", "vcf", "expected_vcz_names"),
    [
        ({"call_genotype"}, "GT", ["call_genotype"]),
        ({"call_genotype"}, "FMT/GT", ["call_genotype"]),
        ({"call_genotype"}, "FORMAT/GT", ["call_genotype"]),
        ({"call_DP"}, "DP", ["call_DP"]),
        ({"variant_DP"}, "DP", ["variant_DP"]),
        ({"call_DP", "variant_DP"}, "DP", ["call_DP", "variant_DP"]),
        ({"call_DP", "variant_DP"}, "FORMAT/DP", ["call_DP"]),
        ({"call_DP", "variant_DP"}, "INFO/DP", ["variant_DP"]),
        ({"variant_DP"}, "FORMAT/DP", []),
        ({"call_DP"}, "INFO/DP", []),
        (set(), "CHROM", ["variant_contig"]),
        (set(), "POS", ["variant_position"]),
        (set(), "ID", ["variant_id"]),
        (set(), "REF", ["variant_allele"]),
        (set(), "ALT", ["variant_allele"]),
        (set(), "QUAL", ["variant_quality"]),
        (set(), "FILTER", ["variant_filter"]),
    ],
)
def test_vcf_name_to_vcz_names(vczs, vcf, expected_vcz_names):
    assert vcf_name_to_vcz_names(vczs, vcf) == expected_vcz_names


@pytest.mark.parametrize("dtype", ["O", "T"])
def test_as_fixed_length_string(dtype):
    assert_array_equal(
        _as_fixed_length_string(np.array(["A", "BB"], dtype=dtype)),
        np.array(["A", "BB"], dtype="S2"),
    )


@pytest.mark.parametrize("dtype", ["O", "T"])
def test_as_fixed_length_unicode(dtype):
    assert_array_equal(
        _as_fixed_length_unicode(np.array(["A", "BB"], dtype=dtype)),
        np.array(["A", "BB"], dtype="U2"),
    )


@pytest.mark.parametrize(
    ("arr", "expected_missing"),
    [
        (
            np.array([0, 1, INT_MISSING, INT_MISSING, INT_FILL, 2], np.int32),
            np.array([False, False, True, True, False, False]),
        ),
        (
            np.array(
                [0.0, 1.0, FLOAT32_MISSING, FLOAT32_MISSING, FLOAT32_FILL, np.nan],
                np.float32,
            ),
            np.array([False, False, True, True, False, False]),
        ),
        (
            np.array(["a", "b", STR_MISSING, STR_MISSING, STR_FILL, " "]),
            np.array([False, False, True, True, False, False]),
        ),
        (
            np.array([True, True, False, True]),
            np.array([False, False, True, False]),
        ),
    ],
)
def test_missing(arr, expected_missing):
    assert_array_equal(missing(arr), expected_missing)


def test_missing__failure():
    with pytest.raises(ValueError, match="unrecognised dtype"):
        missing(np.array([1, 2], dtype=np.complex64))


class TestArrayDims:
    def test_zarr_v3_dimension_names(self):
        store = zarr.storage.MemoryStore()
        root = zarr.group(store=store, zarr_format=3)
        arr = root.create_array(
            "x", shape=(10, 3), dtype="f4", dimension_names=("variants", "ploidy")
        )
        assert utils.array_dims(arr) == ("variants", "ploidy")

    def test_zarr_v3_1d(self):
        store = zarr.storage.MemoryStore()
        root = zarr.group(store=store, zarr_format=3)
        arr = root.create_array(
            "x", shape=(5,), dtype="<U8", dimension_names=("filters",)
        )
        assert utils.array_dims(arr) == ("filters",)

    def test_zarr_v2_array_dimensions_attr(self):
        store = zarr.storage.MemoryStore()
        root = zarr.group(store=store, zarr_format=2)
        arr = root.create_array("x", shape=(10, 3), dtype="f4")
        arr.attrs["_ARRAY_DIMENSIONS"] = ["variants", "ploidy"]
        assert utils.array_dims(arr) == ["variants", "ploidy"]

    def test_zarr_v2_1d(self):
        store = zarr.storage.MemoryStore()
        root = zarr.group(store=store, zarr_format=2)
        arr = root.create_array("x", shape=(5,), dtype="<U8")
        arr.attrs["_ARRAY_DIMENSIONS"] = ["filters"]
        assert utils.array_dims(arr) == ["filters"]

    def test_v2_attr_takes_precedence(self):
        """When _ARRAY_DIMENSIONS attr is set, it is returned even on v3."""
        store = zarr.storage.MemoryStore()
        root = zarr.group(store=store, zarr_format=3)
        arr = root.create_array(
            "x", shape=(10,), dtype="i4", dimension_names=("variants",)
        )
        arr.attrs["_ARRAY_DIMENSIONS"] = ["overridden"]
        assert utils.array_dims(arr) == ["overridden"]


class TestOpenZarr:
    def _write_minimal_group(self, path):
        root = zarr.open(path, mode="w")
        root.create_array("variant_position", shape=(4,), dtype="int32")
        root["variant_position"][:] = [10, 20, 30, 40]

    def test_zip_path(self):
        root = open_zarr(FIXTURE_VCZ_ZIP)
        assert isinstance(root.store, zarr.storage.ZipStore)
        assert root["sample_id"][:].tolist() == ["NA00001", "NA00002", "NA00003"]

    def test_zip_str(self):
        root = open_zarr(str(FIXTURE_VCZ_ZIP))
        assert isinstance(root.store, zarr.storage.ZipStore)

    def test_zip_with_fsspec_backend(self):
        root = open_zarr(FIXTURE_VCZ_ZIP, zarr_backend_storage="fsspec")
        assert isinstance(root.store, zarr.storage.ZipStore)

    def test_directory_path(self, tmp_path):
        vcz = tmp_path / "minimal.vcz"
        self._write_minimal_group(vcz)
        root = open_zarr(vcz)
        assert not isinstance(root.store, zarr.storage.ZipStore)
        assert root["variant_position"][:].tolist() == [10, 20, 30, 40]

    def test_obstore(self, tmp_path):
        vcz = tmp_path / "minimal.vcz"
        self._write_minimal_group(vcz)
        root = open_zarr(vcz, zarr_backend_storage="obstore")
        assert isinstance(root.store, zarr.storage.ObjectStore)
        assert root["variant_position"][:].tolist() == [10, 20, 30, 40]
        # open with store object
        root = open_zarr(root.store, zarr_backend_storage="obstore")
        assert isinstance(root.store, zarr.storage.ObjectStore)
        assert root["variant_position"][:].tolist() == [10, 20, 30, 40]

    def test_nonexistent_zip_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            open_zarr(tmp_path / "does-not-exist.vcz.zip")
