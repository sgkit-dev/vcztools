import pathlib
import sys

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
    array_memory_bytes,
    missing,
    normalise_local_selection,
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


class TestNormaliseLocalSelection:
    """Collapse a contiguous, sorted, no-duplicate per-chunk selection
    into ``None`` (full chunk), a ``slice`` (contiguous range), or pass
    the original ndarray through (anything else)."""

    def test_empty_returns_input(self):
        local_sel = np.array([], dtype=np.int64)
        result = normalise_local_selection(local_sel, chunk_size=10)
        assert result is local_sel

    def test_full_chunk_returns_none(self):
        local_sel = np.arange(0, 4, dtype=np.int64)
        assert normalise_local_selection(local_sel, chunk_size=4) is None

    def test_full_chunk_size_two(self):
        local_sel = np.array([0, 1], dtype=np.int64)
        assert normalise_local_selection(local_sel, chunk_size=2) is None

    def test_contiguous_from_zero_partial(self):
        local_sel = np.array([0, 1, 2], dtype=np.int64)
        assert normalise_local_selection(local_sel, chunk_size=10) == slice(0, 3)

    def test_contiguous_offset(self):
        local_sel = np.array([3, 4, 5], dtype=np.int64)
        assert normalise_local_selection(local_sel, chunk_size=10) == slice(3, 6)

    def test_single_element_is_slice(self):
        local_sel = np.array([5], dtype=np.int64)
        assert normalise_local_selection(local_sel, chunk_size=10) == slice(5, 6)

    def test_non_contiguous_returns_input(self):
        local_sel = np.array([1, 3, 5], dtype=np.int64)
        result = normalise_local_selection(local_sel, chunk_size=10)
        assert result is local_sel

    def test_out_of_order_returns_input(self):
        local_sel = np.array([3, 2, 1], dtype=np.int64)
        result = normalise_local_selection(local_sel, chunk_size=10)
        assert result is local_sel

    def test_last_element_early_out(self):
        # First/last endpoints are consistent with a contiguous range but
        # the last element doesn't match stop-1 — exercises the cheap reject.
        local_sel = np.array([3, 4, 6], dtype=np.int64)
        result = normalise_local_selection(local_sel, chunk_size=10)
        assert result is local_sel

    def test_array_equal_reject(self):
        # Last element equals stop-1 so the cheap reject passes, but the
        # interior breaks the arange — exercises the array_equal reject.
        local_sel = np.array([1, 1, 3], dtype=np.int64)
        result = normalise_local_selection(local_sel, chunk_size=10)
        assert result is local_sel


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


class TestArrayMemoryBytes:
    """Direct unit tests for ``utils.array_memory_bytes``.

    The readahead pipeline calls this on the first chunk's prefetched
    blocks to size its window. For variable-length string arrays the
    returned value must include heap-allocated string content, not
    just ``arr.nbytes`` (which is metadata-only for ``object`` and
    ``StringDType``). Drive the helper through both branches with
    raw numpy arrays so a regression to the lower bound is caught
    without going through Zarr.
    """

    def test_fixed_size_numeric_is_exact(self):
        arr = np.zeros(100, dtype=np.int16)
        assert array_memory_bytes(arr) == 200

    def test_fixed_width_unicode_falls_through_to_nbytes(self):
        arr = np.array(["abc"] * 4, dtype="<U8")
        assert array_memory_bytes(arr) == arr.nbytes

    def test_string_dtype_includes_utf8_content(self):
        long = "a" * 250
        arr = np.array([long] * 4, dtype=np.dtypes.StringDType())
        result = array_memory_bytes(arr)
        # Must include the 4 * 250 = 1000 bytes of content beyond
        # the per-element metadata cells.
        assert result >= int(arr.nbytes) + 1000

    def test_string_dtype_uses_utf8_byte_length_not_codepoints(self):
        # "αβγ" is 3 codepoints but 6 UTF-8 bytes.
        arr = np.array(["αβγ"] * 4, dtype=np.dtypes.StringDType())
        result = array_memory_bytes(arr)
        assert result == int(arr.nbytes) + 4 * 6

    def test_object_dtype_includes_python_string_overhead(self):
        arr = np.array(["", "", "", "", ""], dtype=object)
        result = array_memory_bytes(arr)
        # Empty Python str is ~41 bytes on CPython 3.12; require the
        # measurement to reflect the per-element header, not the
        # 8-byte pointer-only lower bound that arr.nbytes reports.
        assert result > 4 * arr.nbytes

    def test_object_dtype_scales_with_content(self):
        arr_short = np.array(["a"] * 4, dtype=object)
        arr_long = np.array(["a" * 1000] * 4, dtype=object)
        # Each long element exceeds the short element by ~1000 bytes
        # (Python str storage is 1 byte per ASCII char).
        delta = array_memory_bytes(arr_long) - array_memory_bytes(arr_short)
        assert delta >= 4 * 999

    def test_object_dtype_matches_sys_getsizeof_sum(self):
        elements = ["", "a", "bb", "ccc", "dddd" * 100]
        arr = np.array(elements, dtype=object)
        expected = sum(sys.getsizeof(s) for s in elements)
        assert array_memory_bytes(arr) == expected


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
