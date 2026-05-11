import pathlib
import sys

import icechunk as ic
import numpy as np
import obstore as obs
import pytest
import zarr
from numpy.testing import assert_array_equal

from tests import vcz_builder
from tests.utils import to_vcz_icechunk
from vcztools import regions as regions_mod
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
        cr = utils.ChunkRead(index=3, num_selected=5)
        assert cr.index == 3
        assert cr.num_selected == 5
        assert cr.selection is None

    def test_with_selection(self):
        sel = np.array([0, 2], dtype=np.int64)
        cr = utils.ChunkRead(index=1, num_selected=sel.size, selection=sel)
        assert cr.index == 1
        assert cr.num_selected == 2
        assert_array_equal(cr.selection, [0, 2])

    def test_simple_plan_exact_multiple(self):
        plan = utils.ChunkRead.simple_plan(length=12, chunk_size=4)
        assert [(cr.index, cr.num_selected, cr.selection) for cr in plan] == [
            (0, 4, None),
            (1, 4, None),
            (2, 4, None),
        ]

    def test_simple_plan_partial_last_chunk(self):
        plan = utils.ChunkRead.simple_plan(length=10, chunk_size=4)
        assert [(cr.index, cr.num_selected, cr.selection) for cr in plan] == [
            (0, 4, None),
            (1, 4, None),
            (2, 2, None),
        ]

    def test_simple_plan_single_partial_chunk(self):
        plan = utils.ChunkRead.simple_plan(length=3, chunk_size=10)
        assert [(cr.index, cr.num_selected, cr.selection) for cr in plan] == [
            (0, 3, None),
        ]

    def test_simple_plan_zero_length(self):
        assert utils.ChunkRead.simple_plan(length=0, chunk_size=5) == []

    def test_num_selected_matches_selection_length_via_chunk_plan_from_indexes(self):
        # chunk_plan_from_indexes is the canonical builder for
        # selection-bearing entries; assert num_selected matches the
        # actual local selection size for each form
        # (None / slice / ndarray) it produces.
        plan = regions_mod.chunk_plan_from_indexes(
            np.array([0, 1, 2, 5, 7, 8], dtype=np.int64),
            min_chunk=3,
        )
        # chunk 0: contiguous 0..2 → selection=None (full chunk)
        # chunk 1: index 5 → selection=slice or ndarray of size 1
        # chunk 2: indexes 7, 8 → selection=slice(1, 3) or ndarray of size 2
        for cr in plan:
            if cr.selection is None:
                assert cr.num_selected == 3
            elif isinstance(cr.selection, slice):
                assert cr.num_selected == cr.selection.stop - cr.selection.start
            else:
                assert cr.num_selected == len(cr.selection)


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


class TestComputeStreamChunkSize:
    """``compute_stream_chunk_size`` returns the minimum ``chunks[0]``
    among the read fields, with ``min_chunk`` as the fallback when no
    read fields are involved."""

    @staticmethod
    def _vcz_with_overrides(overrides):
        return vcz_builder.make_vcz(
            variant_contig=[0] * 12,
            variant_position=list(range(100, 112)),
            alleles=[("A", "T")] * 12,
            num_samples=2,
            variants_chunk_size=3,
            field_chunk_overrides=overrides,
            call_genotype=np.zeros((12, 2, 2), dtype=np.int8),
        )

    def test_empty_read_fields_falls_back_to_min_chunk(self):
        root = self._vcz_with_overrides(None)
        assert utils.compute_stream_chunk_size(root, [], min_chunk=3) == 3

    def test_single_call_field_returns_min_chunk(self):
        root = self._vcz_with_overrides(None)
        assert (
            utils.compute_stream_chunk_size(root, ["call_genotype"], min_chunk=3) == 3
        )

    def test_single_variant_only_field_returns_field_chunk(self):
        root = self._vcz_with_overrides({"variant_position": 12})
        assert (
            utils.compute_stream_chunk_size(root, ["variant_position"], min_chunk=3)
            == 12
        )

    def test_mixed_fields_returns_minimum(self):
        # variant_position chunked at 12, call_genotype at min_chunk=3 →
        # stream chunk size pinned at min_chunk because the call_* field
        # is referenced.
        root = self._vcz_with_overrides({"variant_position": 12})
        assert (
            utils.compute_stream_chunk_size(
                root, ["variant_position", "call_genotype"], min_chunk=3
            )
            == 3
        )

    def test_two_variant_only_with_distinct_chunks_returns_min(self):
        # variant_position chunked at 12, variant_contig at 6 → stream
        # chunk size is the minimum of the read fields' chunks[0].
        root = self._vcz_with_overrides({"variant_position": 12, "variant_contig": 6})
        assert (
            utils.compute_stream_chunk_size(
                root, ["variant_position", "variant_contig"], min_chunk=3
            )
            == 6
        )


class TestRebucketToStreamPlan:
    """``rebucket_to_stream_plan`` merges canonical (min_chunk-unit)
    chunk reads into stream-chunk-unit entries; selections become
    stream-chunk-local."""

    def test_identity_when_sizes_match(self):
        canonical = [utils.ChunkRead(index=i, num_selected=4) for i in range(3)]
        result = utils.rebucket_to_stream_plan(
            canonical, min_chunk=4, stream_chunk_size=4
        )
        assert result is canonical

    def test_full_simple_plan_collapses_to_selection_none(self):
        # 12 variants at min_chunk=3 → 4 canonical entries, all full.
        # stream_chunk_size=12 → one stream chunk covering the whole axis.
        canonical = utils.ChunkRead.simple_plan(length=12, chunk_size=3)
        result = utils.rebucket_to_stream_plan(
            canonical, min_chunk=3, stream_chunk_size=12
        )
        assert len(result) == 1
        assert result[0].index == 0
        assert result[0].num_selected == 12
        assert result[0].selection is None

    def test_multiple_stream_chunks_full_coverage(self):
        # 12 variants at min_chunk=3, stream_chunk_size=6 → 2 stream chunks,
        # each covering two canonical chunks fully.
        canonical = utils.ChunkRead.simple_plan(length=12, chunk_size=3)
        result = utils.rebucket_to_stream_plan(
            canonical, min_chunk=3, stream_chunk_size=6
        )
        assert [(r.index, r.num_selected, r.selection) for r in result] == [
            (0, 6, None),
            (1, 6, None),
        ]

    def test_partial_last_stream_chunk(self):
        # 10 variants at min_chunk=3 → canonical [3, 3, 3, 1].
        # stream_chunk_size=6 → stream chunk 0 covers entries (0, 1) fully;
        # stream chunk 1 covers (2, 3): 3 + 1 = 4 selected.
        canonical = utils.ChunkRead.simple_plan(length=10, chunk_size=3)
        result = utils.rebucket_to_stream_plan(
            canonical, min_chunk=3, stream_chunk_size=6
        )
        assert len(result) == 2
        assert (result[0].index, result[0].num_selected, result[0].selection) == (
            0,
            6,
            None,
        )
        # 4 of 6 stream-chunk rows selected; fast path emits slice(0, 4).
        assert result[1].index == 1
        assert result[1].num_selected == 4
        assert result[1].selection == slice(0, 4)

    def test_sparse_canonical_with_slice_selections(self):
        # Region-style: canonical entries with explicit slice selections
        # for two distinct min-chunks in one stream chunk.
        canonical = [
            utils.ChunkRead(index=0, num_selected=2, selection=slice(1, 3)),
            utils.ChunkRead(index=1, num_selected=2, selection=slice(0, 2)),
        ]
        result = utils.rebucket_to_stream_plan(
            canonical, min_chunk=3, stream_chunk_size=6
        )
        assert len(result) == 1
        entry = result[0]
        assert entry.index == 0
        assert entry.num_selected == 4
        # Indices [1, 2, 3, 4] in stream-chunk-local coords → slice(1, 5).
        assert entry.selection == slice(1, 5)

    def test_sparse_canonical_with_ndarray_selection(self):
        # Two non-contiguous local picks; rebase plus normalise leaves
        # an ndarray (the contiguous-range collapser rejects).
        canonical = [
            utils.ChunkRead(
                index=0,
                num_selected=2,
                selection=np.array([0, 2], dtype=np.int64),
            ),
            utils.ChunkRead(
                index=1,
                num_selected=1,
                selection=np.array([1], dtype=np.int64),
            ),
        ]
        result = utils.rebucket_to_stream_plan(
            canonical, min_chunk=3, stream_chunk_size=6
        )
        assert len(result) == 1
        entry = result[0]
        assert entry.num_selected == 3
        # Rebased: chunk 0 → [0, 2], chunk 1 → [3+1] = [4]. Stream-local
        # indices = [0, 2, 4] → non-contiguous → ndarray.
        assert_array_equal(entry.selection, [0, 2, 4])

    def test_sparse_canonical_skips_intermediate_min_chunks(self):
        # User selection omits min-chunk 1; stream chunk 0 (covering
        # min-chunks 0, 1, 2 when stream_chunk_size=9, min_chunk=3) keeps
        # the rebased indices from 0 and 2.
        canonical = [
            utils.ChunkRead(index=0, num_selected=3),
            utils.ChunkRead(index=2, num_selected=3),
        ]
        result = utils.rebucket_to_stream_plan(
            canonical, min_chunk=3, stream_chunk_size=9
        )
        assert len(result) == 1
        entry = result[0]
        assert entry.num_selected == 6
        # min-chunk 0 → [0, 1, 2]; min-chunk 2 → [6, 7, 8]; not contiguous.
        assert isinstance(entry.selection, np.ndarray)
        assert_array_equal(entry.selection, [0, 1, 2, 6, 7, 8])

    def test_rejects_misaligned_stream_chunk_size(self):
        with pytest.raises(ValueError, match="positive multiple"):
            utils.rebucket_to_stream_plan([], min_chunk=3, stream_chunk_size=5)

    def test_rejects_zero_stream_chunk_size(self):
        with pytest.raises(ValueError, match="positive multiple"):
            utils.rebucket_to_stream_plan([], min_chunk=3, stream_chunk_size=0)

    def test_empty_input(self):
        result = utils.rebucket_to_stream_plan([], min_chunk=3, stream_chunk_size=6)
        assert result == []


class TestValidateVariantsAxisChunking:
    """``validate_variants_axis_chunking`` asserts that every
    variant-axis field's ``chunks[0]`` is a positive integer multiple
    of ``min_chunk``. The validator is scoped to the variants axis:
    fields whose first dimension is not ``variants`` are skipped
    entirely, and the samples-axis (``chunks[1]``) chunking of
    ``call_*`` fields is not checked here.
    """

    @staticmethod
    def _root():
        store = zarr.storage.MemoryStore()
        return zarr.group(store=store, zarr_format=3)

    @staticmethod
    def _add(root, name, shape, chunks, dims):
        root.create_array(
            name, shape=shape, chunks=chunks, dtype="i4", dimension_names=dims
        )

    def test_zero_min_chunk_raises(self):
        with pytest.raises(ValueError, match="min_chunk must be positive"):
            utils.validate_variants_axis_chunking(self._root(), 0)

    def test_negative_min_chunk_raises(self):
        with pytest.raises(ValueError, match="min_chunk must be positive"):
            utils.validate_variants_axis_chunking(self._root(), -5)

    def test_empty_root_passes(self):
        # No variant-axis fields → vacuously valid.
        utils.validate_variants_axis_chunking(self._root(), 3)

    def test_uniform_variant_axis_chunks_pass(self):
        root = self._root()
        self._add(root, "variant_position", (12,), (3,), ("variants",))
        self._add(
            root,
            "call_genotype",
            (12, 4, 2),
            (3, 4, 2),
            ("variants", "samples", "ploidy"),
        )
        utils.validate_variants_axis_chunking(root, 3)

    def test_variant_only_at_multiple_of_min_chunk_passes(self):
        # variant_position chunked at 6 = 2 * min_chunk(3).
        root = self._root()
        self._add(root, "variant_position", (12,), (6,), ("variants",))
        self._add(
            root,
            "call_genotype",
            (12, 4, 2),
            (3, 4, 2),
            ("variants", "samples", "ploidy"),
        )
        utils.validate_variants_axis_chunking(root, 3)

    def test_min_chunk_one_accepts_any_size(self):
        # Every positive int is a multiple of 1 → all chunk sizes pass.
        root = self._root()
        self._add(root, "variant_position", (10,), (7,), ("variants",))
        utils.validate_variants_axis_chunking(root, 1)

    def test_variant_only_not_multiple_raises(self):
        root = self._root()
        self._add(root, "variant_position", (10,), (5,), ("variants",))
        # min_chunk=3 does not divide 5.
        with pytest.raises(ValueError, match=r"variant_position\.chunks\[0\]=5"):
            utils.validate_variants_axis_chunking(root, 3)

    def test_call_field_not_multiple_raises(self):
        root = self._root()
        self._add(root, "call_DP", (10, 4), (7, 4), ("variants", "samples"))
        with pytest.raises(ValueError, match=r"call_DP\.chunks\[0\]=7"):
            utils.validate_variants_axis_chunking(root, 5)

    def test_static_fields_skipped(self):
        # sample_id and contig_id have no variants axis; their chunking
        # is ignored regardless of min_chunk.
        root = self._root()
        self._add(root, "sample_id", (4,), (1,), ("samples",))
        self._add(root, "contig_id", (3,), (2,), ("contigs",))
        utils.validate_variants_axis_chunking(root, 5)

    def test_samples_only_field_alongside_variants_field_ignored(self):
        # sample_id.chunks[0]=1 would fail the rule on the variants axis
        # but lives on the samples axis, so it's skipped.
        root = self._root()
        self._add(root, "variant_position", (10,), (5,), ("variants",))
        self._add(root, "sample_id", (4,), (1,), ("samples",))
        utils.validate_variants_axis_chunking(root, 5)

    def test_call_fields_with_distinct_samples_axis_chunks_pass(self):
        # call_genotype samples chunk = 4; call_DP samples chunk = 2.
        # Variants-axis chunks[0] match min_chunk; the validator does
        # NOT cross-check the samples-axis chunks[1] across call_* fields.
        root = self._root()
        self._add(
            root,
            "call_genotype",
            (9, 4, 2),
            (3, 4, 2),
            ("variants", "samples", "ploidy"),
        )
        self._add(root, "call_DP", (9, 4), (3, 2), ("variants", "samples"))
        utils.validate_variants_axis_chunking(root, 3)

    def test_call_fields_with_distinct_variant_chunks_pass_when_multiples(self):
        # call_genotype.chunks[0]=3 (= min_chunk); call_DP.chunks[0]=6
        # (= 2 * min_chunk). Both are multiples of min_chunk, so this
        # passes. (compute_min_variants_chunk_size would still reject
        # disagreeing call_* sizes, but that's a separate validator.)
        root = self._root()
        self._add(
            root,
            "call_genotype",
            (12, 2, 2),
            (3, 2, 2),
            ("variants", "samples", "ploidy"),
        )
        self._add(root, "call_DP", (12, 2), (6, 2), ("variants", "samples"))
        utils.validate_variants_axis_chunking(root, 3)

    def test_call_field_violates_while_others_match_raises(self):
        # call_DP has chunks[0]=7, which is not a multiple of min_chunk=3;
        # the validator surfaces that specific field by name even though
        # call_genotype is fine.
        root = self._root()
        self._add(
            root,
            "call_genotype",
            (12, 2, 2),
            (3, 2, 2),
            ("variants", "samples", "ploidy"),
        )
        self._add(root, "call_DP", (12, 2), (7, 2), ("variants", "samples"))
        with pytest.raises(ValueError, match=r"call_DP\.chunks\[0\]=7"):
            utils.validate_variants_axis_chunking(root, 3)


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
    """Matrix coverage of ``open_zarr`` across every (backend, path-type)
    combination.
    """

    def _write_minimal_group(self, path):
        root = zarr.open(path, mode="w")
        root.create_array("variant_position", shape=(4,), dtype="int32")
        root["variant_position"][:] = [10, 20, 30, 40]

    # --- Path-type detection (auto backend) ---

    def test_zip_path(self):
        root = open_zarr(FIXTURE_VCZ_ZIP)
        assert isinstance(root.store, zarr.storage.ZipStore)
        assert root["sample_id"][:].tolist() == ["NA00001", "NA00002", "NA00003"]

    def test_zip_str(self):
        root = open_zarr(str(FIXTURE_VCZ_ZIP))
        assert isinstance(root.store, zarr.storage.ZipStore)

    def test_local_dir_path_uses_local_store(self, tmp_path):
        vcz = tmp_path / "minimal.vcz"
        self._write_minimal_group(vcz)
        root = open_zarr(vcz)
        assert isinstance(root.store, zarr.storage.LocalStore)
        assert root["variant_position"][:].tolist() == [10, 20, 30, 40]

    def test_local_dir_str_uses_local_store(self, tmp_path):
        vcz = tmp_path / "minimal.vcz"
        self._write_minimal_group(vcz)
        root = open_zarr(str(vcz))
        assert isinstance(root.store, zarr.storage.LocalStore)

    def test_url_with_default_backend_raises(self, tmp_path):
        # The default backend is local-only; URLs require an explicit
        # backend_storage value.
        vcz = tmp_path / "minimal.vcz"
        self._write_minimal_group(vcz)
        with pytest.raises(ValueError, match="requires backend_storage"):
            open_zarr(vcz.as_uri())

    def test_zarr_group_passthrough(self):
        store = zarr.storage.MemoryStore()
        group = zarr.group(store=store, zarr_format=3)
        assert open_zarr(group) is group

    def test_zarr_store_passthrough(self):
        # An already-built store short-circuits the backend dispatch:
        # the resulting Group reads from that store regardless of which
        # backend produced it.
        store = zarr.storage.MemoryStore()
        group = zarr.group(store=store, zarr_format=3)
        group.create_array("variant_position", shape=(3,), dtype="int32")
        group["variant_position"][:] = [1, 2, 3]
        root = open_zarr(store)
        assert root["variant_position"][:].tolist() == [1, 2, 3]

    @pytest.mark.parametrize(
        ("zarr_format", "expected_zarr_format"), [(2, 2), (3, 3), (None, 3)]
    )
    def test_zarr_format(self, tmp_path, zarr_format, expected_zarr_format):
        vcz = tmp_path / "minimal.vcz"
        root = open_zarr(vcz, mode="w", zarr_format=zarr_format)
        root.create_array("variant_position", shape=(4,), dtype="int32")
        root["variant_position"][:] = [10, 20, 30, 40]

        root = open_zarr(vcz)
        assert root.metadata.zarr_format == expected_zarr_format
        assert isinstance(root.store, zarr.storage.LocalStore)
        assert root["variant_position"][:].tolist() == [10, 20, 30, 40]

    # --- Backend selection ---

    def test_fsspec_backend_uses_fsspec_store(self, tmp_path):
        vcz = tmp_path / "minimal.vcz"
        self._write_minimal_group(vcz)
        root = open_zarr(vcz.as_uri(), backend_storage="fsspec")
        assert isinstance(root.store, zarr.storage.FsspecStore)
        assert root["variant_position"][:].tolist() == [10, 20, 30, 40]

    def test_local_path_with_fsspec_backend_promoted_to_file_url(self, tmp_path):
        # Local Path/str inputs to the fsspec backend are auto-promoted
        # to file:// URIs so FsspecStore.from_url accepts them.
        vcz = tmp_path / "minimal.vcz"
        self._write_minimal_group(vcz)
        root_path = open_zarr(vcz, backend_storage="fsspec")
        assert isinstance(root_path.store, zarr.storage.FsspecStore)
        root_str = open_zarr(str(vcz), backend_storage="fsspec")
        assert isinstance(root_str.store, zarr.storage.FsspecStore)

    def test_fsspec_backend_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported file_or_url type"):
            open_zarr(42, backend_storage="fsspec")

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unsupported Zarr backend storage"):
            open_zarr(FIXTURE_VCZ_ZIP, backend_storage="bogus")

    def test_local_backend_unsupported_type_raises(self):
        # The default (local) backend rejects anything that isn't a
        # str/Path-like via the underlying LocalStore.
        with pytest.raises((TypeError, ValueError)):
            open_zarr(42)

    def test_obstore_local_path(self, tmp_path):
        vcz = tmp_path / "minimal.vcz"
        self._write_minimal_group(vcz)
        root = open_zarr(vcz, backend_storage="obstore")
        assert isinstance(root.store, zarr.storage.ObjectStore)
        assert root["variant_position"][:].tolist() == [10, 20, 30, 40]

    def test_obstore_local_str(self, tmp_path):
        vcz = tmp_path / "minimal.vcz"
        self._write_minimal_group(vcz)
        root = open_zarr(str(vcz), backend_storage="obstore")
        assert isinstance(root.store, zarr.storage.ObjectStore)
        assert root["variant_position"][:].tolist() == [10, 20, 30, 40]

    def test_obstore_passthrough_store_object(self, tmp_path):
        vcz = tmp_path / "minimal.vcz"
        self._write_minimal_group(vcz)
        first = open_zarr(vcz, backend_storage="obstore")
        second = open_zarr(first.store, backend_storage="obstore")
        assert isinstance(second.store, zarr.storage.ObjectStore)
        assert second["variant_position"][:].tolist() == [10, 20, 30, 40]

    def test_icechunk_local_path(self, tmp_path):
        vcz_dir = tmp_path / "minimal.vcz"
        self._write_minimal_group(vcz_dir)
        ic_path = to_vcz_icechunk(vcz_dir, tmp_path)
        root = open_zarr(ic_path, backend_storage="icechunk")
        assert root["variant_position"][:].tolist() == [10, 20, 30, 40]

    def test_nonexistent_zip_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            open_zarr(tmp_path / "does-not-exist.vcz.zip")

    # --- missing optional-extra dependencies ---

    def test_obstore_missing_install_hint(self, monkeypatch, tmp_path):
        vcz = tmp_path / "minimal.vcz"
        self._write_minimal_group(vcz)
        monkeypatch.setitem(sys.modules, "obstore", None)
        with pytest.raises(ImportError, match=r"pip install vcztools\[obstore\]"):
            open_zarr(vcz, backend_storage="obstore")

    def test_icechunk_missing_install_hint(self, monkeypatch, tmp_path):
        vcz = tmp_path / "minimal.vcz"
        self._write_minimal_group(vcz)
        monkeypatch.setitem(sys.modules, "icechunk", None)
        with pytest.raises(ImportError, match=r"pip install vcztools\[icechunk\]"):
            open_zarr(vcz, backend_storage="icechunk")

    # --- storage_options plumbing ---

    def test_storage_options_rejected_for_local_zip(self):
        # The default backend is local-only — neither LocalStore nor
        # ZipStore takes resilience options. Both cases raise.
        with pytest.raises(ValueError, match="not supported for local stores"):
            open_zarr(FIXTURE_VCZ_ZIP, storage_options={"foo": "bar"})

    def test_storage_options_rejected_for_local_dir(self, tmp_path):
        vcz = tmp_path / "minimal.vcz"
        self._write_minimal_group(vcz)
        with pytest.raises(ValueError, match="not supported for local stores"):
            open_zarr(vcz, storage_options={"foo": "bar"})

    def test_fsspec_storage_options_forwarded(self, monkeypatch, tmp_path):
        vcz = tmp_path / "minimal.vcz"
        self._write_minimal_group(vcz)
        captured = {}
        original = zarr.storage.FsspecStore.from_url

        def spy(url, *, storage_options=None, **kwargs):
            captured["url"] = url
            captured["storage_options"] = storage_options
            return original(url)

        monkeypatch.setattr(zarr.storage.FsspecStore, "from_url", spy)
        open_zarr(
            vcz.as_uri(),
            backend_storage="fsspec",
            storage_options={"foo": "bar"},
        )
        assert captured["storage_options"] == {"foo": "bar"}

    def test_storage_options_forwarded_to_obstore(self, monkeypatch, tmp_path):
        vcz = tmp_path / "minimal.vcz"
        self._write_minimal_group(vcz)
        captured = {}
        original = obs.store.from_url

        def spy(url, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return original(url, mkdir=kwargs.get("mkdir", False))

        monkeypatch.setattr(obs.store, "from_url", spy)
        open_zarr(
            vcz,
            backend_storage="obstore",
            storage_options={"client_options": {"timeout": "30s"}},
        )
        assert captured["kwargs"]["client_options"] == {"timeout": "30s"}

    def test_storage_options_forwarded_to_icechunk_s3(self, monkeypatch):

        captured = {}

        def spy(*, bucket, prefix, from_env, **kwargs):
            captured.update(
                bucket=bucket, prefix=prefix, from_env=from_env, kwargs=kwargs
            )
            return object()  # opaque; we're only verifying forwarding

        monkeypatch.setattr(ic, "s3_storage", spy)
        utils.make_icechunk_storage(
            "s3://bucket/prefix", storage_options={"region_name": "us-east-1"}
        )
        assert captured["bucket"] == "bucket"
        assert captured["prefix"] == "prefix"
        assert captured["kwargs"] == {"region_name": "us-east-1"}

    def test_storage_options_forwarded_to_icechunk_azure(self, monkeypatch):

        captured = {}

        def spy(*, account, container, prefix, from_env, **kwargs):
            captured.update(
                account=account,
                container=container,
                prefix=prefix,
                from_env=from_env,
                kwargs=kwargs,
            )
            return object()

        monkeypatch.setattr(ic, "azure_storage", spy)
        utils.make_icechunk_storage(
            "az://account/container/prefix",
            storage_options={"account_key": "secret"},
        )
        assert captured["account"] == "account"
        assert captured["container"] == "container"
        assert captured["prefix"] == "prefix"
        assert captured["kwargs"] == {"account_key": "secret"}

    def test_storage_options_rejected_for_icechunk_local_str(self, tmp_path):
        with pytest.raises(
            ValueError, match="not supported for local icechunk storage"
        ):
            utils.make_icechunk_storage(str(tmp_path), storage_options={"foo": "bar"})

    def test_storage_options_rejected_for_icechunk_local_path(self, tmp_path):
        with pytest.raises(
            ValueError, match="not supported for local icechunk storage"
        ):
            utils.make_icechunk_storage(tmp_path, storage_options={"foo": "bar"})

    # --- Backend error/edge paths ---

    def test_obstore_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported file_or_url type"):
            open_zarr(42, backend_storage="obstore")

    def test_obstore_remote_url_storage_options_forwarded(self, monkeypatch):
        captured = {}

        def spy(url, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            raise RuntimeError("captured")

        monkeypatch.setattr(obs.store, "from_url", spy)
        with pytest.raises(RuntimeError, match="captured"):
            open_zarr(
                "s3://bucket/path",
                backend_storage="obstore",
                storage_options={"region_name": "us-east-1"},
            )
        assert captured["url"] == "s3://bucket/path"
        assert captured["kwargs"] == {"region_name": "us-east-1"}

    def test_icechunk_store_passthrough(self, monkeypatch, tmp_path):
        vcz_dir = tmp_path / "minimal.vcz"
        self._write_minimal_group(vcz_dir)
        ic_path = to_vcz_icechunk(vcz_dir, tmp_path)
        # Build the IcechunkStore once, then pass it back through.
        first = open_zarr(ic_path, backend_storage="icechunk")
        assert isinstance(first.store, ic.store.IcechunkStore)
        second = open_zarr(first.store, backend_storage="icechunk")
        assert second.store is first.store

    def test_icechunk_local_str_returns_storage(self, monkeypatch, tmp_path):

        captured = {}

        def spy(path):
            captured["path"] = path
            return "local-storage"

        monkeypatch.setattr(ic.Storage, "new_local_filesystem", spy)
        result = utils.make_icechunk_storage(str(tmp_path))
        assert result == "local-storage"
        assert captured["path"] == str(tmp_path)

    def test_icechunk_unsupported_url_raises(self):
        with pytest.raises(ValueError, match="Unsupported URL for icechunk"):
            utils.make_icechunk_storage("ftp://host/path")

    def test_icechunk_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported URL type for icechunk"):
            utils.make_icechunk_storage(42)

    def test_azure_az_url_missing_container_raises(self):
        # az://account/ has an account but no container path segment.
        with pytest.raises(ValueError, match="must include a container name"):
            utils.make_icechunk_storage("az://account/")

    def test_azure_az_url_no_account_raises(self):
        with pytest.raises(ValueError, match="must use the form"):
            utils.make_icechunk_storage("az:///container/prefix")

    def test_azure_abfs_url_missing_at_raises(self):
        with pytest.raises(ValueError, match="ABFS Icechunk URLs"):
            utils.make_icechunk_storage("abfs://container/prefix")

    def test_azure_unsupported_https_url_raises(self):
        with pytest.raises(ValueError, match="Unsupported Azure URL"):
            # https:// must end in *.blob.core.windows.net or
            # *.dfs.core.windows.net — anything else is rejected.
            utils.make_icechunk_storage("https://example.com/foo/bar")
