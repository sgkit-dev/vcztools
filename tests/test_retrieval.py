import numpy as np
import numpy.testing as nt
import pandas as pd
import pytest

from tests import vcz_builder
from tests.utils import make_reader
from vcztools import regions as regions_mod
from vcztools import samples as samples_mod
from vcztools import utils
from vcztools.bcftools_filter import BcftoolsFilter
from vcztools.retrieval import CachedChunk, VczReader


def test_variant_chunks(fx_sample_vcz):
    reader = make_reader(
        fx_sample_vcz.group,
        regions="20:1230236-",
        samples=["NA00002", "NA00003"],
        include="FMT/DP>3",
        filter_on_subset_samples=False,
    )
    chunk_data = next(
        reader.variant_chunks(
            fields=["variant_contig", "variant_position", "call_DP", "call_GQ"],
        )
    )
    nt.assert_array_equal(chunk_data["variant_contig"], [1, 1])
    nt.assert_array_equal(chunk_data["variant_position"], [1230237, 1234567])
    nt.assert_array_equal(chunk_data["call_DP"], [[4, 2], [2, 3]])
    nt.assert_array_equal(chunk_data["call_GQ"], [[48, 61], [17, 40]])
    # note second site (at pos 1234567) is included even though both samples in mask
    # are False (NA00002 and NA00003), since sample NA00001 matched filter criteria,
    # but was then removed by samples_selection
    nt.assert_array_equal(
        chunk_data["sample_filter_mask"], [[True, False], [False, False]]
    )


def test_variant_chunks_empty_fields(fx_sample_vcz):
    reader = VczReader(fx_sample_vcz.group)
    with pytest.raises(StopIteration):
        next(reader.variant_chunks(fields=[]))


@pytest.mark.parametrize(
    ("regions", "samples"),
    [
        ("20:1230236-", ["NA00002", "NA00003"]),
        (["20:1230236-"], ["NA00002", "NA00003"]),
    ],
)
def test_variant_iter(fx_sample_vcz, regions, samples):
    reader = make_reader(
        fx_sample_vcz.group,
        regions=regions,
        samples=samples,
        include="FMT/DP>3",
        filter_on_subset_samples=False,
    )
    it = reader.variants(
        fields=["variant_contig", "variant_position", "call_DP", "call_GQ"],
    )

    variant1 = next(it)
    assert variant1["variant_contig"] == 1
    assert variant1["variant_position"] == 1230237
    nt.assert_array_equal(variant1["call_DP"], [4, 2])
    nt.assert_array_equal(variant1["call_GQ"], [48, 61])
    nt.assert_array_equal(variant1["sample_filter_mask"], [True, False])

    variant2 = next(it)
    assert variant2["variant_contig"] == 1
    assert variant2["variant_position"] == 1234567
    nt.assert_array_equal(variant2["call_DP"], [2, 3])
    nt.assert_array_equal(variant2["call_GQ"], [17, 40])
    nt.assert_array_equal(variant2["sample_filter_mask"], [False, False])

    with pytest.raises(StopIteration):
        next(it)


def test_variant_iter_empty_fields(fx_sample_vcz):
    reader = VczReader(fx_sample_vcz.group)
    with pytest.raises(StopIteration):
        next(reader.variants(fields=[]))


def _make_filter_vcz(num_variants=9, variants_chunk_size=3):
    """Build a multi-chunk VCZ with two filters for FILTER expression tests."""
    # Alternate PASS-only and q10-only rows so every chunk has both kinds.
    variant_filter = np.zeros((num_variants, 2), dtype=bool)
    for i in range(num_variants):
        if i % 2 == 0:
            variant_filter[i, 0] = True  # PASS
        else:
            variant_filter[i, 1] = True  # q10
    return vcz_builder.make_vcz(
        variant_contig=[0] * num_variants,
        variant_position=list(range(100, 100 + num_variants)),
        alleles=[("A", "T")] * num_variants,
        filters=("PASS", "q10"),
        variant_filter=variant_filter,
        variants_chunk_size=variants_chunk_size,
    )


class TestFilterMultiChunk:
    """Regression tests for VariantChunkReader with mixed-axis fields.

    FILTER expressions reference both variant_filter (variant-chunked) and
    filter_id (single-chunk on the filters axis). VariantChunkReader must
    handle this without IndexError or silent data loss.
    """

    @staticmethod
    def _chunks(root, **kwargs):
        reader = make_reader(root, **kwargs)
        return reader.variant_chunks(fields=["variant_position"])

    def test_include_filter_eq_pass(self):
        results = list(self._chunks(_make_filter_vcz(), include='FILTER="PASS"'))
        positions = np.concatenate([chunk["variant_position"] for chunk in results])
        # Even-indexed variants (0, 2, 4, 6, 8) are PASS → positions 100, 102, ...
        nt.assert_array_equal(positions, [100, 102, 104, 106, 108])

    def test_include_filter_ne_pass(self):
        results = list(self._chunks(_make_filter_vcz(), include='FILTER!="PASS"'))
        positions = np.concatenate([chunk["variant_position"] for chunk in results])
        # Odd-indexed variants are q10 → positions 101, 103, ...
        nt.assert_array_equal(positions, [101, 103, 105, 107])

    def test_exclude_filter_pass(self):
        results = list(self._chunks(_make_filter_vcz(), exclude='FILTER="PASS"'))
        positions = np.concatenate([chunk["variant_position"] for chunk in results])
        nt.assert_array_equal(positions, [101, 103, 105, 107])

    def test_filter_subset_match(self):
        results = list(self._chunks(_make_filter_vcz(), include='FILTER~"q10"'))
        positions = np.concatenate([chunk["variant_position"] for chunk in results])
        nt.assert_array_equal(positions, [101, 103, 105, 107])

    def test_filter_with_regions(self):
        root = _make_filter_vcz()
        reader = make_reader(root, regions="chr1:104-107", include='FILTER="PASS"')
        results = list(reader.variant_chunks(fields=["variant_position"]))
        positions = np.concatenate([chunk["variant_position"] for chunk in results])
        # Positions 104-107 intersected with PASS (even) → 104, 106
        nt.assert_array_equal(positions, [104, 106])

    def test_chunk_reader_static_fields_across_chunks(self):
        """Static fields read identically from every CachedChunk."""
        root = _make_filter_vcz()
        num_chunks = int(root["variant_filter"].cdata_shape[0])
        assert num_chunks == 3
        for chunk_idx in range(num_chunks):
            chunk = CachedChunk(
                root,
                utils.ChunkRead(index=chunk_idx),
                read_plan=None,
                output_columns=None,
            )
            nt.assert_array_equal(chunk.filter_view("filter_id"), ["PASS", "q10"])


class TestVczReaderRegions:
    """Cover the three accepted region/target input shapes plus error paths.

    Region/target parsing now lives in
    :func:`vcztools.regions.build_chunk_plan`; these tests exercise it
    directly for validation and round-trip the happy paths through
    :class:`VczReader` via the ``make_reader`` helper.
    """

    @staticmethod
    def _vcz():
        # 10 variants on chr1 at positions 1..10.
        return vcz_builder.make_vcz(
            variant_contig=[0] * 10,
            variant_position=list(range(1, 11)),
            alleles=[("A", "T")] * 10,
            contigs=("chr1",),
            variants_chunk_size=3,
        )

    def _positions(self, reader):
        chunks = list(reader.variant_chunks(fields=["variant_position"]))
        if not chunks:
            return np.array([], dtype=np.int32)
        return np.concatenate([c["variant_position"] for c in chunks])

    def test_regions_string(self):
        reader = make_reader(self._vcz(), regions="chr1:3-5")
        nt.assert_array_equal(self._positions(reader), [3, 4, 5])

    def test_regions_list_of_strings(self):
        reader = make_reader(self._vcz(), regions=["chr1:3-5", "chr1:8-9"])
        nt.assert_array_equal(self._positions(reader), [3, 4, 5, 8, 9])

    def test_regions_dataframe(self):
        df = pd.DataFrame(
            {
                "contig": ["chr1"],
                "start": pd.array([3], dtype="Int64"),
                "end": pd.array([5], dtype="Int64"),
            }
        )
        reader = make_reader(self._vcz(), regions=df)
        nt.assert_array_equal(self._positions(reader), [3, 4, 5])

    def test_regions_dataframe_with_na_end(self):
        df = pd.DataFrame(
            {
                "contig": ["chr1"],
                "start": pd.array([8], dtype="Int64"),
                "end": pd.array([pd.NA], dtype="Int64"),
            }
        )
        reader = make_reader(self._vcz(), regions=df)
        nt.assert_array_equal(self._positions(reader), [8, 9, 10])

    def test_targets_complement_flag(self):
        reader = make_reader(self._vcz(), targets="chr1:3-5", targets_complement=True)
        nt.assert_array_equal(self._positions(reader), [1, 2, 6, 7, 8, 9, 10])

    def test_regions_rejects_caret_prefix(self):
        with pytest.raises(ValueError, match="targets_complement=True"):
            regions_mod.build_chunk_plan(self._vcz(), regions="^chr1:1-3")

    def test_targets_rejects_caret_prefix(self):
        with pytest.raises(ValueError, match="targets_complement=True"):
            regions_mod.build_chunk_plan(self._vcz(), targets="^chr1:1-3")

    def test_regions_rejects_comma(self):
        with pytest.raises(ValueError, match=r"regions string .* contains ','"):
            regions_mod.build_chunk_plan(self._vcz(), regions="chr1:1-3,chr1:5-7")

    def test_targets_rejects_comma(self):
        with pytest.raises(ValueError, match=r"targets string .* contains ','"):
            regions_mod.build_chunk_plan(self._vcz(), targets="chr1:1-3,chr1:5-7")

    def test_regions_invalid_type(self):
        with pytest.raises(TypeError, match="regions must be"):
            regions_mod.build_chunk_plan(self._vcz(), regions=42)

    def test_targets_invalid_type(self):
        with pytest.raises(TypeError, match="targets must be"):
            regions_mod.build_chunk_plan(self._vcz(), targets=42)

    def test_regions_dataframe_missing_columns(self):
        df = pd.DataFrame({"contig": ["chr1"], "start": pd.array([1], dtype="Int64")})
        with pytest.raises(ValueError, match="missing required columns.*end"):
            regions_mod.build_chunk_plan(self._vcz(), regions=df)

    def test_flat_index_array_accepted(self):
        """``set_variants(np.ndarray)`` buckets indexes into a plan."""
        reader = VczReader(self._vcz())
        reader.set_variants(np.array([2, 4, 7], dtype=np.int64))
        nt.assert_array_equal(self._positions(reader), [3, 5, 8])


class TestVczReaderSamples:
    """Cover VczReader sample input: default, integer-index list, error cases."""

    def test_samples_default_selects_all(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        nt.assert_array_equal(reader.sample_ids, ["NA00001", "NA00002", "NA00003"])

    def test_samples_list(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        reader.set_samples([0, 2])
        nt.assert_array_equal(reader.sample_ids, ["NA00001", "NA00003"])
        nt.assert_array_equal(reader.samples_selection, [0, 2])

    def test_samples_preserves_input_order(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        reader.set_samples([2, 0])
        nt.assert_array_equal(reader.sample_ids, ["NA00003", "NA00001"])
        nt.assert_array_equal(reader.samples_selection, [2, 0])

    def test_samples_rejects_string_input(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        with pytest.raises(TypeError, match="integer indexes"):
            reader.set_samples(["NA00001"])

    def test_samples_rejects_string_scalar(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        with pytest.raises(TypeError, match="integer indexes"):
            reader.set_samples("NA00001")

    def test_samples_rejects_non_integer_numpy_array(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        with pytest.raises(TypeError, match="integer indexes"):
            reader.set_samples(np.array([0.0, 2.0]))

    def test_samples_accepts_numpy_int_array(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        reader.set_samples(np.array([0, 2], dtype=np.int64))
        nt.assert_array_equal(reader.sample_ids, ["NA00001", "NA00003"])

    def test_samples_out_of_range_raises(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        with pytest.raises(ValueError, match="sample index out of range"):
            reader.set_samples([0, 99])

    def test_samples_negative_raises(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        with pytest.raises(ValueError, match="sample index out of range"):
            reader.set_samples([-1])

    def test_samples_empty_list(self, fx_sample_vcz):
        # Post-resolve, an empty list means "no samples" (e.g. all
        # requested names were dropped by ignore_missing_samples).
        reader = VczReader(fx_sample_vcz.group)
        reader.set_samples([])
        nt.assert_array_equal(reader.sample_ids, [])
        assert reader.sample_chunk_plan is None


class TestVczReaderSampleChunks:
    """End-to-end sample-chunk pruning.

    Builds a VCZ whose sample axis spans multiple chunks and verifies
    that selecting subsets yields correct ``call_*`` data regardless of
    whether the selection hits one chunk, several chunks, or the
    last (possibly partial) chunk.
    """

    @staticmethod
    def _vcz(num_samples=6, samples_chunk_size=2):
        sample_ids = [f"s{i}" for i in range(num_samples)]
        # call_DP[i, j] = i * 10 + j — unique per (variant, sample) so
        # mis-indexing is caught immediately.
        call_dp = np.array(
            [[i * 10 + j for j in range(num_samples)] for i in range(3)],
            dtype=np.int32,
        )
        return vcz_builder.make_vcz(
            variant_contig=[0, 0, 0],
            variant_position=[1, 2, 3],
            alleles=[("A", "T")] * 3,
            num_samples=num_samples,
            sample_id=sample_ids,
            samples_chunk_size=samples_chunk_size,
            call_fields={"DP": call_dp},
        )

    def _call_dp(self, reader):
        chunks = list(reader.variant_chunks(fields=["call_DP"]))
        return np.concatenate([c["call_DP"] for c in chunks], axis=0)

    @staticmethod
    def _plan_chunk_indexes(plan):
        return [cr.index for cr in plan.chunk_reads]

    def test_single_chunk_selection(self):
        # s2, s3 both live in sample chunk 1 (indexes 2, 3).
        reader = VczReader(self._vcz())
        reader.set_samples([2, 3])
        plan = reader.sample_chunk_plan
        assert self._plan_chunk_indexes(plan) == [1]
        nt.assert_array_equal(plan.chunk_reads[0].selection, [0, 1])
        assert plan.permutation is None
        dp = self._call_dp(reader)
        nt.assert_array_equal(dp, [[2, 3], [12, 13], [22, 23]])

    def test_multi_chunk_selection(self):
        # s1 is in chunk 0; s4 is in chunk 2; chunk 1 is skipped.
        reader = VczReader(self._vcz())
        reader.set_samples([1, 4])
        plan = reader.sample_chunk_plan
        assert self._plan_chunk_indexes(plan) == [0, 2]
        assert plan.permutation is None
        dp = self._call_dp(reader)
        nt.assert_array_equal(dp, [[1, 4], [11, 14], [21, 24]])

    def test_preserves_user_order(self):
        # Same chunks as the multi-chunk test, but the user order is
        # reversed — the output must follow the input list via the
        # plan's permutation.
        reader = VczReader(self._vcz())
        reader.set_samples([4, 1])
        plan = reader.sample_chunk_plan
        assert self._plan_chunk_indexes(plan) == [0, 2]
        nt.assert_array_equal(plan.permutation, [1, 0])
        dp = self._call_dp(reader)
        nt.assert_array_equal(dp, [[4, 1], [14, 11], [24, 21]])

    def test_partial_final_chunk(self):
        # 5 samples with chunk size 2 → chunks sized [2, 2, 1]. s4 sits
        # alone in the final chunk.
        reader = VczReader(self._vcz(num_samples=5, samples_chunk_size=2))
        reader.set_samples([4])
        plan = reader.sample_chunk_plan
        assert self._plan_chunk_indexes(plan) == [2]
        nt.assert_array_equal(plan.chunk_reads[0].selection, [0])
        assert plan.permutation is None
        dp = self._call_dp(reader)
        nt.assert_array_equal(dp, [[4], [14], [24]])

    def test_default_plan_is_identity(self):
        # Default (no set_samples call) plans a read over every sample
        # chunk with each full-chunk selection.
        reader = VczReader(self._vcz())
        plan = reader.sample_chunk_plan
        assert self._plan_chunk_indexes(plan) == [0, 1, 2]
        assert plan.permutation is None
        dp = self._call_dp(reader)
        nt.assert_array_equal(dp.shape, (3, 6))
        nt.assert_array_equal(dp[0], [0, 1, 2, 3, 4, 5])

    def test_default_masked_header_reduces(self):
        # Default with masked "" entries in sample_id: the plan drops
        # the masked indices, matching the old post-filter path.
        root = self._vcz()
        root["sample_id"][:2] = ""
        reader = VczReader(root)
        nt.assert_array_equal(reader.sample_ids, ["s2", "s3", "s4", "s5"])
        dp = self._call_dp(reader)
        nt.assert_array_equal(dp, [[2, 3, 4, 5], [12, 13, 14, 15], [22, 23, 24, 25]])

    def test_empty_samples_list_has_no_plan(self):
        # An empty samples list (via --force-samples dropping every
        # requested unknown, or --drop-genotypes) means no sample-chunk
        # plan — full call arrays are read so AC/AN can still be
        # recomputed.
        reader = VczReader(self._vcz())
        reader.set_samples([])
        assert reader.sample_chunk_plan is None


def _vcz_for_cache_tests(
    num_samples=4, samples_chunk_size=2, variants_chunk_size=2, num_variants=4
):
    call_dp = np.array(
        [[v * 10 + s for s in range(num_samples)] for v in range(num_variants)],
        dtype=np.int32,
    )
    return vcz_builder.make_vcz(
        variant_contig=[0] * num_variants,
        variant_position=list(range(1, num_variants + 1)),
        alleles=[("A", "T")] * num_variants,
        num_samples=num_samples,
        sample_id=[f"s{i}" for i in range(num_samples)],
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
        call_fields={"DP": call_dp},
    )


def _make_cached_chunk(
    root,
    *,
    variant_chunk_idx=0,
    variant_selection=None,
    samples_selection=None,
    filter_on_subset_samples=True,
):
    all_sample_ids = root["sample_id"][:]
    real_indices = np.flatnonzero(all_sample_ids != "")
    samples_chunk_size = int(root["sample_id"].chunks[0])
    if samples_selection is None:
        samples_selection = real_indices
    samples_selection = np.asarray(samples_selection, dtype=np.int64)
    subset_plan = (
        samples_mod.build_chunk_plan(
            samples_selection, samples_chunk_size=samples_chunk_size
        )
        if samples_selection.size > 0
        else None
    )
    real_plan = (
        samples_mod.build_chunk_plan(
            real_indices, samples_chunk_size=samples_chunk_size
        )
        if real_indices.size > 0
        else None
    )
    if filter_on_subset_samples:
        read_plan = subset_plan
        output_columns = None
    else:
        read_plan = real_plan
        output_columns = np.searchsorted(real_indices, samples_selection)
    return CachedChunk(
        root,
        utils.ChunkRead(index=variant_chunk_idx, selection=variant_selection),
        read_plan=read_plan,
        output_columns=output_columns,
    )


class TestCachedChunkCache:
    """CachedChunk reads each (field, sample_chunk) block at most
    once per variant-chunk visit across filter_view / output_view."""

    def test_view_mode_shared_field_raw_blocks_not_refetched(self):
        # View-mode: filter_view reads real-axis blocks; output_view
        # then reads subset-axis blocks. Any (field, sci) needed by
        # both comes from the raw cache (identity check).
        root = _vcz_for_cache_tests(num_samples=6, samples_chunk_size=2)
        chunk = _make_cached_chunk(
            root,
            samples_selection=np.array([1, 5]),
            filter_on_subset_samples=False,
        )
        chunk.filter_view("call_DP")
        # Real plan covers all three sample chunks.
        raw_sci = {k[1] for k in chunk._raw if len(k) == 2 and k[0] == "call_DP"}
        assert raw_sci == {0, 1, 2}
        block_0 = chunk._raw[("call_DP", 0)]
        block_2 = chunk._raw[("call_DP", 2)]
        chunk.output_view("call_DP")  # subset plan: chunks 0 and 2
        # No refetch; same underlying blocks.
        assert chunk._raw[("call_DP", 0)] is block_0
        assert chunk._raw[("call_DP", 2)] is block_2

    def test_subset_mode_filter_and_output_view_share_assembled_array(self):
        # Subset-mode: filter_view and output_view use the same plan
        # object → they share a single assembled array in the view
        # cache, not merely shared raw blocks.
        root = _vcz_for_cache_tests()
        chunk = _make_cached_chunk(root, filter_on_subset_samples=True)
        fv = chunk.filter_view("call_DP")
        ov = chunk.output_view("call_DP")
        assert fv is ov

    def test_variant_field_cached_within_chunk(self):
        chunk = _make_cached_chunk(_vcz_for_cache_tests())
        first = chunk.filter_view("variant_position")
        second = chunk.output_view("variant_position")
        assert first is second

    def test_static_field_cached_within_chunk(self):
        chunk = _make_cached_chunk(_make_filter_vcz())
        first = chunk.filter_view("filter_id")
        second = chunk.filter_view("filter_id")
        assert first is second

    def test_prefetch_warms_raw_cache(self):
        root = _vcz_for_cache_tests()
        chunk = _make_cached_chunk(root)
        chunk.prefetch(["variant_position", "call_DP"])
        assert ("variant_position",) in chunk._raw
        # Subset-mode read_plan covers both sample chunks.
        assert ("call_DP", 0) in chunk._raw
        assert ("call_DP", 1) in chunk._raw


class TestCachedChunkAxes:
    """filter_view and output_view return data in the right sample axis."""

    @staticmethod
    def _vcz():
        # 4 samples, all real; variants_chunk covers both variants.
        num_samples = 4
        num_variants = 2
        call_dp = np.array(
            [[v * 10 + s for s in range(num_samples)] for v in range(num_variants)],
            dtype=np.int32,
        )
        return vcz_builder.make_vcz(
            variant_contig=[0] * num_variants,
            variant_position=list(range(1, num_variants + 1)),
            alleles=[("A", "T")] * num_variants,
            num_samples=num_samples,
            sample_id=[f"s{i}" for i in range(num_samples)],
            variants_chunk_size=num_variants,
            samples_chunk_size=2,
            call_fields={"DP": call_dp},
        )

    def test_subset_mode_filter_view_uses_subset_axis(self):
        chunk = _make_cached_chunk(
            self._vcz(),
            samples_selection=np.array([0, 2]),
            filter_on_subset_samples=True,
        )
        dp = chunk.filter_view("call_DP")
        assert dp.shape == (2, 2)
        nt.assert_array_equal(dp, [[0, 2], [10, 12]])

    def test_view_mode_filter_view_uses_real_axis(self):
        chunk = _make_cached_chunk(
            self._vcz(),
            samples_selection=np.array([0, 2]),
            filter_on_subset_samples=False,
        )
        dp = chunk.filter_view("call_DP")
        assert dp.shape == (2, 4)
        nt.assert_array_equal(dp, [[0, 1, 2, 3], [10, 11, 12, 13]])

    @pytest.mark.parametrize("mode", [True, False])
    def test_output_view_always_subset_axis(self, mode):
        chunk = _make_cached_chunk(
            self._vcz(),
            samples_selection=np.array([0, 2]),
            filter_on_subset_samples=mode,
        )
        dp = chunk.output_view("call_DP")
        assert dp.shape == (2, 2)
        nt.assert_array_equal(dp, [[0, 2], [10, 12]])


class TestVariantChunksFilterPlusSamples:
    """End-to-end: a call_* field referenced by both filter and query,
    with sample subsetting active, preserves bcftools semantics."""

    def test_custom_variant_filter_no_bcftools(self, fx_sample_vcz):
        # Hand-rolled filter implementing the VariantFilter protocol —
        # proves VczReader has no bcftools-specific coupling.
        class PositionGt:
            referenced_fields = frozenset({"variant_position"})
            scope = "variant"

            def __init__(self, threshold):
                self._threshold = threshold

            def evaluate(self, chunk_data):
                return chunk_data["variant_position"] > self._threshold

        reader = VczReader(fx_sample_vcz.group)
        reader.set_variant_filter(PositionGt(1000000))
        chunks = list(reader.variant_chunks(fields=["variant_position"]))
        positions = np.concatenate([c["variant_position"] for c in chunks])
        nt.assert_array_equal(positions, [1110696, 1230237, 1234567, 1235237])

    def test_filter_on_subset_samples_sample_scope(self, fx_sample_vcz):
        # bcftools-query-style post-subset evaluation: filter sees only
        # the selected samples, so position 1234567 (where NA00001 but
        # not NA00002/NA00003 matched FMT/DP>3) is DROPPED.
        reader = VczReader(fx_sample_vcz.group)
        reader.set_variants(
            regions_mod.build_chunk_plan(fx_sample_vcz.group, regions="20:1230236-")
        )
        reader.set_samples([1, 2])
        reader.set_variant_filter(
            BcftoolsFilter(field_names=reader.field_names, include="FMT/DP>3"),
            filter_on_subset_samples=True,
        )
        chunks = list(reader.variant_chunks(fields=["variant_position", "call_DP"]))
        chunk = chunks[0]
        assert chunk["call_DP"].shape[1] == 2
        nt.assert_array_equal(chunk["variant_position"], [1230237])

    def test_filter_on_subset_samples_no_subset_is_noop(self, fx_sample_vcz):
        # With no sample subset, filter_on_subset_samples=True must return
        # identical results to the default mode.
        root = fx_sample_vcz.group

        def build(filter_on_subset_samples):
            reader = VczReader(root)
            reader.set_variants(
                regions_mod.build_chunk_plan(root, regions="20:1230236-")
            )
            reader.set_variant_filter(
                BcftoolsFilter(field_names=reader.field_names, include="FMT/DP>3"),
                filter_on_subset_samples=filter_on_subset_samples,
            )
            return reader

        pre = list(build(False).variant_chunks(fields=["variant_position"]))
        post = list(build(True).variant_chunks(fields=["variant_position"]))
        nt.assert_array_equal(
            np.concatenate([c["variant_position"] for c in pre]),
            np.concatenate([c["variant_position"] for c in post]),
        )

    def test_filter_on_subset_samples_variant_scope_unchanged(self, fx_sample_vcz):
        # Variant-scope filters touch no sample axis; the flag is a
        # no-op regardless of subset.
        root = fx_sample_vcz.group

        def build(filter_on_subset_samples):
            reader = VczReader(root)
            reader.set_samples([0])
            reader.set_variant_filter(
                BcftoolsFilter(field_names=reader.field_names, include="POS > 1000000"),
                filter_on_subset_samples=filter_on_subset_samples,
            )
            return reader

        pre = list(build(False).variant_chunks(fields=["variant_position"]))
        post = list(build(True).variant_chunks(fields=["variant_position"]))
        nt.assert_array_equal(
            np.concatenate([c["variant_position"] for c in pre]),
            np.concatenate([c["variant_position"] for c in post]),
        )

    def test_filter_sees_full_samples_output_is_pruned(self, fx_sample_vcz):
        # Locks in that variants can be included because non-selected
        # samples matched the filter, while the returned call_* arrays
        # are sample-pruned — exercising the single-reader path where
        # both views coexist on one field.
        reader = make_reader(
            fx_sample_vcz.group,
            regions="20:1230236-",
            samples=["NA00002", "NA00003"],
            include="FMT/DP>3",
            filter_on_subset_samples=False,
        )
        chunks = list(reader.variant_chunks(fields=["variant_position", "call_DP"]))
        chunk = chunks[0]
        # Sample-pruned shape: 2 samples (NA00002, NA00003).
        assert chunk["call_DP"].shape[1] == 2
        # Position 1234567 survives only because NA00001 (not in the
        # selection) matched the filter — same lock-in as test_variant_chunks
        # but via the unified reader.
        nt.assert_array_equal(chunk["variant_position"], [1230237, 1234567])


class TestVczReaderMissingSamplesMultiChunk:
    """VczReader with a many-chunk sample axis and a large fraction of
    masked (``sample_id == ""``) samples.

    The store has 50 samples across 10 sample chunks with 30 masked
    indices arranged to exercise a fully-masked chunk, a fully-real
    chunk, mixed chunks, and masked indices on chunk boundaries.
    ``call_DP[v, s] = v*100 + s`` gives every cell a unique value so
    assertions are arithmetic; ``call_GQ`` is crafted per-row so
    sample-scope filter tests have predictable match sets.

    Bcftools has no notion of a missing sample — masked slots are a VCZ
    extension. Correct behaviour is therefore that masked data never
    influences filter evaluation, regardless of ``filter_on_subset_samples``.
    Tests that exercise that invariant today fail and are marked xfail;
    their xfail markers come off when the production path is fixed.
    """

    NUM_SAMPLES = 50
    NUM_VARIANTS = 15
    SAMPLES_CHUNK = 5
    VARIANTS_CHUNK = 5

    MASKED = np.array(
        [
            6,
            7,
            8,  # chunk 1: keep {5, 9}
            11,
            12,
            13,
            14,  # chunk 2: keep {10}
            15,
            16,
            17,
            18,
            19,  # chunk 3: fully masked
            21,
            22,
            23,  # chunk 4: keep {20, 24}
            26,
            27,
            28,
            29,  # chunk 5: keep {25}
            34,  # chunk 6: boundary (last), keep 30-33
            35,
            36,  # chunk 7: boundary (first), keep 37-39
            41,
            42,
            43,
            44,  # chunk 8: keep {40}
            45,
            46,
            47,
            48,  # chunk 9: keep {49}
        ]
    )
    REAL = np.setdiff1d(np.arange(NUM_SAMPLES), MASKED)

    @classmethod
    def _vcz(cls):
        sample_ids = np.array([f"s{i}" for i in range(cls.NUM_SAMPLES)], dtype="<U16")
        sample_ids[cls.MASKED] = ""

        v_idx = np.arange(cls.NUM_VARIANTS, dtype=np.int32)[:, None]
        s_idx = np.arange(cls.NUM_SAMPLES, dtype=np.int32)[None, :]
        call_dp = v_idx * 100 + s_idx

        call_gq = np.zeros((cls.NUM_VARIANTS, cls.NUM_SAMPLES), dtype=np.int32)
        # Row 10 — only masked samples cross "FMT/GQ > 50".
        call_gq[10, cls.MASKED] = 100
        # Row 11 — every real sample matches.
        call_gq[11, cls.REAL] = 100
        # Row 12 — only real sample s25 matches (outside the test subset).
        call_gq[12, 25] = 100
        # Row 13 — only real sample s37 matches (inside the test subset).
        call_gq[13, 37] = 100

        return vcz_builder.make_vcz(
            variant_contig=[0] * cls.NUM_VARIANTS,
            variant_position=list(range(1000, 1000 + cls.NUM_VARIANTS)),
            alleles=[("A", "T")] * cls.NUM_VARIANTS,
            num_samples=cls.NUM_SAMPLES,
            sample_id=sample_ids,
            samples_chunk_size=cls.SAMPLES_CHUNK,
            variants_chunk_size=cls.VARIANTS_CHUNK,
            call_fields={"DP": call_dp, "GQ": call_gq},
        )

    @staticmethod
    def _plan_indexes(plan):
        return [cr.index for cr in plan.chunk_reads]

    def test_default_drops_all_masked_samples(self):
        root = self._vcz()
        reader = VczReader(root)
        expected_ids = [f"s{i}" for i in self.REAL.tolist()]
        nt.assert_array_equal(reader.sample_ids, expected_ids)
        chunks = list(reader.variant_chunks(fields=["call_DP"]))
        dp = np.concatenate([c["call_DP"] for c in chunks], axis=0)
        assert dp.shape == (self.NUM_VARIANTS, self.REAL.size)
        v_idx = np.arange(self.NUM_VARIANTS)[:, None]
        expected_dp = v_idx * 100 + self.REAL[None, :]
        nt.assert_array_equal(dp, expected_dp)
        # Fully-masked chunk 3 is absent from the plan; every other chunk
        # appears exactly once.
        assert self._plan_indexes(reader.sample_chunk_plan) == [
            0,
            1,
            2,
            4,
            5,
            6,
            7,
            8,
            9,
        ]

    def test_subset_spanning_many_chunks_with_missing(self):
        reader = VczReader(self._vcz())
        subset = [0, 10, 20, 37, 49]
        reader.set_samples(subset)
        plan = reader.sample_chunk_plan
        assert self._plan_indexes(plan) == [0, 2, 4, 7, 9]
        assert plan.permutation is None
        expected_local = [[0], [0], [0], [2], [4]]
        for cr, local in zip(plan.chunk_reads, expected_local):
            nt.assert_array_equal(cr.selection, local)
        chunks = list(reader.variant_chunks(fields=["call_DP"]))
        dp = np.concatenate([c["call_DP"] for c in chunks], axis=0)
        v_idx = np.arange(self.NUM_VARIANTS)[:, None]
        expected_dp = v_idx * 100 + np.array(subset)[None, :]
        nt.assert_array_equal(dp, expected_dp)

    def test_subset_with_user_order_permutation(self):
        reader = VczReader(self._vcz())
        subset = [49, 37, 20, 10, 0]
        reader.set_samples(subset)
        plan = reader.sample_chunk_plan
        assert self._plan_indexes(plan) == [0, 2, 4, 7, 9]
        nt.assert_array_equal(plan.permutation, [4, 3, 2, 1, 0])
        chunks = list(reader.variant_chunks(fields=["call_DP"]))
        dp = np.concatenate([c["call_DP"] for c in chunks], axis=0)
        v_idx = np.arange(self.NUM_VARIANTS)[:, None]
        expected_dp = v_idx * 100 + np.array(subset)[None, :]
        nt.assert_array_equal(dp, expected_dp)

    def test_subset_inside_partial_chunk(self):
        reader = VczReader(self._vcz())
        subset = [5, 9]
        reader.set_samples(subset)
        plan = reader.sample_chunk_plan
        assert self._plan_indexes(plan) == [1]
        nt.assert_array_equal(plan.chunk_reads[0].selection, [0, 4])
        assert plan.permutation is None
        chunks = list(reader.variant_chunks(fields=["call_DP"]))
        dp = np.concatenate([c["call_DP"] for c in chunks], axis=0)
        v_idx = np.arange(self.NUM_VARIANTS)[:, None]
        expected_dp = v_idx * 100 + np.array(subset)[None, :]
        nt.assert_array_equal(dp, expected_dp)

    def test_variant_scope_filter_with_subset_and_missing(self):
        root = self._vcz()
        subset_names = ["s0", "s5", "s20", "s40"]
        subset_indexes = [0, 5, 20, 40]
        reader = make_reader(
            root,
            samples=subset_names,
            include="POS > 1007",
        )
        chunks = list(reader.variant_chunks(fields=["variant_position", "call_DP"]))
        positions = np.concatenate([c["variant_position"] for c in chunks])
        nt.assert_array_equal(positions, list(range(1008, 1015)))
        dp = np.concatenate([c["call_DP"] for c in chunks], axis=0)
        assert dp.shape == (7, 4)
        v_range = np.arange(8, 15)[:, None]
        expected_dp = v_range * 100 + np.array(subset_indexes)[None, :]
        nt.assert_array_equal(dp, expected_dp)

    def test_sample_scope_filter_pre_subset_ignores_masked(self):
        root = self._vcz()
        reader = make_reader(
            root,
            samples=["s0", "s10", "s20", "s37", "s49"],
            include="FMT/GQ > 50",
            filter_on_subset_samples=False,
        )
        chunks = list(reader.variant_chunks(fields=["variant_position"]))
        positions = np.concatenate([c["variant_position"] for c in chunks])
        assert 1010 not in positions.tolist()

    def test_sample_scope_filter_pre_subset_keeps_non_subset_real_match(self):
        root = self._vcz()
        subset_indexes = [0, 10, 20, 37, 49]
        reader = make_reader(
            root,
            samples=[f"s{i}" for i in subset_indexes],
            include="FMT/GQ > 50",
            filter_on_subset_samples=False,
        )
        chunks = list(reader.variant_chunks(fields=["variant_position", "call_DP"]))
        positions = np.concatenate([c["variant_position"] for c in chunks])
        # Row 12's only match is real sample s25 (not in the subset); the
        # filter's variant-inclusion decision sees every real sample, so
        # the variant survives. call_* output is subset-pruned.
        assert 1012 in positions.tolist()
        idx = int(np.flatnonzero(positions == 1012)[0])
        dp = np.concatenate([c["call_DP"] for c in chunks], axis=0)
        expected_row = 12 * 100 + np.array(subset_indexes)
        nt.assert_array_equal(dp[idx], expected_row)
        sample_filter_mask = np.concatenate(
            [c["sample_filter_mask"] for c in chunks], axis=0
        )
        nt.assert_array_equal(sample_filter_mask[idx], [False] * 5)

    def test_sample_scope_filter_post_subset_sees_only_subset(self):
        root = self._vcz()
        subset_indexes = [0, 10, 20, 37, 49]
        reader = make_reader(
            root,
            samples=[f"s{i}" for i in subset_indexes],
            include="FMT/GQ > 50",
            filter_on_subset_samples=True,
        )
        chunks = list(reader.variant_chunks(fields=["variant_position", "call_DP"]))
        positions = np.concatenate([c["variant_position"] for c in chunks])
        # Row 12 (only s25 matches — not in subset) is dropped; row 13
        # (s37 matches — in subset) is kept; row 11 (all real match) is
        # kept because subset samples are real. Row 10 (masked-only) is
        # dropped in this mode already.
        assert 1010 not in positions.tolist()
        assert 1011 in positions.tolist()
        assert 1012 not in positions.tolist()
        assert 1013 in positions.tolist()
        idx_13 = int(np.flatnonzero(positions == 1013)[0])
        dp = np.concatenate([c["call_DP"] for c in chunks], axis=0)
        nt.assert_array_equal(dp[idx_13], 13 * 100 + np.array(subset_indexes))
        sample_filter_mask = np.concatenate(
            [c["sample_filter_mask"] for c in chunks], axis=0
        )
        nt.assert_array_equal(
            sample_filter_mask[idx_13], [False, False, False, True, False]
        )

    def test_default_masking_sample_scope_filter_ignores_masked(self):
        root = self._vcz()
        variant_filter = BcftoolsFilter(
            field_names=frozenset(root.keys()), include="FMT/GQ > 50"
        )
        reader = VczReader(root)
        reader.set_variant_filter(variant_filter)
        chunks = list(reader.variant_chunks(fields=["variant_position"]))
        positions = np.concatenate([c["variant_position"] for c in chunks])
        # Correct behaviour: row 10 (masked-only match) is dropped; row
        # 11 (all-real match) is kept.
        assert 1010 not in positions.tolist()
        assert 1011 in positions.tolist()

    def test_both_modes_agree_when_filter_matches_only_subset(self):
        root = self._vcz()
        subset_indexes = [0, 10, 20, 37, 49]
        subset_names = [f"s{i}" for i in subset_indexes]

        reader_pre = make_reader(
            root,
            samples=subset_names,
            include="FMT/DP>1400",
            filter_on_subset_samples=False,
        )
        reader_post = make_reader(
            root,
            samples=subset_names,
            include="FMT/DP>1400",
            filter_on_subset_samples=True,
        )
        pre = list(reader_pre.variant_chunks(fields=["variant_position", "call_DP"]))
        post = list(reader_post.variant_chunks(fields=["variant_position", "call_DP"]))
        assert len(pre) == len(post)
        for p, q in zip(pre, post):
            nt.assert_array_equal(p["variant_position"], q["variant_position"])
            nt.assert_array_equal(p["call_DP"], q["call_DP"])
            nt.assert_array_equal(p["sample_filter_mask"], q["sample_filter_mask"])
        positions = np.concatenate([c["variant_position"] for c in pre])
        nt.assert_array_equal(positions, [1014])
        dp = np.concatenate([c["call_DP"] for c in pre], axis=0)
        nt.assert_array_equal(dp, [[1400, 1410, 1420, 1437, 1449]])
        sample_filter_mask = np.concatenate(
            [c["sample_filter_mask"] for c in pre], axis=0
        )
        nt.assert_array_equal(sample_filter_mask, [[False, True, True, True, True]])
