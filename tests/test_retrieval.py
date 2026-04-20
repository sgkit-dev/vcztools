import numpy as np
import numpy.testing as nt
import pandas as pd
import pytest

from tests import vcz_builder
from tests.utils import make_reader
from vcztools.bcftools_filter import BcftoolsFilter
from vcztools.retrieval import VariantChunkReader, VczReader


def test_variant_chunks(fx_sample_vcz):
    reader = make_reader(
        fx_sample_vcz.group,
        regions="20:1230236-",
        samples=["NA00002", "NA00003"],
        include="FMT/DP>3",
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
    nt.assert_array_equal(chunk_data["call_mask"], [[True, False], [False, False]])


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
    )
    it = reader.variants(
        fields=["variant_contig", "variant_position", "call_DP", "call_GQ"],
    )

    variant1 = next(it)
    assert variant1["variant_contig"] == 1
    assert variant1["variant_position"] == 1230237
    nt.assert_array_equal(variant1["call_DP"], [4, 2])
    nt.assert_array_equal(variant1["call_GQ"], [48, 61])
    nt.assert_array_equal(variant1["call_mask"], [True, False])

    variant2 = next(it)
    assert variant2["variant_contig"] == 1
    assert variant2["variant_position"] == 1234567
    nt.assert_array_equal(variant2["call_DP"], [2, 3])
    nt.assert_array_equal(variant2["call_GQ"], [17, 40])
    nt.assert_array_equal(variant2["call_mask"], [False, False])

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
        """VariantChunkReader returns filter_id identically for every chunk."""
        root = _make_filter_vcz()
        reader = VariantChunkReader(root)
        num_chunks = int(root["variant_filter"].cdata_shape[0])
        assert num_chunks == 3
        for chunk_idx in range(num_chunks):
            reader.set_chunk(chunk_idx)
            nt.assert_array_equal(reader.get("filter_id"), ["PASS", "q10"])


class TestVczReaderRegions:
    """Cover the three accepted region/target input shapes plus error paths."""

    @staticmethod
    def _vcz():
        # 10 variants on chr1 at positions 1..10, alternating AC values.
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
        reader = VczReader(self._vcz(), regions="chr1:3-5")
        nt.assert_array_equal(self._positions(reader), [3, 4, 5])

    def test_regions_list_of_strings(self):
        reader = VczReader(self._vcz(), regions=["chr1:3-5", "chr1:8-9"])
        nt.assert_array_equal(self._positions(reader), [3, 4, 5, 8, 9])

    def test_regions_dataframe(self):
        df = pd.DataFrame(
            {
                "contig": ["chr1"],
                "start": pd.array([3], dtype="Int64"),
                "end": pd.array([5], dtype="Int64"),
            }
        )
        reader = VczReader(self._vcz(), regions=df)
        nt.assert_array_equal(self._positions(reader), [3, 4, 5])

    def test_regions_dataframe_with_na_end(self):
        df = pd.DataFrame(
            {
                "contig": ["chr1"],
                "start": pd.array([8], dtype="Int64"),
                "end": pd.array([pd.NA], dtype="Int64"),
            }
        )
        reader = VczReader(self._vcz(), regions=df)
        nt.assert_array_equal(self._positions(reader), [8, 9, 10])

    def test_targets_complement_flag(self):
        reader = VczReader(self._vcz(), targets="chr1:3-5", targets_complement=True)
        nt.assert_array_equal(self._positions(reader), [1, 2, 6, 7, 8, 9, 10])

    def test_regions_rejects_caret_prefix(self):
        with pytest.raises(ValueError, match="targets_complement=True"):
            VczReader(self._vcz(), regions="^chr1:1-3")

    def test_targets_rejects_caret_prefix(self):
        with pytest.raises(ValueError, match="targets_complement=True"):
            VczReader(self._vcz(), targets="^chr1:1-3")

    def test_regions_rejects_comma(self):
        with pytest.raises(ValueError, match=r"regions string .* contains ','"):
            VczReader(self._vcz(), regions="chr1:1-3,chr1:5-7")

    def test_targets_rejects_comma(self):
        with pytest.raises(ValueError, match=r"targets string .* contains ','"):
            VczReader(self._vcz(), targets="chr1:1-3,chr1:5-7")

    def test_regions_invalid_type(self):
        with pytest.raises(TypeError, match="regions must be"):
            VczReader(self._vcz(), regions=42)

    def test_targets_invalid_type(self):
        with pytest.raises(TypeError, match="targets must be"):
            VczReader(self._vcz(), targets=42)

    def test_regions_dataframe_missing_columns(self):
        df = pd.DataFrame({"contig": ["chr1"], "start": pd.array([1], dtype="Int64")})
        with pytest.raises(ValueError, match="missing required columns.*end"):
            VczReader(self._vcz(), regions=df)


class TestVczReaderSamples:
    """Cover VczReader sample input: None, list, complement, error cases."""

    def test_samples_none_selects_all(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        nt.assert_array_equal(reader.sample_ids, ["NA00001", "NA00002", "NA00003"])

    def test_samples_list(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group, samples=["NA00001", "NA00003"])
        nt.assert_array_equal(reader.sample_ids, ["NA00001", "NA00003"])
        nt.assert_array_equal(reader.samples_selection, [0, 2])

    def test_samples_complement_flag(self, fx_sample_vcz):
        reader = VczReader(
            fx_sample_vcz.group,
            samples=["NA00002"],
            samples_complement=True,
        )
        nt.assert_array_equal(reader.sample_ids, ["NA00001", "NA00003"])
        nt.assert_array_equal(reader.samples_selection, [0, 2])

    def test_samples_rejects_string_input(self, fx_sample_vcz):
        with pytest.raises(TypeError, match="samples must be list"):
            VczReader(fx_sample_vcz.group, samples="NA00001")

    def test_samples_rejects_caret_prefix(self, fx_sample_vcz):
        with pytest.raises(ValueError, match="samples_complement=True"):
            VczReader(fx_sample_vcz.group, samples=["^NA00001"])

    def test_samples_unknown_raises(self, fx_sample_vcz):
        with pytest.raises(ValueError, match="not in header: NO_SAMPLE"):
            VczReader(fx_sample_vcz.group, samples=["NO_SAMPLE"])

    def test_samples_unknown_ignore_missing(self, fx_sample_vcz):
        reader = VczReader(
            fx_sample_vcz.group, samples=["NO_SAMPLE"], ignore_missing_samples=True
        )
        nt.assert_array_equal(reader.sample_ids, [])


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

    def test_single_chunk_selection(self):
        # s2, s3 both live in sample chunk 1 (indexes 2, 3).
        reader = VczReader(self._vcz(), samples=["s2", "s3"])
        nt.assert_array_equal(reader.sample_chunk_plan.chunk_indexes, [1])
        nt.assert_array_equal(reader.sample_chunk_plan.local_selection, [0, 1])
        dp = self._call_dp(reader)
        nt.assert_array_equal(dp, [[2, 3], [12, 13], [22, 23]])

    def test_multi_chunk_selection(self):
        # s1 is in chunk 0; s4 is in chunk 2; chunk 1 is skipped.
        reader = VczReader(self._vcz(), samples=["s1", "s4"])
        nt.assert_array_equal(reader.sample_chunk_plan.chunk_indexes, [0, 2])
        dp = self._call_dp(reader)
        nt.assert_array_equal(dp, [[1, 4], [11, 14], [21, 24]])

    def test_preserves_user_order(self):
        # Same chunks as the multi-chunk test, but the user order is
        # reversed — the output must follow the input list.
        reader = VczReader(self._vcz(), samples=["s4", "s1"])
        nt.assert_array_equal(reader.sample_chunk_plan.chunk_indexes, [0, 2])
        dp = self._call_dp(reader)
        nt.assert_array_equal(dp, [[4, 1], [14, 11], [24, 21]])

    def test_partial_final_chunk(self):
        # 5 samples with chunk size 2 → chunks sized [2, 2, 1]. s4 sits
        # alone in the final chunk.
        reader = VczReader(
            self._vcz(num_samples=5, samples_chunk_size=2), samples=["s4"]
        )
        nt.assert_array_equal(reader.sample_chunk_plan.chunk_indexes, [2])
        nt.assert_array_equal(reader.sample_chunk_plan.local_selection, [0])
        dp = self._call_dp(reader)
        nt.assert_array_equal(dp, [[4], [14], [24]])

    def test_samples_none_plan_is_identity(self):
        # samples=None plans a read over every sample chunk with an
        # identity local selection — same shape and values as a full read.
        reader = VczReader(self._vcz())
        nt.assert_array_equal(reader.sample_chunk_plan.chunk_indexes, [0, 1, 2])
        nt.assert_array_equal(
            reader.sample_chunk_plan.local_selection, [0, 1, 2, 3, 4, 5]
        )
        dp = self._call_dp(reader)
        nt.assert_array_equal(dp.shape, (3, 6))
        nt.assert_array_equal(dp[0], [0, 1, 2, 3, 4, 5])

    def test_samples_none_missing_header_reduces(self):
        # samples=None with masked "" entries in sample_id: the plan
        # drops the masked indices, matching the old post-filter path.
        root = self._vcz()
        root["sample_id"][:2] = ""
        reader = VczReader(root)
        nt.assert_array_equal(reader.sample_ids, ["s2", "s3", "s4", "s5"])
        dp = self._call_dp(reader)
        nt.assert_array_equal(dp, [[2, 3, 4, 5], [12, 13, 14, 15], [22, 23, 24, 25]])

    def test_drop_genotypes_has_no_plan(self):
        reader = VczReader(self._vcz(), drop_genotypes=True)
        assert reader.sample_chunk_plan is None

    def test_force_samples_empty_has_no_plan(self):
        # --force-samples eliminates every requested sample → plan=None,
        # read full call arrays so AC/AN can still be recomputed.
        reader = VczReader(
            self._vcz(), samples=["UNKNOWN"], ignore_missing_samples=True
        )
        assert reader.sample_chunk_plan is None


class TestVariantChunkReaderCaching:
    """The reader must read each (v_chunk, field, s_chunk) block at most
    once per chunk visit, regardless of how many times it is queried."""

    @staticmethod
    def _vcz(num_samples=4, samples_chunk_size=2, variants_chunk_size=2):
        num_variants = 4
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

    def test_field_in_filter_and_query_reuses_cached_blocks(self):
        # Accessing a call_* field first as the filter sees it (full
        # samples) populates every sample chunk. A later query-side
        # access (pruned) must find each block already cached — identity
        # check proves we did not re-fetch.
        reader = VariantChunkReader(self._vcz())
        reader.set_chunk(0)
        reader.get_with_all_samples("call_DP")
        cached_after_full = dict(reader._call_blocks)
        # 2 sample chunks populated.
        assert len(cached_after_full) == 2
        reader.get("call_DP")
        for key, block in cached_after_full.items():
            assert reader._call_blocks[key] is block

    def test_pruned_then_full_reads_only_missing_blocks(self):
        # Pruned first reads only plan.chunk_indexes; a subsequent full
        # access fills the remaining s_chunks without re-reading the
        # already-cached ones.
        root = self._vcz(num_samples=6, samples_chunk_size=2)
        # Plan selects sample 1 (chunk 0) and sample 5 (chunk 2);
        # chunk 1 is NOT in the plan.
        reader = VczReader(root, samples=["s1", "s5"])
        inner = VariantChunkReader(root, sample_chunk_plan=reader.sample_chunk_plan)
        inner.set_chunk(0)
        inner.get("call_DP")
        # Only plan chunks (0 and 2) are cached so far.
        assert {key[2] for key in inner._call_blocks} == {0, 2}
        pruned_block_0 = inner._call_blocks[(0, "call_DP", 0)]
        inner.get_with_all_samples("call_DP")
        # All 3 sample chunks now cached; chunk 0's block is the same
        # object (proving it wasn't re-fetched).
        assert {key[2] for key in inner._call_blocks} == {0, 1, 2}
        assert inner._call_blocks[(0, "call_DP", 0)] is pruned_block_0

    def test_set_chunk_evicts_previous_chunk_blocks(self):
        reader = VariantChunkReader(self._vcz())
        reader.set_chunk(0)
        reader.get_with_all_samples("call_DP")
        assert len(reader._call_blocks) == 2
        reader.set_chunk(1)
        assert reader._call_blocks == {}
        assert reader._variant_blocks == {}

    def test_set_chunk_same_idx_is_idempotent(self):
        reader = VariantChunkReader(self._vcz())
        reader.set_chunk(0)
        reader.get_with_all_samples("call_DP")
        cached_len = len(reader._call_blocks)
        reader.set_chunk(0)
        assert len(reader._call_blocks) == cached_len

    def test_static_field_cached_across_chunks(self):
        # filter_id has no variants axis — it survives set_chunk.
        reader = VariantChunkReader(_make_filter_vcz())
        reader.set_chunk(0)
        filters_first = reader.get("filter_id")
        reader.set_chunk(1)
        filters_second = reader.get("filter_id")
        assert filters_first is filters_second

    def test_variant_field_cached_within_chunk(self):
        reader = VariantChunkReader(self._vcz())
        reader.set_chunk(0)
        first = reader.get("variant_position")
        second = reader.get("variant_position")
        # Same object — second access hit the cache.
        assert first is second


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

        reader = VczReader(fx_sample_vcz.group, variant_filter=PositionGt(1000000))
        chunks = list(reader.variant_chunks(fields=["variant_position"]))
        positions = np.concatenate([c["variant_position"] for c in chunks])
        nt.assert_array_equal(positions, [1110696, 1230237, 1234567, 1235237])

    def test_filter_on_subset_samples_sample_scope(self, fx_sample_vcz):
        # bcftools-query-style post-subset evaluation: filter sees only
        # the selected samples, so position 1234567 (where NA00001 but
        # not NA00002/NA00003 matched FMT/DP>3) is DROPPED.
        variant_filter = BcftoolsFilter(
            field_names=set(VczReader(fx_sample_vcz.group).root),
            include="FMT/DP>3",
        )
        reader = VczReader(
            fx_sample_vcz.group,
            regions="20:1230236-",
            samples=["NA00002", "NA00003"],
            variant_filter=variant_filter,
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
        field_names = set(VczReader(root).root)

        def build(filter_on_subset_samples):
            return VczReader(
                root,
                regions="20:1230236-",
                variant_filter=BcftoolsFilter(
                    field_names=field_names, include="FMT/DP>3"
                ),
                filter_on_subset_samples=filter_on_subset_samples,
            )

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
        field_names = set(VczReader(root).root)

        def build(filter_on_subset_samples):
            return VczReader(
                root,
                samples=["NA00001"],
                variant_filter=BcftoolsFilter(
                    field_names=field_names, include="POS > 1000000"
                ),
                filter_on_subset_samples=filter_on_subset_samples,
            )

        pre = list(build(False).variant_chunks(fields=["variant_position"]))
        post = list(build(True).variant_chunks(fields=["variant_position"]))
        nt.assert_array_equal(
            np.concatenate([c["variant_position"] for c in pre]),
            np.concatenate([c["variant_position"] for c in post]),
        )

    def test_drop_genotypes_rejects_sample_scope_filter(self, fx_sample_vcz):
        field_names = set(VczReader(fx_sample_vcz.group).root)
        variant_filter = BcftoolsFilter(field_names=field_names, include="FMT/DP>3")
        with pytest.raises(
            ValueError, match="sample-scope variant_filter is incompatible"
        ):
            VczReader(
                fx_sample_vcz.group,
                variant_filter=variant_filter,
                drop_genotypes=True,
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
        )
        chunks = list(reader.variant_chunks(fields=["variant_position", "call_DP"]))
        chunk = chunks[0]
        # Sample-pruned shape: 2 samples (NA00002, NA00003).
        assert chunk["call_DP"].shape[1] == 2
        # Position 1234567 survives only because NA00001 (not in the
        # selection) matched the filter — same lock-in as test_variant_chunks
        # but via the unified reader.
        nt.assert_array_equal(chunk["variant_position"], [1230237, 1234567])
