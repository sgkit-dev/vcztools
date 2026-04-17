import numpy as np
import numpy.testing as nt
import pandas as pd
import pytest

from tests import vcz_builder
from vcztools.retrieval import VariantChunkReader, VczReader, variant_chunk_iter


def test_variant_chunks(fx_sample_vcz):
    reader = VczReader(
        fx_sample_vcz.group,
        regions="20:1230236-",
        samples="NA00002,NA00003",
    )
    chunk_data = next(
        reader.variant_chunks(
            fields=["variant_contig", "variant_position", "call_DP", "call_GQ"],
            include="FMT/DP>3",
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


def test_variant_chunk_iter_empty_fields(fx_sample_vcz):
    with pytest.raises(StopIteration):
        print(next(variant_chunk_iter(fx_sample_vcz.group, fields=[])))


@pytest.mark.parametrize(
    ("regions", "samples"),
    [("20:1230236-", "NA00002,NA00003"), (["20:1230236-"], ["NA00002", "NA00003"])],
)
def test_variant_iter(fx_sample_vcz, regions, samples):
    reader = VczReader(fx_sample_vcz.group, regions=regions, samples=samples)
    it = reader.variants(
        fields=["variant_contig", "variant_position", "call_DP", "call_GQ"],
        include="FMT/DP>3",
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

    def test_include_filter_eq_pass(self):
        root = _make_filter_vcz()
        results = list(
            variant_chunk_iter(
                root,
                fields=["variant_position"],
                include='FILTER="PASS"',
            )
        )
        positions = np.concatenate([chunk["variant_position"] for chunk in results])
        # Even-indexed variants (0, 2, 4, 6, 8) are PASS → positions 100, 102, ...
        nt.assert_array_equal(positions, [100, 102, 104, 106, 108])

    def test_include_filter_ne_pass(self):
        root = _make_filter_vcz()
        results = list(
            variant_chunk_iter(
                root,
                fields=["variant_position"],
                include='FILTER!="PASS"',
            )
        )
        positions = np.concatenate([chunk["variant_position"] for chunk in results])
        # Odd-indexed variants are q10 → positions 101, 103, ...
        nt.assert_array_equal(positions, [101, 103, 105, 107])

    def test_exclude_filter_pass(self):
        root = _make_filter_vcz()
        results = list(
            variant_chunk_iter(
                root,
                fields=["variant_position"],
                exclude='FILTER="PASS"',
            )
        )
        positions = np.concatenate([chunk["variant_position"] for chunk in results])
        nt.assert_array_equal(positions, [101, 103, 105, 107])

    def test_filter_subset_match(self):
        root = _make_filter_vcz()
        results = list(
            variant_chunk_iter(
                root,
                fields=["variant_position"],
                include='FILTER~"q10"',
            )
        )
        positions = np.concatenate([chunk["variant_position"] for chunk in results])
        nt.assert_array_equal(positions, [101, 103, 105, 107])

    def test_filter_with_regions(self):
        root = _make_filter_vcz()
        reader = VczReader(root, regions="chr1:104-107")
        results = list(
            reader.variant_chunks(
                fields=["variant_position"],
                include='FILTER="PASS"',
            )
        )
        positions = np.concatenate([chunk["variant_position"] for chunk in results])
        # Positions 104-107 intersected with PASS (even) → 104, 106
        nt.assert_array_equal(positions, [104, 106])

    def test_chunk_reader_static_fields_across_chunks(self):
        """VariantChunkReader returns filter_id identically for every chunk."""
        root = _make_filter_vcz()
        reader = VariantChunkReader(root, fields=["variant_filter", "filter_id"])
        assert len(reader) == 3
        for chunk_idx in range(len(reader)):
            chunk_data = reader[chunk_idx]
            assert "filter_id" in chunk_data
            nt.assert_array_equal(chunk_data["filter_id"], ["PASS", "q10"])


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
