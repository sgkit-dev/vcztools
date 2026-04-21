import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from tests import vcz_builder
from vcztools import regions

ALL_CONTIGS = ["chr1", "chr2", "chr3"]


def _regions_df(rows):
    """Build the canonical regions DataFrame for the test fixtures.

    ``rows`` is a list of ``(contig, start, end)`` tuples; ``None`` is
    mapped to ``pd.NA`` in the nullable Int64 start/end columns.
    """
    contigs = [r[0] for r in rows]
    starts = [pd.NA if r[1] is None else r[1] for r in rows]
    ends = [pd.NA if r[2] is None else r[2] for r in rows]
    return pd.DataFrame(
        {
            "contig": contigs,
            "start": pd.array(starts, dtype="Int64"),
            "end": pd.array(ends, dtype="Int64"),
        }
    )


class TestParseRegionString:
    @pytest.mark.parametrize(
        ("region_str", "expected"),
        [
            ("chr1", ("chr1", None, None)),
            ("chr1:12", ("chr1", 12, 12)),
            ("chr1:12-", ("chr1", 12, None)),
            ("chr1:12-103", ("chr1", 12, 103)),
            # Numeric contig names
            ("22", ("22", None, None)),
            ("22:100-200", ("22", 100, 200)),
            # Contig names containing colons
            ("chr1:1:100-200", ("chr1:1", 100, 200)),
        ],
    )
    def test_values(self, region_str, expected):
        assert regions.parse_region_string(region_str) == expected


class TestReadRegionsFile:
    def test_basic(self):
        result = regions.read_regions_file("tests/data/txt/regions-3col.tsv")
        expected = _regions_df(
            [
                ("20", 1230237, 1235237),
                ("X", 10, 10),
            ]
        )
        assert_frame_equal(result, expected)

    def test_extra_columns_ignored(self, tmp_path):
        f = tmp_path / "regions.tsv"
        f.write_text("chr1\t100\t200\textra\tmore\n")
        result = regions.read_regions_file(str(f))
        assert_frame_equal(result, _regions_df([("chr1", 100, 200)]))

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.tsv"
        f.write_text("")
        with pytest.raises(ValueError, match="regions file is empty"):
            regions.read_regions_file(str(f))

    def test_one_column(self, tmp_path):
        f = tmp_path / "onecol.tsv"
        f.write_text("chr1\n")
        with pytest.raises(ValueError, match="expected at least 3.*got 1"):
            regions.read_regions_file(str(f))

    def test_two_columns(self, tmp_path):
        f = tmp_path / "twocol.tsv"
        f.write_text("chr1\t100\n")
        with pytest.raises(ValueError, match="expected at least 3.*got 2"):
            regions.read_regions_file(str(f))

    def test_non_numeric_start(self, tmp_path):
        f = tmp_path / "bad.tsv"
        f.write_text("chr1\tabc\t200\n")
        with pytest.raises(ValueError, match="non-numeric start position 'abc'"):
            regions.read_regions_file(str(f))

    def test_non_numeric_end(self, tmp_path):
        f = tmp_path / "bad.tsv"
        f.write_text("chr1\t100\txyz\n")
        with pytest.raises(ValueError, match="non-numeric end position 'xyz'"):
            regions.read_regions_file(str(f))

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            regions.read_regions_file("/nonexistent/path.tsv")

    def test_valid_row_followed_by_bad_row(self, tmp_path):
        f = tmp_path / "mixed.tsv"
        f.write_text("chr1\t100\t200\nchr1\tabc\t300\n")
        with pytest.raises(ValueError, match="non-numeric start position 'abc'"):
            regions.read_regions_file(str(f))


class TestRegionStringsToDataframe:
    def test_single_element_list(self):
        result = regions.region_strings_to_dataframe(["chr1:100-200"])
        assert_frame_equal(result, _regions_df([("chr1", 100, 200)]))

    def test_list_of_strings(self):
        result = regions.region_strings_to_dataframe(["chr1:100-200", "chr2:300-400"])
        assert_frame_equal(
            result, _regions_df([("chr1", 100, 200), ("chr2", 300, 400)])
        )

    def test_open_ended_end(self):
        result = regions.region_strings_to_dataframe(["chr1:100-"])
        assert_frame_equal(result, _regions_df([("chr1", 100, None)]))

    def test_whole_contig(self):
        result = regions.region_strings_to_dataframe(["chr1"])
        assert_frame_equal(result, _regions_df([("chr1", None, None)]))

    def test_contig_with_colon(self):
        result = regions.region_strings_to_dataframe(["chr1:1:100-200"])
        assert_frame_equal(result, _regions_df([("chr1:1", 100, 200)]))

    def test_na_roundtrips_through_int64(self):
        df = regions.region_strings_to_dataframe(["chr1", "chr1:100-200"])
        assert df["start"].dtype == pd.Int64Dtype()
        assert df["end"].dtype == pd.Int64Dtype()
        assert pd.isna(df.loc[0, "start"])
        assert pd.isna(df.loc[0, "end"])
        assert df.loc[1, "start"] == 100
        assert df.loc[1, "end"] == 200


class TestDataframeToRanges:
    def test_single_region(self):
        df = _regions_df([("chr1", 100, 200)])
        result = regions.dataframe_to_ranges(df, ALL_CONTIGS)
        # 1-based start -> 0-based
        assert_array_equal(result.contigs, [0])
        assert_array_equal(result.starts, [99])
        assert_array_equal(result.ends, [200])

    def test_multiple_regions(self):
        df = _regions_df([("chr1", 10, 20), ("chr3", 30, 40)])
        result = regions.dataframe_to_ranges(df, ALL_CONTIGS)
        assert_array_equal(result.contigs, [0, 2])
        assert_array_equal(result.starts, [9, 29])
        assert_array_equal(result.ends, [20, 40])

    def test_start_na_defaults_to_zero(self):
        df = _regions_df([("chr1", None, 200)])
        result = regions.dataframe_to_ranges(df, ALL_CONTIGS)
        assert_array_equal(result.starts, [0])

    def test_end_na_defaults_to_max(self):
        df = _regions_df([("chr1", 100, None)])
        result = regions.dataframe_to_ranges(df, ALL_CONTIGS)
        assert result.ends[0] == np.iinfo(np.int64).max

    def test_both_ends_na(self):
        df = _regions_df([("chr1", None, None), ("chr2", 300, None)])
        result = regions.dataframe_to_ranges(df, ALL_CONTIGS)
        int64_max = np.iinfo(np.int64).max
        assert_array_equal(result.contigs, [0, 1])
        assert_array_equal(result.starts, [0, 299])
        assert_array_equal(result.ends, [int64_max, int64_max])

    def test_unknown_contig_raises(self):
        df = _regions_df([("chrZ", 1, 10)])
        with pytest.raises(ValueError, match="not in list"):
            regions.dataframe_to_ranges(df, ALL_CONTIGS)

    def test_complement_default_false(self):
        df = _regions_df([("chr1", 10, 20)])
        result = regions.dataframe_to_ranges(df, ALL_CONTIGS)
        assert result.complement is False

    def test_complement_true(self):
        df = _regions_df([("chr1", 10, 20)])
        result = regions.dataframe_to_ranges(df, ALL_CONTIGS, complement=True)
        assert result.complement is True

    def test_from_file(self):
        all_contigs = ["19", "20", "X"]
        df = regions.read_regions_file("tests/data/txt/regions-3col.tsv")
        result = regions.dataframe_to_ranges(df, all_contigs)
        assert_array_equal(result.contigs, [1, 2])
        assert_array_equal(result.starts, [1230236, 9])
        assert_array_equal(result.ends, [1235237, 10])

    def test_none_returns_none(self):
        assert regions.dataframe_to_ranges(None, ALL_CONTIGS) is None

    def test_none_with_complement_returns_none(self):
        assert regions.dataframe_to_ranges(None, ALL_CONTIGS, complement=True) is None


class TestGenomicRangesOverlaps:
    def test_basic_overlap(self):
        a = regions.GenomicRanges(contigs=[0, 0], starts=[10, 50], ends=[20, 60])
        b = regions.GenomicRanges(contigs=[0], starts=[15], ends=[55])
        overlap = a.overlaps(b)
        assert_array_equal(overlap, [0, 1])

    def test_no_overlap(self):
        a = regions.GenomicRanges(contigs=[0], starts=[10], ends=[20])
        b = regions.GenomicRanges(contigs=[0], starts=[30], ends=[40])
        overlap = a.overlaps(b)
        assert len(overlap) == 0

    def test_different_contigs_no_overlap(self):
        a = regions.GenomicRanges(contigs=[0], starts=[10], ends=[20])
        b = regions.GenomicRanges(contigs=[1], starts=[10], ends=[20])
        overlap = a.overlaps(b)
        assert len(overlap) == 0

    def test_complement_overlap(self):
        # With complement, overlaps returns indices that do NOT overlap
        a = regions.GenomicRanges(
            contigs=[0, 0, 0],
            starts=[10, 30, 50],
            ends=[20, 40, 60],
            complement=True,
        )
        b = regions.GenomicRanges(contigs=[0], starts=[30], ends=[40])
        overlap = a.overlaps(b)
        # Indices 0 and 2 do not overlap with b, so they are returned
        assert_array_equal(overlap, [0, 2])


class TestRegionsToSelection:
    def test_regions_only(self):
        variant_contig = np.array([0, 0, 0, 0])
        variant_position = np.array([10, 20, 30, 40])
        variant_length = np.array([1, 1, 1, 1])
        region = regions.GenomicRanges(contigs=[0], starts=[14], ends=[35])
        result = regions.regions_to_selection(
            regions=region,
            targets=None,
            variant_contig=variant_contig,
            variant_position=variant_position,
            variant_length=variant_length,
        )
        # Variants at pos 20 and 30 overlap with [15, 35)
        assert_array_equal(result, [1, 2])

    def test_targets_only(self):
        variant_contig = np.array([0, 0, 0, 0])
        variant_position = np.array([10, 20, 30, 40])
        variant_length = np.array([1, 1, 1, 1])
        target = regions.GenomicRanges(contigs=[0], starts=[14], ends=[35])
        result = regions.regions_to_selection(
            regions=None,
            targets=target,
            variant_contig=variant_contig,
            variant_position=variant_position,
            variant_length=variant_length,
        )
        assert_array_equal(result, [1, 2])

    def test_both_regions_and_targets(self):
        variant_contig = np.array([0, 0, 0, 0])
        variant_position = np.array([10, 20, 30, 40])
        variant_length = np.array([1, 1, 1, 1])
        region = regions.GenomicRanges(contigs=[0], starts=[9], ends=[35])
        target = regions.GenomicRanges(contigs=[0], starts=[19], ends=[45])
        result = regions.regions_to_selection(
            regions=region,
            targets=target,
            variant_contig=variant_contig,
            variant_position=variant_position,
            variant_length=variant_length,
        )
        # Intersection: only variants overlapping both
        assert_array_equal(result, [1, 2])

    def test_no_overlap_returns_empty(self):
        variant_contig = np.array([0, 0])
        variant_position = np.array([10, 20])
        variant_length = np.array([1, 1])
        region = regions.GenomicRanges(contigs=[0], starts=[50], ends=[60])
        result = regions.regions_to_selection(
            regions=region,
            targets=None,
            variant_contig=variant_contig,
            variant_position=variant_position,
            variant_length=variant_length,
        )
        assert len(result) == 0

    def test_complement_targets(self):
        variant_contig = np.array([0, 0, 0])
        variant_position = np.array([10, 20, 30])
        variant_length = np.array([1, 1, 1])
        target = regions.GenomicRanges(
            contigs=[0],
            starts=[14],
            ends=[25],
            complement=True,
        )
        result = regions.regions_to_selection(
            regions=None,
            targets=target,
            variant_contig=variant_contig,
            variant_position=variant_position,
            variant_length=variant_length,
        )
        # Complement: exclude variant at pos 20, keep 10 and 30
        assert_array_equal(result, [0, 2])


class TestRegionsToChunkIndexes:
    def test_basic_overlap(self):
        # region_index columns: chunk_index, contig_id, start_pos, end_pos, max_end_pos
        region_index = np.array(
            [
                [0, 0, 10, 20, 25],
                [1, 0, 30, 40, 45],
                [2, 0, 50, 60, 65],
            ]
        )
        query = regions.GenomicRanges(contigs=[0], starts=[25], ends=[55])
        result = regions.regions_to_chunk_indexes(query, region_index)
        assert_array_equal(result, [1, 2])

    def test_no_overlap_returns_empty(self):
        region_index = np.array(
            [
                [0, 0, 10, 20, 25],
                [1, 0, 30, 40, 45],
            ]
        )
        query = regions.GenomicRanges(contigs=[0], starts=[100], ends=[200])
        result = regions.regions_to_chunk_indexes(query, region_index)
        assert len(result) == 0


class TestChunkRead:
    def test_defaults(self):
        cr = regions.ChunkRead(index=3)
        assert cr.index == 3
        assert cr.selection is None

    def test_with_selection(self):
        sel = np.array([0, 2], dtype=np.int64)
        cr = regions.ChunkRead(index=1, selection=sel)
        assert cr.index == 1
        assert_array_equal(cr.selection, [0, 2])


class TestChunkPlanFromIndexes:
    def test_buckets_into_chunks(self):
        plan = regions.chunk_plan_from_indexes(
            np.array([0, 1, 3, 5, 7]), variants_chunk_size=3
        )
        # global indexes 0,1 → chunk 0 local [0,1]
        # global index 3    → chunk 1 local [0]
        # global indexes 5,7 → chunk 1 local [2], chunk 2 local [1]
        # (sorted by chunk)
        assert [cr.index for cr in plan] == [0, 1, 2]
        assert_array_equal(plan[0].selection, [0, 1])
        assert_array_equal(plan[1].selection, [0, 2])
        assert_array_equal(plan[2].selection, [1])

    def test_empty(self):
        plan = regions.chunk_plan_from_indexes(
            np.array([], dtype=np.int64), variants_chunk_size=3
        )
        assert plan == []


class TestBuildChunkPlan:
    @staticmethod
    def _vcz():
        return vcz_builder.make_vcz(
            variant_contig=[0] * 10,
            variant_position=list(range(1, 11)),
            alleles=[("A", "T")] * 10,
            contigs=("chr1",),
            variants_chunk_size=3,
        )

    def test_no_filter_returns_full_plan(self):
        """No regions/targets: one ChunkRead per chunk with
        ``selection is None`` (the "read full chunk" sentinel)."""
        plan = regions.build_chunk_plan(self._vcz())
        # 10 variants, chunk size 3 → 4 chunks
        assert [cr.index for cr in plan] == [0, 1, 2, 3]
        assert all(cr.selection is None for cr in plan)

    def test_regions_filter_selects_chunks(self):
        plan = regions.build_chunk_plan(self._vcz(), regions="chr1:4-5")
        # positions 4,5 fall in chunk 1 (global indexes 3,4 → local 0,1)
        assert [cr.index for cr in plan] == [1]
        assert_array_equal(plan[0].selection, [0, 1])

    def test_targets_complement_flag_real_filter(self):
        """``targets_complement=True`` with a real target string runs
        through the candidate-chunk scan and produces explicit
        selection arrays."""
        plan = regions.build_chunk_plan(
            self._vcz(), targets="chr1:4-5", targets_complement=True
        )
        # Positions 4 and 5 excluded → keep everything else.
        # Concatenating all plan selections should yield the local
        # indexes of positions 1,2,3,6,7,8,9,10 (globals 0,1,2,5,6,7,8,9).
        assert all(cr.selection is not None for cr in plan)
