import numpy as np
import pytest
from numpy.testing import assert_array_equal

from vcztools import regions
from vcztools.regions import Region

ALL_CONTIGS = ["chr1", "chr2", "chr3"]


class TestParseRegionString:
    @pytest.mark.parametrize(
        ("region_str", "expected"),
        [
            ("chr1", Region("chr1")),
            ("chr1:12", Region("chr1", 12, 12)),
            ("chr1:12-", Region("chr1", 12, None)),
            ("chr1:12-103", Region("chr1", 12, 103)),
            # Numeric contig names
            ("22", Region("22")),
            ("22:100-200", Region("22", 100, 200)),
            # Contig names containing colons
            ("chr1:1:100-200", Region("chr1:1", 100, 200)),
        ],
    )
    def test_values(self, region_str, expected):
        assert regions.parse_region_string(region_str) == expected


class TestParseRegionsFile:
    def test_basic(self):
        result = regions.parse_regions_file("tests/data/txt/regions-3col.tsv")
        assert result == [
            Region("20", 1230237, 1235237),
            Region("X", 10, 10),
        ]

    def test_extra_columns_ignored(self, tmp_path):
        f = tmp_path / "regions.tsv"
        f.write_text("chr1\t100\t200\textra\tmore\n")
        result = regions.parse_regions_file(str(f))
        assert result == [Region("chr1", 100, 200)]

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.tsv"
        f.write_text("")
        with pytest.raises(ValueError, match="regions file is empty"):
            regions.parse_regions_file(str(f))

    def test_one_column(self, tmp_path):
        f = tmp_path / "onecol.tsv"
        f.write_text("chr1\n")
        with pytest.raises(ValueError, match="expected at least 3.*got 1"):
            regions.parse_regions_file(str(f))

    def test_two_columns(self, tmp_path):
        f = tmp_path / "twocol.tsv"
        f.write_text("chr1\t100\n")
        with pytest.raises(ValueError, match="expected at least 3.*got 2"):
            regions.parse_regions_file(str(f))

    def test_non_numeric_start(self, tmp_path):
        f = tmp_path / "bad.tsv"
        f.write_text("chr1\tabc\t200\n")
        with pytest.raises(ValueError, match="non-numeric start position 'abc'"):
            regions.parse_regions_file(str(f))

    def test_non_numeric_end(self, tmp_path):
        f = tmp_path / "bad.tsv"
        f.write_text("chr1\t100\txyz\n")
        with pytest.raises(ValueError, match="non-numeric end position 'xyz'"):
            regions.parse_regions_file(str(f))

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            regions.parse_regions_file("/nonexistent/path.tsv")

    def test_valid_row_followed_by_bad_row(self, tmp_path):
        f = tmp_path / "mixed.tsv"
        f.write_text("chr1\t100\t200\nchr1\tabc\t300\n")
        with pytest.raises(ValueError, match="non-numeric start position 'abc'"):
            regions.parse_regions_file(str(f))


class TestRegionsToRanges:
    def test_single_region(self):
        result = regions.regions_to_ranges([Region("chr1", 100, 200)], ALL_CONTIGS)
        # 1-based start → 0-based
        assert_array_equal(result.contigs, [0])
        assert_array_equal(result.starts, [99])
        assert_array_equal(result.ends, [200])

    def test_start_none_defaults_to_zero(self):
        result = regions.regions_to_ranges([Region("chr1", None, 200)], ALL_CONTIGS)
        assert_array_equal(result.starts, [0])

    def test_end_none_defaults_to_max(self):
        result = regions.regions_to_ranges([Region("chr1", 100, None)], ALL_CONTIGS)
        assert result.ends[0] == np.iinfo(np.int32).max

    def test_multiple_regions(self):
        result = regions.regions_to_ranges(
            [Region("chr1", 10, 20), Region("chr3", 30, 40)], ALL_CONTIGS
        )
        assert_array_equal(result.contigs, [0, 2])
        assert_array_equal(result.starts, [9, 29])
        assert_array_equal(result.ends, [20, 40])

    def test_unknown_contig_raises(self):
        with pytest.raises(ValueError, match="not in list"):
            regions.regions_to_ranges([Region("chrZ", 1, 10)], ALL_CONTIGS)

    def test_complement_flag(self):
        result = regions.regions_to_ranges(
            [Region("chr1", 10, 20)], ALL_CONTIGS, complement=True
        )
        assert result.complement is True


class TestParseRegions:
    def test_comma_separated_string(self):
        result = regions.parse_regions("chr1:100-200,chr2:300-400", ALL_CONTIGS)
        assert_array_equal(result.contigs, [0, 1])
        assert_array_equal(result.starts, [99, 299])
        assert_array_equal(result.ends, [200, 400])

    def test_single_region_string(self):
        result = regions.parse_regions("chr1", ALL_CONTIGS)
        assert_array_equal(result.contigs, [0])
        assert_array_equal(result.starts, [0])
        assert result.ends[0] == np.iinfo(np.int32).max

    def test_list_of_strings(self):
        result = regions.parse_regions(["chr1:100-200", "chr2:300-400"], ALL_CONTIGS)
        assert_array_equal(result.contigs, [0, 1])

    def test_list_of_regions(self):
        result = regions.parse_regions(
            [Region("chr1", 100, 200), Region("chr2", 300, 400)], ALL_CONTIGS
        )
        assert_array_equal(result.contigs, [0, 1])
        assert_array_equal(result.starts, [99, 299])
        assert_array_equal(result.ends, [200, 400])

    def test_from_file(self):
        all_contigs = ["19", "20", "X"]
        file_regions = regions.parse_regions_file("tests/data/txt/regions-3col.tsv")
        result = regions.parse_regions(file_regions, all_contigs)
        assert_array_equal(result.contigs, [1, 2])
        assert_array_equal(result.starts, [1230236, 9])
        assert_array_equal(result.ends, [1235237, 10])

    def test_genomic_ranges_passthrough(self):
        gr = regions.GenomicRanges(contigs=[0], starts=[99], ends=[200])
        result = regions.parse_regions(gr, ALL_CONTIGS)
        assert result is gr

    def test_none_returns_none(self):
        assert regions.parse_regions(None, ALL_CONTIGS) is None

    def test_complement_not_allowed(self):
        # Regions do not support complement; the ^ should be treated as
        # part of the contig name and fail to match.
        with pytest.raises(ValueError, match="not in list"):
            regions.parse_regions("^chr1:100-200", ALL_CONTIGS)


class TestParseTargets:
    def test_complement_prefix(self):
        result = regions.parse_targets("^chr1:100-200", ALL_CONTIGS)
        assert result.complement is True
        assert_array_equal(result.contigs, [0])
        assert_array_equal(result.starts, [99])

    def test_no_complement_prefix(self):
        result = regions.parse_targets("chr1:100-200", ALL_CONTIGS)
        assert result.complement is False

    def test_from_file_with_complement(self):
        all_contigs = ["19", "20", "X"]
        file_regions = regions.parse_regions_file("tests/data/txt/regions-3col.tsv")
        result = regions.parse_targets(file_regions, all_contigs, complement=True)
        assert_array_equal(result.contigs, [1, 2])
        assert_array_equal(result.starts, [1230236, 9])
        assert_array_equal(result.ends, [1235237, 10])
        assert result.complement is True

    def test_from_file_without_complement(self):
        all_contigs = ["19", "20", "X"]
        file_regions = regions.parse_regions_file("tests/data/txt/regions-3col.tsv")
        result = regions.parse_targets(file_regions, all_contigs, complement=False)
        assert result.complement is False

    def test_none_returns_none(self):
        assert regions.parse_targets(None, ALL_CONTIGS) is None


class TestGenomicRangesCoordValidation:
    def test_start_overflow_raises(self):
        with pytest.raises(ValueError, match="start coordinate out of range"):
            regions.GenomicRanges(
                contigs=[0],
                starts=[2**31],
                ends=[2**31 + 10],
            )

    def test_end_overflow_raises(self):
        with pytest.raises(ValueError, match="end coordinate out of range"):
            regions.GenomicRanges(
                contigs=[0],
                starts=[0],
                ends=[2**31],
            )


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
