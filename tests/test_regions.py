import pytest
from numpy.testing import assert_array_equal

from vcztools.regions import (
    Region,
    parse_region_string,
    parse_regions,
    parse_regions_file,
    parse_targets,
)


@pytest.mark.parametrize(
    ("region_str", "expected"),
    [
        ("chr1", Region("chr1")),
        ("chr1:12", Region("chr1", 12, 12)),
        ("chr1:12-", Region("chr1", 12, None)),
        ("chr1:12-103", Region("chr1", 12, 103)),
    ],
)
def test_parse_region_string(region_str: str, expected: Region):
    assert parse_region_string(region_str) == expected


class TestParseRegionsFile:
    def test_basic(self):
        regions = parse_regions_file("tests/data/txt/regions-3col.tsv")
        assert regions == [
            Region("20", 1230237, 1235237),
            Region("X", 10, 10),
        ]


class TestParseRegions:
    def test_from_file(self):
        all_contigs = ["19", "20", "X"]
        regions = parse_regions_file("tests/data/txt/regions-3col.tsv")
        genomic_ranges = parse_regions(regions, all_contigs)
        assert_array_equal(genomic_ranges.contigs, [1, 2])
        assert_array_equal(genomic_ranges.starts, [1230236, 9])
        assert_array_equal(genomic_ranges.ends, [1235237, 10])


class TestParseTargets:
    def test_from_file_with_complement(self):
        all_contigs = ["19", "20", "X"]
        regions = parse_regions_file("tests/data/txt/regions-3col.tsv")
        genomic_ranges = parse_targets(regions, all_contigs, complement=True)
        assert_array_equal(genomic_ranges.contigs, [1, 2])
        assert_array_equal(genomic_ranges.starts, [1230236, 9])
        assert_array_equal(genomic_ranges.ends, [1235237, 10])
        assert genomic_ranges.complement
