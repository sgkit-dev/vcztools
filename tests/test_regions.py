import pytest
from numpy.testing import assert_array_equal

from vcztools.regions import parse_region_string, parse_regions, parse_targets


@pytest.mark.parametrize(
    ("targets", "expected"),
    [
        ("chr1", ("chr1", None, None)),
        ("chr1:12", ("chr1", 12, 12)),
        ("chr1:12-", ("chr1", 12, None)),
        ("chr1:12-103", ("chr1", 12, 103)),
    ],
)
def test_parse_region_string(
    targets: str, expected: tuple[str, int | None, int | None]
):
    assert parse_region_string(targets) == expected


def test_parse_regions_file():
    all_contigs = ["19", "20", "X"]
    genomic_ranges = parse_regions(
        regions=None,
        all_contigs=all_contigs,
        regions_file="tests/data/txt/regions-3col.tsv",
    )
    assert_array_equal(genomic_ranges.contigs, [1, 2])
    assert_array_equal(genomic_ranges.starts, [1230236, 9])
    assert_array_equal(genomic_ranges.ends, [1235237, 10])


def test_parse_targets_file():
    all_contigs = ["19", "20", "X"]
    genomic_ranges = parse_targets(
        targets=None,
        all_contigs=all_contigs,
        targets_file="^tests/data/txt/regions-3col.tsv",
    )
    assert_array_equal(genomic_ranges.contigs, [1, 2])
    assert_array_equal(genomic_ranges.starts, [1230236, 9])
    assert_array_equal(genomic_ranges.ends, [1235237, 10])
    assert genomic_ranges.complement
