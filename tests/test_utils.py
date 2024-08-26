import pytest
from numpy.testing import assert_array_equal

from vcztools.utils import (
    search,
    vcf_name_to_vcz_name,
)


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


@pytest.mark.parametrize(
    ("vczs", "vcf", "expected_vcz"),
    [
        (set(), "GT", "call_genotype"),
        (set(), "FMT/GT", "call_genotype"),
        ({"call_DP"}, "DP", "call_DP"),
        ({"variant_DP"}, "DP", "variant_DP"),
        ({"call_DP", "variant_DP"}, "DP", "call_DP"),
        ({"call_DP", "variant_DP"}, "INFO/DP", "variant_DP"),
        ({"call_DP", "variant_DP"}, "FORMAT/DP", "call_DP"),
        ({"variant_DP"}, "FORMAT/DP", "call_DP"),
        (set(), "CHROM", "variant_contig"),
        (set(), "POS", "variant_position"),
        (set(), "ID", "variant_id"),
        (set(), "REF", "variant_allele"),
        (set(), "ALT", "variant_allele"),
        (set(), "QUAL", "variant_quality"),
        (set(), "FILTER", "variant_filter"),
    ],
)
def test_vcf_to_vcz(vczs, vcf, expected_vcz):
    assert vcf_name_to_vcz_name(vczs, vcf) == expected_vcz
