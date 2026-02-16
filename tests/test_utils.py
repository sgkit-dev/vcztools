import numpy as np
import pytest
from numpy.testing import assert_array_equal

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
    missing,
    search,
    vcf_name_to_vcz_names,
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
