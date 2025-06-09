import numpy as np
import pytest

from vcztools.calculate import (
    REF,
    SNP,
    UNCLASSIFIED,
    calculate_if_absent,
    get_variant_type,
)
from vcztools.constants import INT_FILL, INT_MISSING


@pytest.mark.parametrize(
    ("ref", "alt", "expected_type"),
    [
        ("A", "T", SNP),
        ("A", "A", REF),
        ("A", "<NON_REF>", REF),
        ("A", "<*>", REF),
        ("A", "", REF),
        ("A", "AA", UNCLASSIFIED),
        # these are all SNPs since they differ in one base
        ("AC", "TC", SNP),
        ("CA", "CT", SNP),
        ("CAGG", "CTGG", SNP),
    ],
)
def test_get_variant_type(ref, alt, expected_type):
    assert get_variant_type(ref, alt) == expected_type


def test_calculate_if_absent():
    gt = np.array(
        [
            [[0, 0], [0, 1], [1, 1]],
            [[0, 0], [0, 2], [2, 2]],
            [[0, 1], [1, 2], [2, 2]],
            [
                [INT_MISSING, INT_MISSING],
                [INT_MISSING, INT_MISSING],
                [INT_FILL, INT_FILL],
            ],
            [[INT_MISSING, INT_MISSING], [0, 3], [INT_FILL, INT_FILL]],
        ]
    )
    data = {"call_genotype": gt}

    calculate_if_absent(data, "variant_F_MISSING")

    np.testing.assert_array_equal(data["variant_N_MISSING"], np.array([0, 0, 0, 3, 2]))
    np.testing.assert_array_equal(
        data["variant_F_MISSING"], np.array([0, 0, 0, 1, 2 / 3])
    )
