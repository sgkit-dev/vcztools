import pytest

from vcztools.calculate import REF, SNP, UNCLASSIFIED, get_variant_type


@pytest.mark.parametrize(
    ("ref", "alt", "expected_type"),
    [
        ("A", "T", SNP),
        ("A", "A", REF),
        ("A", "<NON_REF>", REF),
        ("A", "<*>", REF),
        ("A", "", REF),
        ("A", "AA", UNCLASSIFIED),
    ],
)
def test_get_variant_type(ref, alt, expected_type):
    assert get_variant_type(ref, alt) == expected_type
