import pytest

from vcztools.ftoa import ftoa

@pytest.mark.parametrize(
    ("float_", "string_", "precision"),
    [
        (1.12345, "1.123", 3),
        (1.12345, "1.1234", 4),
        (1.12345, "1.12345", 5),
    ]
)
def test_ftoa(float_, string_, precision):
    assert ftoa(float_, precision) == string_
