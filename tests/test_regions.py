from typing import Optional
import pytest
from vcztools.regions import parse_region


@pytest.mark.parametrize(
    "targets, expected",
    [
        ("chr1", ("chr1", None, None)),
        ("chr1:12", ("chr1", 12, 12)),
        ("chr1:12-", ("chr1", 12, None)),
        ("chr1:12-103", ("chr1", 12, 103)),
    ],
)
def test_parse_region(
    targets: str, expected: tuple[str, Optional[int], Optional[int]]
):
    assert parse_region(targets) == expected
