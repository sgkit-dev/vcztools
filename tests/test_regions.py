from typing import Optional
import pytest
from vcztools.regions import parse_targets_string


@pytest.mark.parametrize(
    "targets, expected",
    [
        ("chr1:12-103", ("chr1", 12, 103)),
    ],
)
def test_parse_targets_string(
    targets: str, expected: tuple[str, Optional[int], Optional[int]]
):
    assert parse_targets_string(targets) == expected
