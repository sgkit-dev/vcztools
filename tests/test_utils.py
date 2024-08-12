import pyparsing as pp
import pytest
from numpy.testing import assert_array_equal

from vcztools.utils import FilterExpressionParser, search


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


class TestFilterExpressionParser:
    @pytest.fixture()
    def identifier_parser(self):
        return FilterExpressionParser()._identifier_parser

    @pytest.fixture()
    def parser(self):
        return FilterExpressionParser()

    @pytest.mark.parametrize(
        ("expression", "expected_result"),
        [
            ("1", [1]),
            ("1.0", [1.0]),
            ("1e-4", [0.0001]),
            ('"String"', ["String"]),
            ("POS", ["POS"]),
            ("INFO/DP", ["INFO/DP"]),
            ("FORMAT/GT", ["FORMAT/GT"]),
            ("FMT/GT", ["FMT/GT"]),
            ("GT", ["GT"]),
        ],
    )
    def test_valid_identifiers(self, identifier_parser, expression, expected_result):
        assert identifier_parser(expression).as_list() == expected_result

    @pytest.mark.parametrize(
        "expression",
        [
            "",
            "FORMAT/ GT",
            "format / GT",
            "fmt / GT",
            "info / DP",
            "'String'",
        ],
    )
    def test_invalid_identifiers(self, identifier_parser, expression):
        with pytest.raises(pp.ParseException):
            identifier_parser(expression)

    @pytest.mark.parametrize(
        ("expression", "expected_result"),
        [
            ("POS>=100", [["POS", ">=", 100]]),
            (
                "FMT/DP>10 && FMT/GQ>10",
                [[["FMT/DP", ">", 10], "&&", ["FMT/GQ", ">", 10]]],
            ),
            ("QUAL>10 || FMT/GQ>10", [[["QUAL", ">", 10], "||", ["FMT/GQ", ">", 10]]]),
            (
                "FMT/DP>10 && FMT/GQ>10 || QUAL > 10",
                [
                    [
                        [["FMT/DP", ">", 10], "&&", ["FMT/GQ", ">", 10]],
                        "||",
                        ["QUAL", ">", 10],
                    ]
                ],
            ),
            (
                "QUAL>10 || FMT/DP>10 && FMT/GQ>10",
                [
                    [
                        ["QUAL", ">", 10],
                        "||",
                        [["FMT/DP", ">", 10], "&&", ["FMT/GQ", ">", 10]],
                    ]
                ],
            ),
            (
                "(QUAL>10 || FMT/DP>10) && FMT/GQ>10",
                [
                    [
                        [["QUAL", ">", 10], "||", ["FMT/DP", ">", 10]],
                        "&&",
                        ["FMT/GQ", ">", 10],
                    ]
                ],
            ),
        ],
    )
    def test_validd_expressions(self, parser, expression, expected_result):
        assert parser(expression).as_list() == expected_result
