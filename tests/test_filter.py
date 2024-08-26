import pathlib

import numpy as np
import pyparsing as pp
import pytest
import zarr
from numpy.testing import assert_array_equal

from tests.utils import vcz_path_cache
from vcztools.filter import FilterExpressionEvaluator, FilterExpressionParser


class TestFilterExpressionParser:
    @pytest.fixture()
    def identifier_parser(self, parser):
        return parser._identifier_parser

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
                "QUAL>10 | FMT/DP>10 & FMT/GQ>10",
                [
                    [
                        ["QUAL", ">", 10],
                        "|",
                        [["FMT/DP", ">", 10], "&", ["FMT/GQ", ">", 10]],
                    ],
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
    def test_valid_expressions(self, parser, expression, expected_result):
        assert parser(expression=expression).as_list() == expected_result


class TestFilterExpressionEvaluator:
    @pytest.mark.parametrize(
        ("expression", "expected_result"),
        [
            ("POS < 1000", [1, 1, 0, 0, 0, 0, 0, 0, 1]),
            ("FMT/GQ > 20", [0, 0, 1, 1, 1, 1, 1, 0, 0]),
            ("FMT/DP >= 5 && FMT/GQ > 10", [0, 0, 1, 1, 1, 0, 0, 0, 0]),
            ("FMT/DP >= 5 & FMT/GQ>10", [0, 0, 1, 0, 1, 0, 0, 0, 0]),
            ("QUAL > 10 || FMT/GQ>10", [0, 0, 1, 1, 1, 1, 1, 0, 0]),
            ("(QUAL > 10 || FMT/GQ>10) && POS > 100000", [0, 0, 0, 0, 1, 1, 1, 0, 0]),
            ("(FMT/DP >= 8 | FMT/GQ>40) && POS > 100000", [0, 0, 0, 0, 0, 1, 0, 0, 0]),
            ("INFO/DP > 10", [0, 0, 1, 1, 0, 1, 0, 0, 0]),
            ("GT > 0", [1, 1, 1, 1, 1, 0, 1, 0, 1]),
            ("GT > 0 & FMT/HQ >= 10", [0, 0, 1, 1, 1, 0, 0, 0, 0]),
        ],
    )
    def test(self, expression, expected_result):
        original = pathlib.Path("tests/data/vcf") / "sample.vcf.gz"
        vcz = vcz_path_cache(original)
        root = zarr.open(vcz, mode="r")

        parser = FilterExpressionParser()
        parse_results = parser(expression)[0]
        evaluator = FilterExpressionEvaluator(parse_results)
        assert_array_equal(evaluator(root, 0), expected_result)

        invert_evaluator = FilterExpressionEvaluator(parse_results, invert=True)
        assert_array_equal(invert_evaluator(root, 0), np.logical_not(expected_result))
