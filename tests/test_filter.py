import pathlib

import numpy as np
import numpy.testing as nt
import pyparsing as pp
import pytest
import zarr

from tests.utils import vcz_path_cache
from vcztools import filter as filter_mod


class TestFilterExpressionParser:
    @pytest.mark.parametrize(
        "expression",
        [
            "",
            "| |",
            "a +",
            '"stri + 2',
        ],
    )
    def test_invalid_expressions(self, expression):
        parser = filter_mod.make_bcftools_filter_parser(map_vcf_identifiers=False)
        with pytest.raises(pp.ParseException):
            parser.parse_string(expression, parse_all=True)


class TestFilterExpressionSample:
    @pytest.mark.parametrize(
        ("expression", "expected_result"),
        [
            ("POS < 1000", [1, 1, 0, 0, 0, 0, 0, 0, 1]),
            ("INFO/DP > 10", [0, 0, 1, 1, 0, 1, 0, 0, 0]),
            # Not supporting format fields for now: #180
            # ("FMT/GQ > 20", [0, 0, 1, 1, 1, 1, 1, 0, 0]),
            # ("FMT/DP >= 5 && FMT/GQ > 10", [0, 0, 1, 1, 1, 0, 0, 0, 0]),
            # ("GT > 0", [1, 1, 1, 1, 1, 0, 1, 0, 1]),
            # ("GT > 0 & FMT/HQ >= 10", [0, 0, 1, 1, 1, 0, 0, 0, 0]),
            # ("FMT/DP >= 5 & FMT/GQ>10", [0, 0, 1, 0, 1, 0, 0, 0, 0]),
            # ("QUAL > 10 || FMT/GQ>10", [0, 0, 1, 1, 1, 1, 1, 0, 0]),
            # ("(QUAL > 10 || FMT/GQ>10) && POS > 100000", [0, 0, 0, 0, 1, 1, 1, 0, 0]),
            # ("(FMT/DP >= 8 | FMT/GQ>40) && POS > 100000",
            #     [0, 0, 0, 0, 0, 1, 0, 0, 0]),
        ],
    )
    def test(self, expression, expected_result):
        original = pathlib.Path("tests/data/vcf") / "sample.vcf.gz"
        vcz = vcz_path_cache(original)
        root = zarr.open(vcz, mode="r")
        data = {field: root[field][:] for field in root.keys()}
        filter_expr = filter_mod.FilterExpression(
            field_names=set(root), include=expression
        )
        result = filter_expr.evaluate(data)
        nt.assert_array_equal(result, expected_result)

        filter_expr = filter_mod.FilterExpression(
            field_names=set(root), exclude=expression
        )
        result = filter_expr.evaluate(data)
        nt.assert_array_equal(result, np.logical_not(expected_result))


def numpify_values(data):
    return {k: np.array(v) for k, v in data.items()}


class TestFilterExpression:
    @pytest.mark.parametrize(
        ("expression", "data", "expected"),
        [
            ("POS<5", {"variant_position": [1, 5, 6, 10]}, [1, 0, 0, 0]),
            ("INFO/XX>=10", {"variant_XX": [1, 5, 6, 10]}, [0, 0, 0, 1]),
            ("INFO/XX / 2 >=5", {"variant_XX": [1, 5, 6, 10]}, [0, 0, 0, 1]),
            ("POS<5 | POS>8", {"variant_position": [1, 5, 6, 10]}, [1, 0, 0, 1]),
            (
                "POS<0 & POS<1 & POS<2 & POS<3 & POS<4",
                {"variant_position": range(10)},
                np.zeros(10, dtype=bool),
            ),
        ],
    )
    def test_evaluate(self, expression, data, expected):
        fee = filter_mod.FilterExpression(include=expression)
        result = fee.evaluate(numpify_values(data))
        nt.assert_array_equal(result, expected)


class TestBcftoolsParser:
    @pytest.mark.parametrize(
        "expr",
        [
            "2",
            '"x"',
            '"INFO/STRING"',
            "2 + 2",
            "(2 + 3) / 2",
            "2 / (2 + 3)",
            "1 + 1 + 1 + 1 + 1",
            "5 * (2 / 3)",
            "5 * 2 / 3",
            "1 + 2 - 3 / 4 * 5 + 6 * 7 / 8",
            "5 / (1 + 2 - 4) / (4 * 5 + 6 * 7 / 8)",
            "5 < 2",
            "5 > 2",
            "0 == 0",
            "0 != 0",
            "(1 + 2) == 0",
            "1 + 2 == 0",
            "1 + 2 == 1 + 2 + 3",
            "(1 + 2) == (1 + 2 + 3)",
            "(1 == 1) != (2 == 2)",
            '("x" == "x")',
        ],
    )
    def test_python_arithmetic_expressions(self, expr):
        parser = filter_mod.make_bcftools_filter_parser()
        parsed = parser.parse_string(expr, parse_all=True)
        result = parsed[0].eval({})
        assert result == eval(expr)

    @pytest.mark.parametrize(
        ("expr", "data"),
        [
            ("a", {"a": 1}),
            ("a + a", {"a": 1}),
            ("a + 2 * a - 1", {"a": 7}),
            ("a - b < a + b", {"a": 7, "b": 6}),
            ("(a - b) < (a + b)", {"a": 7, "b": 6}),
            ("(a - b) < (a + b)", {"a": 7.0, "b": 6.666}),
            ("a == a", {"a": 1}),
            ('a == "string"', {"a": "string"}),
        ],
    )
    def test_python_arithmetic_expressions_data(self, expr, data):
        parser = filter_mod.make_bcftools_filter_parser(map_vcf_identifiers=False)
        parsed = parser.parse_string(expr, parse_all=True)
        result = parsed[0].eval(data)
        assert result == eval(expr, data)

    @pytest.mark.parametrize(
        ("expr", "data"),
        [
            ("a", {"a": [1, 2, 3]}),
            ("a + a", {"a": [1, 2, 3]}),
            ("1 + a + a", {"a": [1, 2, 3]}),
            ("a + b", {"a": [1, 2, 3], "b": [5, 6, 7]}),
            ("(a + b) < c", {"a": [1, 2, 3], "b": [5, 6, 7], "c": [5, 10, 15]}),
        ],
    )
    def test_numpy_arithmetic_expressions_data(self, expr, data):
        parser = filter_mod.make_bcftools_filter_parser(map_vcf_identifiers=False)
        parsed = parser.parse_string(expr, parse_all=True)
        npdata = numpify_values(data)
        result = parsed[0].eval(npdata)
        evaled = eval(expr, npdata)
        nt.assert_array_equal(result, evaled)

    @pytest.mark.parametrize(
        ("expr", "expected"),
        [
            ("1 & 1", True),
            ("0 & 1", False),
            ("1 & 0", False),
            ("0 & 0", False),
            ("1 | 1", True),
            ("0 | 1", True),
            ("1 | 0", True),
            ("0 | 0", False),
            ("(1 < 2) | 0", True),
            ("(1 < 2) & 0", False),
        ],
    )
    def test_boolean_operator_expressions(self, expr, expected):
        parser = filter_mod.make_bcftools_filter_parser()
        parsed = parser.parse_string(expr, parse_all=True)
        result = parsed[0].eval({})
        assert result == expected

    @pytest.mark.parametrize(
        ("expr", "data", "expected"),
        [
            ("a == b", {"a": [0, 1], "b": [1, 1]}, [False, True]),
            ("a = b", {"a": [0, 1], "b": [1, 1]}, [False, True]),
            ("a & b", {"a": [0, 1], "b": [1, 1]}, [False, True]),
            ("a && b", {"a": [0, 1], "b": [1, 1]}, [False, True]),
            ("a | b", {"a": [0, 1], "b": [1, 1]}, [True, True]),
            ("a || b", {"a": [0, 1], "b": [1, 1]}, [True, True]),
            ("(a < 2) & (b > 1)", {"a": [0, 1], "b": [1, 2]}, [False, True]),
        ],
    )
    def test_boolean_operator_expressions_data(self, expr, data, expected):
        parser = filter_mod.make_bcftools_filter_parser(map_vcf_identifiers=False)
        parsed = parser.parse_string(expr, parse_all=True)
        result = parsed[0].eval(numpify_values(data))
        nt.assert_array_equal(result, expected)


class TestAPIErrors:
    def test_include_and_exclude(self):
        with pytest.raises(ValueError, match="Cannot handle both an include "):
            filter_mod.FilterExpression(include="x", exclude="y")
