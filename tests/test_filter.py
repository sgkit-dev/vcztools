import numpy as np
import numpy.testing as nt
import pyparsing as pp
import pytest

from vcztools import bcftools_filter as filter_mod


class TestFilterExpressionParser:
    @pytest.fixture
    def fx_parser(self):
        return filter_mod.make_bcftools_filter_parser(map_vcf_identifiers=False)

    @pytest.mark.parametrize(
        "expression",
        [
            "",
            "| |",
            "a +",
            '"stri + 2',
        ],
    )
    def test_invalid_expressions(self, fx_parser, expression):
        with pytest.raises(pp.ParseException):
            fx_parser.parse_string(expression, parse_all=True)

    @pytest.mark.parametrize(
        ("expression", "exception_class"),
        [
            # NOTE: using an integer here so that we don't trigger the
            # generic string issue. Can fix this later when we've gotten
            # some partial string handling implemented
            ("INFO/HAYSTACK ~ 0", filter_mod.UnsupportedRegexError),
            ('DP="."', filter_mod.UnsupportedMissingDataError),
            ("ID!=@~/file", filter_mod.UnsupportedFileReferenceError),
            ("INFO/TAG=@file", filter_mod.UnsupportedFileReferenceError),
            ("INFO/X[0] == 1", filter_mod.UnsupportedArraySubscriptError),
            ("INFO/AF[0] > 0.3", filter_mod.UnsupportedArraySubscriptError),
            ("FORMAT/AD[0:0] > 30", filter_mod.UnsupportedArraySubscriptError),
            ("DP4[*] == 0", filter_mod.UnsupportedArraySubscriptError),
            ("FORMAT/DP[1-3] > 10", filter_mod.UnsupportedArraySubscriptError),
            ("FORMAT/DP[1-] < 7", filter_mod.UnsupportedArraySubscriptError),
            ("FORMAT/DP[0,2-4] > 20", filter_mod.UnsupportedArraySubscriptError),
            ("FORMAT/AD[0:*]", filter_mod.UnsupportedArraySubscriptError),
            ("FORMAT/AD[0:]", filter_mod.UnsupportedArraySubscriptError),
            ("FORMAT/AD[*:1]", filter_mod.UnsupportedArraySubscriptError),
            (
                "(DP4[0]+DP4[1])/(DP4[2]+DP4[3]) > 0.3",
                filter_mod.UnsupportedArraySubscriptError,
            ),
            ("binom(FMT/AD)", filter_mod.UnsupportedFunctionsError),
            ("fisher(INFO/DP4)", filter_mod.UnsupportedFunctionsError),
            ("fisher(FMT/ADF,FMT/ADR)", filter_mod.UnsupportedFunctionsError),
            ("N_PASS(GQ>90)", filter_mod.UnsupportedFunctionsError),
            ('TYPE="bnd"', filter_mod.UnsupportedTypeFieldError),
            # GT is rejected by Identifier.__init__ before the mapper
            # runs, so it fires under ``map_vcf_identifiers=False`` too.
            ("GT==0", filter_mod.UnsupportedGenotypeValuesError),
        ],
    )
    def test_unsupported_syntax(self, fx_parser, expression, exception_class):
        with pytest.raises(exception_class):
            fx_parser.parse_string(expression, parse_all=True)


class TestIdentifierResolutionErrors:
    """``Identifier.__init__`` raises two ``ValueError`` variants
    after consulting the name mapper — neither reachable with the
    identity mapper used by ``TestFilterExpressionParser``."""

    def test_undefined_tag(self):
        parser = filter_mod.make_bcftools_filter_parser(
            all_fields=set(), map_vcf_identifiers=True
        )
        with pytest.raises(ValueError, match='the tag "BLAH" is not defined'):
            parser.parse_string("BLAH>0", parse_all=True)

    def test_ambiguous_identifier(self):
        parser = filter_mod.make_bcftools_filter_parser(
            all_fields={"variant_DP", "call_DP"}, map_vcf_identifiers=True
        )
        with pytest.raises(ValueError, match="ambiguous filtering expression"):
            parser.parse_string("DP>0", parse_all=True)


class TestFilterExpressionSample:
    @pytest.mark.parametrize(
        ("expression", "expected_result"),
        [
            ('CHROM = "20"', [0, 0, 1, 1, 1, 1, 1, 1, 0]),
            ("POS < 1000", [1, 1, 0, 0, 0, 0, 0, 0, 1]),
            ("INFO/DP > 10", [0, 0, 1, 1, 0, 1, 0, 0, 0]),
            (
                "FMT/GQ > 20",
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [1, 1, 1],
                    [1, 0, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                    [0, 0, 1],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
            ),
            (
                "FMT/DP >= 5 && FMT/GQ > 10",
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 0, 1],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
            ),
            (
                "FMT/DP >= 5 & FMT/GQ > 10",
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 1, 1],
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
            ),
            (
                "QUAL > 10 || FMT/GQ > 10",
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
            ),
            (
                "(QUAL > 10 || FMT/GQ > 10) && POS > 100000",
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
            ),
            (
                "(FMT/DP >= 8 | FMT/GQ > 40) && POS > 100000",
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [1, 1, 1],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
            ),
        ],
    )
    def test(self, fx_sample_vcz, expression, expected_result):
        root = fx_sample_vcz.group
        data = {field: root[field][:] for field in root.keys()}
        filter_expr = filter_mod.BcftoolsFilter(
            field_names=set(root), include=expression
        )
        result = filter_expr.evaluate(data)
        nt.assert_array_equal(result, expected_result)

        filter_expr = filter_mod.BcftoolsFilter(
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
        fee = filter_mod.BcftoolsFilter(field_names=data.keys(), include=expression)
        result = fee.evaluate(numpify_values(data))
        nt.assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ('FILTER="PASS"', [False, True, False, False, False, False]),
            ('FILTER="."', [True, False, False, False, False, False]),
            ('FILTER="A"', [False, False, True, False, False, False]),
            ('FILTER!="A"', [True, True, False, True, True, True]),
            ('FILTER~"A"', [False, False, True, False, True, True]),
            ('FILTER="A;B"', [False, False, False, False, True, False]),
            ('FILTER="B;A"', [False, False, False, False, True, False]),
            ('FILTER!="A;B"', [True, True, True, True, False, True]),
            ('FILTER~"A;B"', [False, False, False, False, True, True]),
            ('FILTER~"B;A"', [False, False, False, False, True, True]),
            ('FILTER!~"A;B"', [True, True, True, True, False, False]),
        ],
    )
    def test_evaluate_filter_comparison(self, expression, expected):
        data = {
            "variant_filter": [
                [False, False, False, False],
                [True, False, False, False],
                [False, True, False, False],
                [False, False, True, False],
                [False, True, True, False],
                [False, True, True, True],
            ],
            "filter_id": ["PASS", "A", "B", "C"],
        }
        fee = filter_mod.BcftoolsFilter(include=expression)
        result = fee.evaluate(numpify_values(data))
        nt.assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ('TYPE="ref"', [True, False, False, False, False, False]),
            ('TYPE=="ref"', [True, False, False, False, False, False]),
            ('TYPE!="ref"', [False, True, True, True, True, True]),
            ('TYPE~"ref"', [True, False, False, False, False, False]),
            ('TYPE!~"ref"', [False, True, True, True, True, True]),
            ('TYPE="snp"', [False, True, False, False, False, True]),
            ('TYPE=="snp"', [False, True, False, False, False, True]),
            ('TYPE!="snp"', [True, False, True, True, True, False]),
            ('TYPE~"snp"', [False, True, False, False, True, True]),
            ('TYPE!~"snp"', [True, False, True, True, False, False]),
        ],
    )
    def test_evaluate_type_operation(self, expression, expected):
        data = {
            "variant_allele": [
                ["A", "", "", ""],
                ["A", "T", "", ""],
                ["A", "AT", "", ""],
                ["A", "CT", "", ""],
                ["A", "T", "CT", ""],
                ["A", "T", "G", "C"],
            ],
        }
        fee = filter_mod.BcftoolsFilter(include=expression)
        result = fee.evaluate(numpify_values(data))
        nt.assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("N_ALT == 0", [True, False, False, False]),
            ("N_ALT == 1", [False, True, False, False]),
            ("N_ALT >= 2", [False, False, True, True]),
            ("N_ALT <= 1", [True, True, False, False]),
            ("N_ALT < 3", [True, True, True, False]),
            ("N_ALT > 0", [False, True, True, True]),
            ("N_ALT >= 1 && N_ALT <= 1", [False, True, False, False]),
            ('N_ALT >= 1 && TYPE~"snp"', [False, True, True, True]),
        ],
    )
    def test_evaluate_n_alt(self, expression, expected):
        data = {
            "variant_allele": [
                ["A", "", "", ""],
                ["A", "T", "", ""],
                ["A", "T", "G", ""],
                ["A", "T", "G", "C"],
            ],
        }
        fee = filter_mod.BcftoolsFilter(include=expression)
        result = fee.evaluate(numpify_values(data))
        nt.assert_array_equal(result, expected)

    def test_n_alt_referenced_fields(self):
        fee = filter_mod.BcftoolsFilter(include="N_ALT >= 2")
        assert fee.referenced_fields == {"variant_allele"}
        assert fee.scope == "variant"

    @pytest.mark.parametrize(
        ("expr", "expected"),
        [
            ("a == b", {"variant_a", "variant_b"}),
            ("a == b + c", {"variant_a", "variant_b", "variant_c"}),
            ("(a + 1) < (b + c) - d / a", {f"variant_{x}" for x in "abcd"}),
            ("-(a + b)", {f"variant_{x}" for x in "ab"}),
        ],
    )
    def test_referenced_fields(self, expr, expected):
        fe = filter_mod.BcftoolsFilter(
            field_names={f"variant_{x}" for x in "abcd"}, include=expr
        )
        assert fe.referenced_fields == expected

    @pytest.mark.parametrize(
        ("expr", "expected"),
        [
            ("a == b", "(variant_a)==(variant_b)"),
            ("a + 1", "(variant_a)+(1)"),
            ("-a + 1", "(-(variant_a))+(1)"),
            ("a + 1 + 2", "(variant_a)+(1)+(2)"),
            ("a + (1 + 2)", "(variant_a)+((1)+(2))"),
            ("POS<10", "(variant_position)<(10)"),
            ('ID=="rs6054257"', "(variant_id)==('rs6054257')"),
        ],
    )
    def test_repr(self, expr, expected):
        fe = filter_mod.BcftoolsFilter(
            field_names={"variant_a", "variant_b"}, include=expr
        )
        assert repr(fe.parse_result[0]) == expected


class TestBcftoolsParser:
    @pytest.mark.parametrize(
        "expr",
        [
            "2",
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
            "-1 == 1 + 2 - 4",
            '("x" == "x")',
            '"x"',
            '"INFO/STRING"',
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
            ('("x" == "x")', {}),
            ('"x"', {}),
            ('"INFO/STRING"', {}),
            ('a == "string"', {"a": "string"}),
        ],
    )
    def test_python_string_expressions_data(self, expr, data):
        parser = filter_mod.make_bcftools_filter_parser(map_vcf_identifiers=False)
        parsed = parser.parse_string(expr, parse_all=True)
        result = parsed[0].eval(data)
        assert result == eval(expr, data)

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
            ("-a == -a", {"a": 1}),
            # Avoid -1 as a literal value: the filter evaluator treats it
            # as the int-missing sentinel (vcztools.constants.INT_MISSING).
            ("-a == b", {"a": 3, "b": -3}),
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
        ("expr", "data"),
        [
            ("call_a", {"call_a": [[[1]], [[2]], [[3]]]}),
        ],
    )
    def test_numpy_higher_dimension_arithmetic_expressions_data(self, expr, data):
        parser = filter_mod.make_bcftools_filter_parser(map_vcf_identifiers=False)
        parsed = parser.parse_string(expr, parse_all=True)
        npdata = numpify_values(data)
        with pytest.raises(filter_mod.UnsupportedHigherDimensionalFormatFieldsError):
            parsed[0].eval(npdata)

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
            ("a | b", {"a": [0, 1], "b": [1, 1]}, [True, True]),
            ("(a < 2) & (b > 1)", {"a": [0, 1], "b": [1, 2]}, [False, True]),
            # AND has precedence over OR
            ("t | f & f", {"t": [1], "f": [0]}, [True or False and False]),
            ("(t | f) & f", {"t": [1], "f": [0]}, [(True or False) and False]),
            (
                "call_a && call_b",
                {
                    "call_a": [
                        [0, 0, 0, 0],
                        [0, 0, 1, 1],
                        [0, 0, 0, 0],
                    ],
                    "call_b": [
                        [0, 0, 0, 0],
                        [0, 1, 0, 1],
                        [1, 1, 1, 1],
                    ],
                },
                [
                    [False, False, False, False],
                    [False, True, True, True],
                    # all False since condition a is not met (all 0)
                    [False, False, False, False],
                ],
            ),
            (
                "call_a || call_b",
                {
                    "call_a": [
                        [0, 0, 0, 0],
                        [0, 0, 1, 1],
                        [0, 0, 0, 0],
                    ],
                    "call_b": [
                        [0, 0, 0, 0],
                        [0, 1, 0, 1],
                        [1, 1, 1, 1],
                    ],
                },
                [
                    [False, False, False, False],
                    # all True since variant site is included
                    [True, True, True, True],
                    [True, True, True, True],
                ],
            ),
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
            filter_mod.BcftoolsFilter(include="x", exclude="y")

    def test_undefined_filter_name(self):
        flt = filter_mod.BcftoolsFilter(
            field_names={"filter_id", "variant_filter"},
            include='FILTER="NOPE"',
        )
        data = {
            "filter_id": np.array(["PASS", "q10"]),
            "variant_filter": np.array([[True, False], [False, True]]),
        }
        with pytest.raises(ValueError, match='The filter "NOPE" is not present'):
            flt.evaluate(data)
