import numpy as np
import numpy.testing as nt
import pyparsing as pp
import pytest

from tests import utils
from vcztools import bcftools_filter as filter_mod
from vcztools import constants
from vcztools import virtual_fields as virtual_fields_mod


def _materialise_virtuals(data, names):
    """Populate ``data`` (a per-chunk-style dict from
    :func:`numpify_values`) with the named virtual fields by running
    the registry's compute functions in dependency order. Mirrors what
    ``VczReader._variant_chunks_gen`` does for the filter-side data
    dict so unit-level :meth:`BcftoolsFilter.evaluate` tests stay
    representative."""
    cache: dict = {}
    for name in names:
        vf = virtual_fields_mod.REGISTRY[name]
        if not all(d in data for d in vf.deps):
            raise KeyError(f"deps for {name} not in test data")
        deps = {dep: np.asarray(data[dep]) for dep in vf.deps}
        # These value-derived fields ignore the chunk context (only the
        # structural variant_index field reads it).
        data[name] = vf.compute(deps, cache, None)
    return data


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

    @pytest.mark.parametrize(
        "expression",
        [
            "N_ALT >= 1",
            "N_MISSING == 0",
            "F_MISSING < 0.5",
            "N_MISSING > 0 && F_MISSING < 0.5",
        ],
    )
    def test_supported_calculated_variables(self, fx_parser, expression):
        # These calculated variables are intercepted by their own
        # pp.Keyword parse actions, so they never reach Identifier and
        # the UnsupportedCalculatedVariableError check.
        fx_parser.parse_string(expression, parse_all=True)


class TestIdentifierResolutionErrors:
    """``Identifier.__init__`` raises errors that depend on the name
    mapper, so they aren't reachable with the identity mapper used by
    ``TestFilterExpressionParser``."""

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

    @pytest.mark.parametrize(
        "expression",
        ["MAC==1", "MAF>=0.05", "ILEN>0", "N_SAMPLES>100"],
    )
    def test_unsupported_calculated_variable(self, expression):
        # When the calculated-variable tag isn't otherwise defined in
        # the dataset, parsing surfaces the dedicated error.
        parser = filter_mod.make_bcftools_filter_parser(
            all_fields=set(), map_vcf_identifiers=True
        )
        with pytest.raises(filter_mod.UnsupportedCalculatedVariableError):
            parser.parse_string(expression, parse_all=True)

    @pytest.mark.parametrize(
        ("expression", "all_fields"),
        [
            ("AC>0", {"variant_AC"}),
            ("AN>0", {"variant_AN"}),
            ("AF<0.5", {"variant_AF"}),
        ],
    )
    def test_ac_an_af_resolve_via_stored_field(self, expression, all_fields):
        # AC / AN / AF resolve through the ordinary Identifier rule.
        # ``all_fields`` is the union the caller passes in — for the
        # VczReader that's ``field_names | virtual_field_names`` so
        # virtual entries make the parser accept the expression even
        # without a stored counterpart.
        parser = filter_mod.make_bcftools_filter_parser(
            all_fields=all_fields, map_vcf_identifiers=True
        )
        parser.parse_string(expression, parse_all=True)

    @pytest.mark.parametrize("expression", ["AC>0", "AN>0", "AF<0.5"])
    def test_ac_an_af_undefined_without_stored_or_virtual(self, expression):
        # Neither a stored ``variant_AC`` nor a virtual one available →
        # the identifier path falls through to the generic "the tag X
        # is not defined" error.
        parser = filter_mod.make_bcftools_filter_parser(
            all_fields=set(), map_vcf_identifiers=True
        )
        with pytest.raises(ValueError, match="is not defined"):
            parser.parse_string(expression, parse_all=True)


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
            utils.FilterReader(set(root)), include=expression
        )
        result = filter_expr.evaluate(data)
        nt.assert_array_equal(result, expected_result)

        filter_expr = filter_mod.BcftoolsFilter(
            utils.FilterReader(set(root)), exclude=expression
        )
        result = filter_expr.evaluate(data)
        nt.assert_array_equal(result, np.logical_not(expected_result))


def numpify_values(data):
    return {k: np.array(v) for k, v in data.items()}


class TestEvaluationHelpers:
    """Direct characterisation of the array-combination helpers, exercised on
    raw numpy arrays so each function's contract is pinned independently of any
    parsed expression."""

    def test_align_dims_expands_one_d_against_two_d(self):
        one_d = np.array([True, False, True])
        two_d = np.array([[1, 2], [3, 4], [5, 6]])
        a, b = filter_mod._align_dims(one_d, two_d)
        assert a.shape == (3, 1)
        assert b.shape == (3, 2)
        a, b = filter_mod._align_dims(two_d, one_d)
        assert a.shape == (3, 2)
        assert b.shape == (3, 1)

    def test_align_dims_passes_through_matching_ranks(self):
        for left, right in [
            (np.array([1, 2]), np.array([3, 4])),  # both 1-D
            (np.zeros((2, 3)), np.ones((2, 3))),  # both 2-D
            (np.array(5), np.array([1, 2])),  # scalar + 1-D
        ]:
            a, b = filter_mod._align_dims(left, right)
            assert a.shape == np.asarray(left).shape
            assert b.shape == np.asarray(right).shape

    def test_align_dims_coerces_python_inputs(self):
        a, b = filter_mod._align_dims([1, 2], 5)
        assert isinstance(a, np.ndarray)
        assert isinstance(b, np.ndarray)

    def test_collapse_to_variant(self):
        two_d = np.array([[False, False], [False, True], [True, True]])
        nt.assert_array_equal(
            filter_mod._collapse_to_variant(two_d), [False, True, True]
        )
        one_d = np.array([True, False])
        nt.assert_array_equal(filter_mod._collapse_to_variant(one_d), [True, False])

    @pytest.mark.parametrize(
        ("fn", "expected"),
        [(filter_mod.single_and, False), (filter_mod.single_or, True)],
    )
    def test_single_ops_both_one_d(self, fn, expected):
        a = np.array([True, False])
        b = np.array([False, False])
        nt.assert_array_equal(fn(a, b), [expected, False])

    def test_single_and_broadcasts_one_d_across_axis(self):
        variant = np.array([True, False])  # per-variant
        per_sample = np.array([[True, True], [True, True]])  # per-sample
        # The variant mask gates every sample column.
        nt.assert_array_equal(
            filter_mod.single_and(variant, per_sample),
            [[True, True], [False, False]],
        )

    def test_single_or_two_d_same_width_elementwise(self):
        a = np.array([[True, False], [False, False]])
        b = np.array([[False, False], [True, False]])
        nt.assert_array_equal(
            filter_mod.single_or(a, b), [[True, False], [True, False]]
        )

    def test_double_and_scalar_broadcasts(self):
        per_sample = np.array([[True, False], [True, True]])
        # A truthy scalar passes the per-sample mask through unchanged.
        nt.assert_array_equal(
            filter_mod.double_and(np.array(1), per_sample), per_sample
        )

    def test_double_and_cross_sample_semantics(self):
        # Each side must match in *some* sample for the site to survive; the
        # surviving site then keeps the per-sample union.
        a = np.array([[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        b = np.array([[0, 0, 0, 0], [0, 1, 0, 1], [1, 1, 1, 1]])
        nt.assert_array_equal(
            filter_mod.double_and(a, b),
            [
                [False, False, False, False],
                [False, True, True, True],
                [False, False, False, False],
            ],
        )

    def test_double_or_cross_sample_semantics(self):
        a = np.array([[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        b = np.array([[0, 0, 0, 0], [0, 1, 0, 1], [1, 1, 1, 1]])
        nt.assert_array_equal(
            filter_mod.double_or(a, b),
            [
                [False, False, False, False],
                [True, True, True, True],
                [True, True, True, True],
            ],
        )


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
            # Per-allele INFO/AC (2-D, Number=A) compared against per-variant
            # INFO/AN (1-D). Variants: monomorphic-REF, polymorphic,
            # monomorphic-ALT. AC>0 alone keeps the monomorphic-ALT site;
            # the canonical "has variation" idiom drops it.
            (
                "AC>0",
                {"variant_AC": [[0], [2], [4]], "variant_AN": [4, 4, 4]},
                [0, 1, 1],
            ),
            (
                "AC>0 && AC<AN",
                {"variant_AC": [[0], [2], [4]], "variant_AN": [4, 4, 4]},
                [0, 1, 0],
            ),
            (
                "AC<AN",
                {"variant_AC": [[0], [2], [4]], "variant_AN": [4, 4, 4]},
                [1, 1, 0],
            ),
            (
                "AC==AN",
                {"variant_AC": [[0], [2], [4]], "variant_AN": [4, 4, 4]},
                [0, 0, 1],
            ),
            # Arithmetic between a 2-D and a 1-D operand must broadcast too.
            (
                "AC / AN > 0.4",
                {"variant_AC": [[0], [2], [4]], "variant_AN": [4, 4, 4]},
                [0, 1, 1],
            ),
            # bcftools missing semantics for tag-vs-tag equality: a site whose
            # AC and AN are both absent (sentinel) compares equal, so `==` is
            # True and `!=` is False there. The fourth variant below is missing.
            (
                "AC==AN",
                {"variant_AC": [[0], [2], [4], [-1]], "variant_AN": [4, 4, 4, -1]},
                [0, 0, 1, 1],
            ),
            (
                "AC!=AN",
                {"variant_AC": [[0], [2], [4], [-1]], "variant_AN": [4, 4, 4, -1]},
                [1, 1, 0, 0],
            ),
            (
                "AC>0 && AC<AN",
                {"variant_AC": [[0], [2], [4], [-1]], "variant_AN": [4, 4, 4, -1]},
                [0, 1, 0, 0],
            ),
        ],
    )
    def test_evaluate(self, expression, data, expected):
        fee = filter_mod.BcftoolsFilter(
            utils.FilterReader(data.keys()), include=expression
        )
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
        fee = filter_mod.BcftoolsFilter(utils.FilterReader(()), include=expression)
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
        fee = filter_mod.BcftoolsFilter(utils.FilterReader(()), include=expression)
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
        fee = filter_mod.BcftoolsFilter(
            utils.FilterReader({"variant_N_ALT", "variant_allele"}), include=expression
        )
        evaluated = _materialise_virtuals(numpify_values(data), ["variant_N_ALT"])
        result = fee.evaluate(evaluated)
        nt.assert_array_equal(result, expected)

    def test_n_alt_referenced_fields(self):
        fee = filter_mod.BcftoolsFilter(
            utils.FilterReader({"variant_N_ALT"}), include="N_ALT >= 2"
        )
        assert fee.referenced_fields == {"variant_N_ALT"}
        assert fee.scope == "variant"

    # Synthetic 3-sample, diploid genotype matrix used by the
    # N_MISSING / F_MISSING tests below. Per-variant missing counts:
    # [0, 0, 0, 3, 2]; fractions: [0, 0, 0, 1.0, 2/3].
    _MISSING_GT_DATA = {
        "call_genotype": [
            [[0, 0], [0, 1], [1, 1]],
            [[0, 0], [0, 2], [2, 2]],
            [[0, 1], [1, 2], [2, 2]],
            [
                [constants.INT_MISSING, constants.INT_MISSING],
                [constants.INT_MISSING, constants.INT_MISSING],
                [constants.INT_FILL, constants.INT_FILL],
            ],
            [
                [constants.INT_MISSING, constants.INT_MISSING],
                [0, 3],
                [constants.INT_FILL, constants.INT_FILL],
            ],
        ],
        "variant_position": [100, 200, 300, 400, 500],
    }

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("N_MISSING == 0", [True, True, True, False, False]),
            ("N_MISSING == 2", [False, False, False, False, True]),
            ("N_MISSING == 3", [False, False, False, True, False]),
            ("N_MISSING >= 1", [False, False, False, True, True]),
            ("N_MISSING > 2", [False, False, False, True, False]),
            ("N_MISSING <= 2", [True, True, True, False, True]),
            ("N_MISSING != 0", [False, False, False, True, True]),
            ("N_MISSING > 0 && N_MISSING < 3", [False, False, False, False, True]),
            ("N_MISSING == 0 || N_MISSING == 3", [True, True, True, True, False]),
        ],
    )
    def test_evaluate_n_missing(self, expression, expected):
        fee = filter_mod.BcftoolsFilter(
            utils.FilterReader({"variant_N_MISSING", "call_genotype"}),
            include=expression,
        )
        data = _materialise_virtuals(
            numpify_values(self._MISSING_GT_DATA), ["variant_N_MISSING"]
        )
        result = fee.evaluate(data)
        nt.assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("F_MISSING == 0", [True, True, True, False, False]),
            ("F_MISSING < 0.5", [True, True, True, False, False]),
            ("F_MISSING > 0.5", [False, False, False, True, True]),
            ("F_MISSING == 1", [False, False, False, True, False]),
            ("F_MISSING > 0", [False, False, False, True, True]),
            ("F_MISSING >= 0.6 && F_MISSING < 1", [False, False, False, False, True]),
        ],
    )
    def test_evaluate_f_missing(self, expression, expected):
        fee = filter_mod.BcftoolsFilter(
            utils.FilterReader({"variant_F_MISSING", "call_genotype"}),
            include=expression,
        )
        data = _materialise_virtuals(
            numpify_values(self._MISSING_GT_DATA), ["variant_F_MISSING"]
        )
        result = fee.evaluate(data)
        nt.assert_array_equal(result, expected)

    def test_evaluate_n_missing_combined_with_other(self):
        # Combination with TYPE checks that the N_MISSING virtual field
        # interoperates with other variant-scoped operators.
        data = {
            "call_genotype": np.array(
                [
                    [[0, 0], [0, 1]],  # 0 missing, SNP
                    [[-1, -1], [-1, -1]],  # 2 missing, SNP
                    [[0, 0], [-1, -1]],  # 1 missing, indel
                ]
            ),
            "variant_allele": np.array(
                [
                    ["A", "T", ""],
                    ["A", "C", ""],
                    ["A", "AT", ""],
                ]
            ),
            "variant_position": np.array([1, 2, 3]),
        }
        fee = filter_mod.BcftoolsFilter(
            utils.FilterReader({"variant_N_MISSING", "variant_allele"}),
            include='N_MISSING > 0 && TYPE~"snp"',
        )
        data = _materialise_virtuals(data, ["variant_N_MISSING"])
        nt.assert_array_equal(fee.evaluate(data), [False, True, False])

    def test_n_missing_referenced_fields(self):
        # The Identifier resolves to ``variant_N_MISSING``; the dispatcher
        # is responsible for materialising the value from ``call_genotype``
        # before evaluate runs.
        fee = filter_mod.BcftoolsFilter(
            utils.FilterReader({"variant_N_MISSING"}), include="N_MISSING == 0"
        )
        assert fee.referenced_fields == {"variant_N_MISSING"}
        assert fee.scope == "variant"

    def test_f_missing_referenced_fields(self):
        fee = filter_mod.BcftoolsFilter(
            utils.FilterReader({"variant_F_MISSING"}), include="F_MISSING < 0.5"
        )
        assert fee.referenced_fields == {"variant_F_MISSING"}
        assert fee.scope == "variant"

    def test_combined_missing_referenced_fields(self):
        fee = filter_mod.BcftoolsFilter(
            utils.FilterReader(
                {
                    "variant_DP",
                    "variant_N_MISSING",
                    "variant_F_MISSING",
                }
            ),
            include="N_MISSING == 0 && F_MISSING < 0.05 && INFO/DP > 10",
        )
        assert fee.referenced_fields == {
            "variant_N_MISSING",
            "variant_F_MISSING",
            "variant_DP",
        }

    @pytest.mark.parametrize("tag", ["N_MISSING", "F_MISSING"])
    def test_missing_unavailable_without_call_genotype(self, tag):
        # On a store with no call_genotype these are not virtual fields,
        # so the field-name resolution surface omits them and the filter
        # rejects the expression rather than silently passing all rows.
        with pytest.raises(ValueError, match=f'the tag "{tag}" is not defined'):
            filter_mod.BcftoolsFilter(
                utils.FilterReader({"variant_position"}), include=f"{tag} == 0"
            )

    def test_evaluate_n_missing_with_fill_only(self):
        # Mixed-ploidy samples encode the unused ploidy slot as
        # INT_FILL (-2) — those rows are not "missing" if at least one
        # slot is a valid allele.
        data = {
            "call_genotype": np.array(
                [
                    [[0, constants.INT_FILL], [0, 1]],  # haploid + diploid
                    [
                        [constants.INT_FILL, constants.INT_FILL],
                        [
                            constants.INT_MISSING,
                            constants.INT_MISSING,
                        ],
                    ],
                ]
            ),
            "variant_position": np.array([1, 2]),
        }
        fee = filter_mod.BcftoolsFilter(
            utils.FilterReader({"variant_N_MISSING", "call_genotype"}),
            include="N_MISSING == 2",
        )
        data = _materialise_virtuals(data, ["variant_N_MISSING"])
        nt.assert_array_equal(fee.evaluate(data), [False, True])

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
            utils.FilterReader({f"variant_{x}" for x in "abcd"}), include=expr
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
            # The CHROM / FILTER / TYPE operators have dedicated node classes.
            ('CHROM="20"', "(variant_contig)=('20')"),
            ('FILTER="PASS"', "(variant_filter)=('PASS')"),
            ('TYPE="snp"', "(variant_allele)=('snp')"),
        ],
    )
    def test_repr(self, expr, expected):
        fe = filter_mod.BcftoolsFilter(
            utils.FilterReader({"variant_a", "variant_b", "variant_filter"}),
            include=expr,
        )
        assert repr(fe.parse_result[0]) == expected


class TestMixedDimensionEvaluation:
    """Expressions that combine operands of differing dimensionality: a
    per-allele field (variant-scope, 2-D, axis 1 = alleles) and a per-sample
    field (sample-scope, 2-D, axis 1 = samples)."""

    def _evaluate(self, expression, data):
        fee = filter_mod.BcftoolsFilter(
            utils.FilterReader(data.keys()), include=expression
        )
        return fee.evaluate(numpify_values(data))

    def test_per_allele_and_per_sample_logical(self):
        # AC>0 collapses over alleles to a per-variant decision, then gates
        # every sample column of the FMT/DP>0 mask. The whole expression is
        # sample-scope, so the result stays 2-D (per-sample).
        data = {
            "variant_AC": [[0], [2]],
            "call_DP": [[1, 0], [5, 5]],
        }
        result = self._evaluate("AC>0 & FMT/DP>0", data)
        nt.assert_array_equal(result, [[False, False], [True, True]])

    def test_per_allele_or_per_sample_logical(self):
        data = {
            "variant_AC": [[0], [2]],
            "call_DP": [[1, 0], [0, 0]],
        }
        result = self._evaluate("AC>0 | FMT/DP>0", data)
        nt.assert_array_equal(result, [[True, False], [True, True]])

    def test_chained_mixed_scope(self):
        # AC>0 (per-allele) and INFO/DP>0 (per-variant) both collapse/broadcast
        # to gate the per-sample FMT/DP>0 mask.
        data = {
            "variant_AC": [[2], [2]],
            "variant_DP": [5, 0],
            "call_DP": [[5, 5], [5, 5]],
        }
        result = self._evaluate("AC>0 & FMT/DP>0 & INFO/DP>0", data)
        # variant 1 fails the per-variant INFO/DP>0 leg → all samples dropped.
        nt.assert_array_equal(result, [[True, True], [False, False]])

    def test_per_variant_scalar_vs_per_sample(self):
        # A 1-D INFO field broadcasts against a 2-D FORMAT field in a comparison.
        data = {
            "variant_DP": [10, 2],
            "call_DP": [[5, 12], [1, 3]],
        }
        result = self._evaluate("INFO/DP > FMT/DP", data)
        nt.assert_array_equal(result, [[True, False], [True, False]])


class TestRaggedFieldEvaluation:
    """Per-allele (Number=A) arrays are padded with INT_FILL to the chunk's
    widest row; fill and missing sentinels must never satisfy a comparison."""

    def _evaluate(self, expression, data):
        fee = filter_mod.BcftoolsFilter(
            utils.FilterReader(data.keys()), include=expression
        )
        return fee.evaluate(numpify_values(data))

    def test_fill_columns_ignored(self):
        # Row 0 is biallelic (one real AC, then INT_FILL padding); row 1 is
        # triallelic. A threshold on AC must not see the fill column.
        data = {
            "variant_AC": [
                [10, constants.INT_FILL],
                [3, 7],
            ]
        }
        nt.assert_array_equal(self._evaluate("AC>5", data), [True, True])
        nt.assert_array_equal(self._evaluate("AC>8", data), [True, False])

    def test_missing_excluded(self):
        # INT_MISSING (-1) must not satisfy an ordering comparison even though
        # the raw sentinel is < the threshold.
        data = {"variant_AC": [[constants.INT_MISSING], [4]]}
        nt.assert_array_equal(self._evaluate("AC>=0", data), [False, True])
        # bcftools `!=` is True against a present value on the missing side.
        nt.assert_array_equal(self._evaluate("AC!=4", data), [True, False])


class TestEvaluationErrors:
    def _filter(self, expression, fields):
        return filter_mod.BcftoolsFilter(utils.FilterReader(fields), include=expression)

    def test_per_allele_vs_per_sample_comparison_rejected(self):
        data = {"variant_AC": [[1, 2]], "call_DP": [[3, 4, 5]]}
        fee = self._filter("INFO/AC > FMT/DP", data.keys())
        with pytest.raises(ValueError, match="per-allele.*per-sample"):
            fee.evaluate(numpify_values(data))

    def test_parse_error(self):
        with pytest.raises(filter_mod.ParseError, match="parse error"):
            self._filter("POS <", {"variant_position"})

    def test_no_op_filter_passes_everything(self):
        fee = filter_mod.BcftoolsFilter(utils.FilterReader({"variant_position"}))
        data = {"variant_position": np.array([1, 2, 3])}
        nt.assert_array_equal(fee.evaluate(data), [True, True, True])

    def test_logical_result_compared(self):
        # A logical sub-expression used as a comparison operand exercises
        # BinaryOperator.missing on a logical node (a boolean mask is never
        # missing).
        data = {"call_DP": [[6, 2], [1, 1]], "call_GQ": [[20, 0], [0, 0]]}
        fee = self._filter("(FMT/DP>5 & FMT/GQ>10) == 1", data.keys())
        result = fee.evaluate(numpify_values(data))
        nt.assert_array_equal(result, [[True, False], [False, False]])


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
            filter_mod.BcftoolsFilter(utils.FilterReader(()), include="x", exclude="y")

    def test_undefined_filter_name(self):
        flt = filter_mod.BcftoolsFilter(
            utils.FilterReader({"filter_id", "variant_filter"}),
            include='FILTER="NOPE"',
        )
        data = {
            "filter_id": np.array(["PASS", "q10"]),
            "variant_filter": np.array([[True, False], [False, True]]),
        }
        with pytest.raises(ValueError, match='The filter "NOPE" is not present'):
            flt.evaluate(data)
