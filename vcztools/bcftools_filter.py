import functools
import logging
import operator

import numpy as np
import pyparsing as pp

from vcztools.calculate import SNP, calculate_variant_type

from . import constants
from .utils import vcf_name_to_vcz_names

logger = logging.getLogger(__name__)

# Parsing is WAY slower without this!
pp.ParserElement.enable_packrat()


class ParseError(ValueError):
    def __init__(self, msg):
        super().__init__(f"Filter expression parse error: {msg}")


class UnsupportedFilteringFeatureError(ValueError):
    def __init__(self):
        super().__init__(
            f"Unsupported filtering feature: {self.feature}. Please see "
            f"https://github.com/sgkit-dev/vcztools/issues/{self.issue} "
            "for details and let us know if this is important to you."
        )


class UnsupportedMissingDataError(UnsupportedFilteringFeatureError):
    issue = "163"
    feature = "Missing data"


class UnsupportedGenotypeValuesError(UnsupportedFilteringFeatureError):
    issue = "165"
    feature = "Genotype values"


class UnsupportedTypeFieldError(UnsupportedFilteringFeatureError):
    issue = "166"
    feature = "TYPE field (except 'ref' and 'snp')"


class UnsupportedArraySubscriptError(UnsupportedFilteringFeatureError):
    issue = "167"
    feature = "Array subscripts"


class UnsupportedRegexError(UnsupportedFilteringFeatureError):
    issue = "174"
    feature = "Regular expressions"


class UnsupportedFileReferenceError(UnsupportedFilteringFeatureError):
    issue = "175"
    feature = "File references"


class UnsupportedFunctionsError(UnsupportedFilteringFeatureError):
    issue = "190"
    feature = "Function evaluation"


class UnsupportedHigherDimensionalFormatFieldsError(UnsupportedFilteringFeatureError):
    issue = "232"
    feature = "Higher dimensional FORMAT fields"


class UnsupportedCalculatedVariableError(UnsupportedFilteringFeatureError):
    issue = "171"
    feature = "Calculated variables (MAC, MAF, ILEN, N_SAMPLES)"


# bcftools calculated variables we recognise but do not yet implement.
# Intercepted in Identifier.__init__ so the user sees a dedicated error
# instead of the generic "the tag X is not defined". AC, AN, AF, NS,
# N_ALT, N_MISSING and F_MISSING are surfaced by the VczReader's
# virtual-field registry (see ``vcztools/virtual_fields.py``); the
# generic Identifier path picks them up uniformly via the
# ``virtual_field_names`` set that callers fold into ``all_fields``.
UNSUPPORTED_CALCULATED_VARIABLES = frozenset({"N_SAMPLES", "MAC", "MAF", "ILEN"})


# The parser and evaluation model here are based on the eval_arith example
# in the pyparsing docs:
# https://github.com/pyparsing/pyparsing/blob/master/examples/eval_arith.py


def _missing_mask(value):
    # Unlike ``utils.is_missing`` we also mask INT_FILL (trailing padding in
    # Number=A arrays) and use np.isnan for floats so that -value still
    # registers as missing (the bit-pattern check would miss sign-flipped NaN).
    if value.dtype.kind == "i":
        return (value == constants.INT_MISSING) | (value == constants.INT_FILL)
    elif value.dtype.kind == "f":
        return np.isnan(value)
    return False


class EvaluationNode:
    """
    Base class for all of the parsed nodes in the expression
    evaluation tree.
    """

    def __init__(self, tokens):
        self.tokens = tokens[0]

    def __repr__(self):
        return repr(self.tokens)

    def referenced_fields(self):
        return frozenset()

    def scope(self):
        # A 2-D mask with sample-axis semantics (axis 1 = samples) is
        # produced only by identifiers on ``call_*`` fields; every other
        # node is variant-scoped (axis 1, if present, is alleles or
        # filters and collapses to a 1-D variant mask at the root).
        return "variant"

    def missing(self, data):
        # Per-element mask of "this slot is a missing/fill sentinel".
        # Propagates through arithmetic so that ComparisonOperator can
        # force the comparison result to False wherever either operand
        # was missing. Default False means "never missing".
        return False


class Constant(EvaluationNode):
    def eval(self, data):
        return self.tokens


class Number(Constant):
    pass


class String(Constant):
    def __init__(self, tokens):
        super().__init__(tokens)
        if self.tokens == ".":
            raise UnsupportedMissingDataError()


class FileReference(Constant):
    def __init__(self, tokens):
        raise UnsupportedFileReferenceError()


class Function(EvaluationNode):
    def __init__(self, tokens):
        raise UnsupportedFunctionsError()


class Identifier(EvaluationNode):
    def __init__(self, mapper, tokens):
        token = tokens[0]
        if token == "GT":
            raise UnsupportedGenotypeValuesError()
        field_names = mapper(token)
        if len(field_names) == 0:
            # Known bcftools calculated variables we haven't implemented
            # yet — surface a dedicated error rather than the generic
            # "the tag X is not defined". An identifier like ``AC`` that
            # IS present as a real INFO field uses that real field
            # (matches bcftools), so the check only fires when the tag
            # is otherwise undefined.
            if token in UNSUPPORTED_CALCULATED_VARIABLES:
                raise UnsupportedCalculatedVariableError()
            raise ValueError(f'the tag "{token}" is not defined')
        elif len(field_names) == 1:
            self.field_name = field_names[0]
            logger.debug(f"Mapped {token} to {self.field_name}")
        else:
            raise ValueError(
                f'ambiguous filtering expression: "{token}", '
                f"both INFO/{token} and FORMAT/{token} are defined"
            )

    def eval(self, data):
        value = np.asarray(data[self.field_name])
        if self.field_name.startswith("call_") and len(value.shape) > 2:
            raise UnsupportedHigherDimensionalFormatFieldsError()
        return value

    def missing(self, data):
        return _missing_mask(np.asarray(data[self.field_name]))

    def __repr__(self):
        return self.field_name

    def referenced_fields(self):
        return frozenset([self.field_name])

    def scope(self):
        return "sample" if self.field_name.startswith("call_") else "variant"


class IndexedIdentifier(EvaluationNode):
    def __init__(self, tokens):
        # The tokens here are the already resolved idenfitier
        # and the index
        raise UnsupportedArraySubscriptError()


class RegexOperator(EvaluationNode):
    def __init__(self, tokens):
        raise UnsupportedRegexError()


# NOTE we should perhaps add a Operator superclass of UnaryMinus,
# BinaryOperator and ComparisonOperator to reduce duplication
# when doing things like referenced_fields.


class UnaryMinus(EvaluationNode):
    def __init__(self, tokens):
        super().__init__(tokens)
        self.op, self.operand = self.tokens
        assert self.op == "-"

    def eval(self, data):
        # Use numpy unary minus rather than ``-1 * value``: the latter emits
        # "invalid value encountered in multiply" when value contains NaN
        # (e.g. missing QUAL). Sign-flip preserves NaN-ness silently.
        return -self.operand.eval(data)

    def missing(self, data):
        return self.operand.missing(data)

    def __repr__(self):
        return f"-({repr(self.operand)})"

    def referenced_fields(self):
        return self.operand.referenced_fields()

    def scope(self):
        return self.operand.scope()


def _align_dims(a, b):
    # Expand a 1-D per-variant operand to (n, 1) when the other operand is 2-D
    # (axis 1 = alleles or samples), so the two broadcast together. This covers
    # both boolean masks and raw field values, e.g. comparing per-allele
    # INFO/AC (Number=A) against per-variant INFO/AN. Scalars and both-1-D /
    # both-2-D operands are returned unchanged.
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim == 1 and b.ndim == 2:
        a = np.expand_dims(a, axis=1)
    elif a.ndim == 2 and b.ndim == 1:
        b = np.expand_dims(b, axis=1)
    return a, b


def _collapse_to_variant(value):
    # A variant-scope 2-D mask carries axis 1 = alleles; collapse it to a
    # per-variant decision (the variant matches if any allele does). A 1-D
    # mask is already per-variant and returned unchanged.
    if value.ndim == 2:
        return np.any(value, axis=1)
    return value


def single_and(a, b):
    a, b = _align_dims(a, b)
    return np.logical_and(a, b)


def single_or(a, b):
    a, b = _align_dims(a, b)
    return np.logical_or(a, b)


def double_and(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    # A scalar operand (e.g. a bare constant) broadcasts against either shape,
    # and two variant masks combine element-wise.
    if a.ndim == 0 or b.ndim == 0 or (a.ndim == 1 and b.ndim == 1):
        return np.logical_and(a, b)

    a, b = _align_dims(a, b)

    if a.ndim == 2 and b.ndim == 2:
        # a variant site is included only if both conditions are met
        # but not necessarily in the same sample
        variant_mask = np.logical_and(np.any(a, axis=1), np.any(b, axis=1))
        variant_mask = np.expand_dims(variant_mask, axis=1)
        # a sample is included if either condition is met
        sample_mask = np.logical_or(a, b)
        # but if a variant site is not included then none of its samples should be
        return np.logical_and(variant_mask, sample_mask)
    else:  # pragma: no cover
        # Unreachable: scalars are handled above and call_* fields beyond 2-D
        # are rejected at eval, so _align_dims always yields both-2-D here.
        raise NotImplementedError(
            f"&& not implemented for dimensions {a.ndim} and {b.ndim}"
        )


def double_or(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim == 0 or b.ndim == 0 or (a.ndim == 1 and b.ndim == 1):
        return np.logical_or(a, b)

    a, b = _align_dims(a, b)

    if a.ndim == 2 and b.ndim == 2:
        # a variant site is included if either condition is met in any sample
        variant_mask = np.logical_or(np.any(a, axis=1), np.any(b, axis=1))
        variant_mask = np.expand_dims(variant_mask, axis=1)
        # a sample is included if either condition is met
        sample_mask = np.logical_or(a, b)
        # but if a variant site is included then all of its samples should be
        return np.logical_or(variant_mask, sample_mask)
    else:  # pragma: no cover
        # Unreachable: see the matching note in double_and.
        raise NotImplementedError(
            f"|| not implemented for dimensions {a.ndim} and {b.ndim}"
        )


class BinaryOperator(EvaluationNode):
    op_map = {
        "*": operator.mul,
        "/": operator.truediv,
        "+": operator.add,
        "-": operator.sub,
        # Note that by lumping logical operators in here we forgo any short
        # circuit optimisations
        "&": single_and,
        "|": single_or,
        "&&": double_and,
        "||": double_or,
    }
    _ARITHMETIC_OPS = frozenset({"*", "/", "+", "-"})

    def __init__(self, tokens):
        super().__init__(tokens)
        # get the  operators and operands in pairs
        self.operands = self.tokens[0::2]
        self.ops = self.tokens[1::2]

    def eval(self, data):
        values = [operand.eval(data) for operand in self.operands]
        if self.ops[0] not in self._ARITHMETIC_OPS:
            values = self._prepare_logical_operands(values)
        ret = values[0]
        for op, rhs in zip(self.ops, values[1:]):
            arith_fn = self.op_map[op]
            # Arithmetic must broadcast a per-variant (1-D) operand against a
            # per-allele (2-D) one, e.g. INFO/AC / INFO/AN. The logical ops
            # align inside single_and / double_and / etc.
            if op in self._ARITHMETIC_OPS:
                ret, rhs = _align_dims(ret, rhs)
            # Suppress "invalid value encountered" warnings from arithmetic
            # on NaN-encoded missing floats (e.g. QUAL=.).
            with np.errstate(invalid="ignore"):
                ret = arith_fn(ret, rhs)
        return ret

    def _prepare_logical_operands(self, values):
        # When the expression mixes a per-sample (sample-scope) operand with a
        # per-allele (variant-scope, 2-D) one, collapse the allele axis to a
        # per-variant decision so the masks combine: a per-allele mask carries
        # axis 1 = alleles while a per-sample mask carries axis 1 = samples.
        # All-variant logical expressions keep their per-allele axis (so
        # ``AC>0 & AC<2`` stays a same-allele test) and pass through unchanged.
        if self.scope() != "sample":
            return values
        prepared = []
        for operand, value in zip(self.operands, values):
            if operand.scope() == "variant":
                value = _collapse_to_variant(np.asarray(value))
            prepared.append(value)
        return prepared

    def missing(self, data):
        # Logical ops operate on boolean results that ComparisonOperator
        # has already masked, so there's nothing to propagate.
        if self.ops[0] not in self._ARITHMETIC_OPS:
            return False
        result = False
        for operand in self.operands:
            result, mask = _align_dims(result, operand.missing(data))
            result = np.logical_or(result, mask)
        return result

    def __repr__(self):
        ret = f"({repr(self.operands[0])})"
        for op, operand in zip(self.ops, self.operands[1:]):
            ret += f"{op}({repr(operand)})"
        return ret

    def referenced_fields(self):
        ret = self.operands[0].referenced_fields()
        for operand in self.operands[1:]:
            ret |= operand.referenced_fields()
        return ret

    def scope(self):
        for operand in self.operands:
            if operand.scope() == "sample":
                return "sample"
        return "variant"


class ComparisonOperator(EvaluationNode):
    op_map = {
        "=": operator.eq,
        "==": operator.eq,
        "<": operator.lt,
        ">": operator.gt,
        "!=": operator.ne,
        ">=": operator.ge,
        "<=": operator.le,
    }

    def __init__(self, tokens):
        super().__init__(tokens)
        self.op1, self.op, self.op2 = self.tokens
        self.comparison_fn = self.op_map[self.op]

    def eval(self, data):
        raw1 = np.asarray(self.op1.eval(data))
        raw2 = np.asarray(self.op2.eval(data))
        # A 2-D per-allele (variant-scope) operand and a 2-D per-sample
        # (sample-scope) operand have incompatible axis-1 meanings (alleles vs
        # samples), so element-wise comparison is undefined.
        if raw1.ndim == 2 and raw2.ndim == 2 and self.op1.scope() != self.op2.scope():
            raise ValueError(
                "cannot compare a per-allele (INFO Number=A) field with a "
                "per-sample (FORMAT) field"
            )
        # Align so a per-variant (1-D) operand broadcasts against a per-allele
        # (2-D) one, e.g. INFO/AC < INFO/AN. The root collapses the resulting
        # 2-D variant mask via np.any.
        v1, v2 = _align_dims(raw1, raw2)
        # Comparing sentinel-encoded missing values (e.g. INFO/AC=-1) as if
        # they were real data gives wrong answers (-1 satisfies "<2"), so the
        # comparison result is forced at missing positions to match bcftools.
        # errstate silences NaN comparison warnings on float missing.
        with np.errstate(invalid="ignore"):
            result = self.comparison_fn(v1, v2)
        m1, m2 = _align_dims(self.op1.missing(data), self.op2.missing(data))
        any_missing = np.logical_or(m1, m2)
        both_missing = np.logical_and(m1, m2)
        # bcftools treats two missing values as equal: `==` is True and `!=`
        # is False where both operands are absent, a present value never
        # equals a missing one, and ordering comparisons are False whenever
        # either operand is missing. For a tag against a (never-missing)
        # constant this reduces to the exactly-one-missing column.
        if self.comparison_fn is operator.eq:
            result = np.where(both_missing, True, np.where(any_missing, False, result))
        elif self.comparison_fn is operator.ne:
            result = np.where(both_missing, False, np.where(any_missing, True, result))
        else:
            result = np.where(any_missing, False, result)
        return result

    def __repr__(self):
        return f"({repr(self.op1)}){self.op}({repr(self.op2)})"

    def referenced_fields(self):
        return self.op1.referenced_fields() | self.op2.referenced_fields()

    def scope(self):
        if self.op1.scope() == "sample" or self.op2.scope() == "sample":
            return "sample"
        return "variant"


# CHROM field expressions are translated to contig IDs to avoid string
# comparisons for every variant site


class ChromString(Constant):
    def __init__(self, tokens):
        super().__init__(tokens)

    def eval(self, data):
        contig_ids = list(data["contig_id"])
        try:
            return contig_ids.index(self.tokens)
        except ValueError:
            return -1  # won't match anything

    def referenced_fields(self):
        return frozenset(["contig_id"])


class ChromFieldOperator(EvaluationNode):
    op_map = {
        "=": operator.eq,
        "==": operator.eq,
        "!=": operator.ne,
    }

    def __init__(self, tokens):
        super().__init__(tokens)
        self.op1, self.op, self.op2 = tokens  # not self.tokens
        self.comparison_fn = self.op_map[self.op]

    def eval(self, data):
        return self.comparison_fn(self.op1.eval(data), self.op2.eval(data))

    def __repr__(self):
        return f"({repr(self.op1)}){self.op}({repr(self.op2)})"

    def referenced_fields(self):
        return self.op1.referenced_fields() | self.op2.referenced_fields()


# FILTER field expressions have special set-like semantics
# so they are handled by dedicated operators.


class FilterString(Constant):
    def __init__(self, tokens):
        super().__init__(tokens)

    def eval(self, data):
        # convert string to a 1D boolean array (one element per filter)
        if self.tokens == ".":
            return np.zeros_like(data["filter_id"], dtype=bool)
        filters = self.tokens.split(";")
        for filter in filters:
            if filter not in data["filter_id"]:
                raise ValueError(f'The filter "{filter}" is not present in header')
        return np.isin(data["filter_id"], filters)

    def referenced_fields(self):
        return frozenset(["filter_id"])


# 'a' is a 2D boolean array with shape (variants, filters)
# 'b' is a 1D boolean array with shape (filters)


def filter_eq(a, b):
    return np.all(a == b, axis=1)


def filter_ne(a, b):
    return ~filter_eq(a, b)


def filter_subset_match(a, b):
    return np.all(a[:, b], axis=1)


def filter_complement_match(a, b):
    return ~filter_subset_match(a, b)


class FilterFieldOperator(EvaluationNode):
    op_map = {
        "=": filter_eq,
        "==": filter_eq,
        "!=": filter_ne,
        "~": filter_subset_match,
        "!~": filter_complement_match,
    }

    def __init__(self, tokens):
        super().__init__(tokens)
        self.op1, self.op, self.op2 = tokens  # not self.tokens
        self.comparison_fn = self.op_map[self.op]

    def eval(self, data):
        return self.comparison_fn(self.op1.eval(data), self.op2.eval(data))

    def __repr__(self):
        return f"({repr(self.op1)}){self.op}({repr(self.op2)})"

    def referenced_fields(self):
        return self.op1.referenced_fields() | self.op2.referenced_fields()


def type_eq(a, b):
    if b == "ref":
        return np.all(a < 0, axis=1)
    elif b == "snp":
        all_a = np.bitwise_and.reduce(a, axis=1)
        all_a = np.where(all_a < 0, 0, all_a)  # remove missing
        return np.bitwise_and(all_a, SNP) == SNP
    else:  # pragma: no cover
        # Unreachable: TypeOperator.__init__ rejects any type outside ref/snp.
        raise NotImplementedError(f"TYPE comparison not implemented for '{b}'")


def type_ne(a, b):
    return ~type_eq(a, b)


def type_subset_match(a, b):
    if b == "ref":
        return np.all(a < 0, axis=1)
    elif b == "snp":
        any_a = np.where(a < 0, 0, a)  # remove missing
        any_a = np.bitwise_or.reduce(any_a, axis=1)
        return np.bitwise_and(any_a, SNP) == SNP
    else:  # pragma: no cover
        # Unreachable: TypeOperator.__init__ rejects any type outside ref/snp.
        raise NotImplementedError(f"TYPE comparison not implemented for '{b}'")


def type_complement_match(a, b):
    return ~type_subset_match(a, b)


class TypeIdentifier(EvaluationNode):
    def eval(self, data):
        variant_allele = np.asarray(data["variant_allele"])
        variant_type = calculate_variant_type(variant_allele)
        return variant_type

    def __repr__(self):
        return "variant_allele"

    def referenced_fields(self):
        return frozenset(["variant_allele"])


class TypeOperator(EvaluationNode):
    op_map = {
        "=": type_eq,
        "==": type_eq,
        "!=": type_ne,
        "~": type_subset_match,
        "!~": type_complement_match,
    }

    def __init__(self, tokens):
        super().__init__(tokens)
        self.op1, self.op, self.op2 = tokens
        if self.op2 not in ("ref", "snp"):
            raise UnsupportedTypeFieldError()
        self.comparison_fn = self.op_map[self.op]

    def eval(self, data):
        return self.comparison_fn(self.op1.eval(data), self.op2)

    def __repr__(self):
        return f"({repr(self.op1)}){self.op}({repr(self.op2)})"

    def referenced_fields(self):
        return self.op1.referenced_fields()


def _identity_list(x):
    return [x]


def make_bcftools_filter_parser(all_fields=None, map_vcf_identifiers=True):
    if all_fields is None:
        all_fields = set()

    number = pp.common.number.set_parse_action(Number)
    string = pp.QuotedString('"').set_parse_action(String)
    constant = number | string

    file_expr = (pp.Literal("@") + pp.Word(pp.printables)).set_parse_action(
        FileReference
    )

    identifier = pp.common.identifier()
    vcf_prefixes = pp.Literal("INFO/") | pp.Literal("FORMAT/") | pp.Literal("FMT/")
    vcf_identifier = pp.Combine(vcf_prefixes + identifier) | identifier

    name_mapper = _identity_list
    if map_vcf_identifiers:
        name_mapper = functools.partial(vcf_name_to_vcz_names, all_fields)

    chrom_field_identifier = pp.Literal("CHROM")
    chrom_field_identifier = chrom_field_identifier.set_parse_action(
        functools.partial(Identifier, name_mapper)
    )
    chrom_string = pp.QuotedString('"').set_parse_action(ChromString)
    chrom_field_expr = chrom_field_identifier + pp.one_of("= == !=") + chrom_string
    chrom_field_expr = chrom_field_expr.set_parse_action(ChromFieldOperator)

    filter_field_identifier = pp.Literal("FILTER")
    filter_field_identifier = filter_field_identifier.set_parse_action(
        functools.partial(Identifier, name_mapper)
    )
    filter_string = pp.QuotedString('"').set_parse_action(FilterString)
    filter_field_expr = filter_field_identifier + pp.one_of("= != ~ !~") + filter_string
    filter_field_expr = filter_field_expr.set_parse_action(FilterFieldOperator)

    type_identifier = pp.Literal("TYPE")
    type_identifier = type_identifier.set_parse_action(TypeIdentifier)
    type_string = pp.QuotedString('"')
    type_expr = type_identifier + pp.one_of("= == != ~ !~") + type_string
    type_expr = type_expr.set_parse_action(TypeOperator)

    lbracket, rbracket = map(pp.Suppress, "[]")
    # TODO we need to define the indexing grammar more carefully, but
    # this at least let's us match correct strings and raise an informative
    # error
    index_expr = pp.OneOrMore(
        pp.common.number
        | pp.Literal("*")
        | pp.Literal(":")
        | pp.Literal("-")
        | pp.Literal(",")
    )
    indexed_identifier = pp.Group(vcf_identifier + (lbracket + index_expr + rbracket))

    identifier = vcf_identifier.set_parse_action(
        functools.partial(Identifier, name_mapper)
    )
    indexed_identifier = indexed_identifier.set_parse_action(IndexedIdentifier)

    expr = pp.Forward()
    expr_list = pp.DelimitedList(pp.Group(expr))
    lpar, rpar = map(pp.Suppress, "()")
    function = pp.common.identifier() + lpar - pp.Group(expr_list) + rpar
    function = function.set_parse_action(Function)

    comp_op = pp.one_of("< = == > >= <= !=")
    filter_expression = pp.infix_notation(
        chrom_field_expr
        | filter_field_expr
        | type_expr
        | function
        | constant
        | indexed_identifier
        | identifier
        | file_expr,
        [
            ("-", 1, pp.OpAssoc.RIGHT, UnaryMinus),
            (pp.one_of("* /"), 2, pp.OpAssoc.LEFT, BinaryOperator),
            (pp.one_of("+ -"), 2, pp.OpAssoc.LEFT, BinaryOperator),
            (comp_op, 2, pp.OpAssoc.LEFT, ComparisonOperator),
            (pp.Keyword("&"), 2, pp.OpAssoc.LEFT, BinaryOperator),
            (pp.Keyword("&&"), 2, pp.OpAssoc.LEFT, BinaryOperator),
            (pp.Keyword("|"), 2, pp.OpAssoc.LEFT, BinaryOperator),
            (pp.Keyword("||"), 2, pp.OpAssoc.LEFT, BinaryOperator),
            # NOTE Putting the Regex operator at the end for now as
            # I haven't figured out what the actual precedence is.
            (pp.one_of("~ !~"), 2, pp.OpAssoc.LEFT, RegexOperator),
        ],
    )
    expr <<= filter_expression
    return expr


class BcftoolsFilter:
    """Bcftools ``-i``/``-e`` expression compiled into a
    :class:`~vcztools.VariantFilter`.

    Accepts the same expression syntax as ``bcftools view -i`` /
    ``-e``. Mutually exclusive: only one of ``include`` or ``exclude``
    may be non-``None``. When neither is set the filter is a no-op;
    callers should skip the filter in that case rather than evaluate it.

    The first argument is the target :class:`~vcztools.VczReader`. Its
    ``field_names`` and ``virtual_field_names`` form the resolution
    surface the parser uses to map bare VCF names like ``DP`` to their
    VCZ equivalents (``call_DP`` vs ``variant_DP``). Both expression
    parsing and field-name resolution happen at instantiation time.
    """

    def __init__(self, reader, *, include=None, exclude=None):
        field_names = reader.field_names | reader.virtual_field_names
        self.parse_result = None
        self.referenced_fields = set()
        self.scope = "variant"
        self.invert = False
        expr = None
        if include is not None and exclude is not None:
            raise ValueError(
                "Cannot handle both an include expression and an exclude expression."
            )
        if include is not None:
            expr = include
            self.invert = False
        elif exclude is not None:
            expr = exclude
            self.invert = True

        if expr is not None:
            parser = make_bcftools_filter_parser(field_names)
            try:
                self.parse_result = parser.parse_string(expr, parse_all=True)
            except pp.ParseException as e:
                raise ParseError(e) from None
            # This isn't a very good pattern, fix
            self.referenced_fields = self.parse_result[0].referenced_fields()
            self.scope = self.parse_result[0].scope()

    def evaluate(self, chunk_data):
        if self.parse_result is None:
            num_variants = len(next(iter(chunk_data.values())))
            return np.ones(num_variants, dtype=bool)

        result = self.parse_result[0].eval(chunk_data)
        if self.invert:
            result = np.logical_not(result)
        # Variant-scoped 2-D results (axis 1 = alleles, from fields like
        # INFO/AC with Number=A) collapse to a 1-D variant mask so that
        # downstream consumers can rely on a 2-D result always meaning
        # axis 1 = samples.
        if self.scope == "variant" and result.ndim == 2:
            result = np.any(result, axis=1)
        return result
