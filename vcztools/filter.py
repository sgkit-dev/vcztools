import functools
import logging
import operator

import numpy as np
import pyparsing as pp

from .utils import vcf_name_to_vcz_name

logger = logging.getLogger(__name__)

# Parsing is WAY slower without this!
pp.ParserElement.enablePackrat()


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


class Unsupported2DFieldsError(UnsupportedFilteringFeatureError):
    issue = "193"
    feature = "2D INFO fields"


# The parser and evaluation model here are based on the eval_arith example
# in the pyparsing docs:
# https://github.com/pyparsing/pyparsing/blob/master/examples/eval_arith.py


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
        self.field_name = mapper(token)
        logger.debug(f"Mapped {token} to {self.field_name}")

    def eval(self, data):
        value = np.asarray(data[self.field_name])
        if (
            not self.field_name.startswith("call_")
            and self.field_name != "variant_filter"
            and len(value.shape) > 1
        ):
            raise Unsupported2DFieldsError()
        return value

    def __repr__(self):
        return self.field_name

    def referenced_fields(self):
        return frozenset([self.field_name])


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
        return -1 * self.operand.eval(data)

    def __repr__(self):
        return f"-({repr(self.operand)})"

    def referenced_fields(self):
        return self.operand.referenced_fields()


def double_and(a, b):
    # if both operands are 1D, then they are just variant masks
    if a.ndim == 1 and b.ndim == 1:
        return np.logical_and(a, b)

    # if either operand is 1D and the other is 2D, then make both 2D
    if a.ndim == 1 and b.ndim == 2:
        a = np.expand_dims(a, axis=1)
    elif a.ndim == 2 and b.ndim == 1:
        b = np.expand_dims(b, axis=1)

    if a.ndim == 2 and b.ndim == 2:
        # a variant site is included only if both conditions are met
        # but not necessarily in the same sample
        variant_mask = np.logical_and(np.any(a, axis=1), np.any(b, axis=1))
        variant_mask = np.expand_dims(variant_mask, axis=1)
        # a sample is included if either condition is met
        sample_mask = np.logical_or(a, b)
        # but if a variant site is not included then none of its samples should be
        return np.logical_and(variant_mask, sample_mask)
    else:
        raise NotImplementedError(
            f"&& not implemented for dimensions {a.ndim} and {b.ndim}"
        )


def double_or(a, b):
    # if both operands are 1D, then they are just variant masks
    if a.ndim == 1 and b.ndim == 1:
        return np.logical_or(a, b)

    # if either operand is 1D and the other is 2D, then make both 2D
    if a.ndim == 1 and b.ndim == 2:
        a = np.expand_dims(a, axis=1)
    elif a.ndim == 2 and b.ndim == 1:
        b = np.expand_dims(b, axis=1)

    if a.ndim == 2 and b.ndim == 2:
        # a variant site is included if either condition is met in any sample
        variant_mask = np.logical_or(np.any(a, axis=1), np.any(b, axis=1))
        variant_mask = np.expand_dims(variant_mask, axis=1)
        # a sample is included if either condition is met
        sample_mask = np.logical_or(a, b)
        # but if a variant site is included then all of its samples should be
        return np.logical_or(variant_mask, sample_mask)
    else:
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
        "&": np.logical_and,
        "|": np.logical_or,
        "&&": double_and,
        "||": double_or,
    }

    def __init__(self, tokens):
        super().__init__(tokens)
        # get the  operators and operands in pairs
        self.operands = self.tokens[0::2]
        self.ops = self.tokens[1::2]

    def eval(self, data):
        # start by eval()'ing the first operand
        ret = self.operands[0].eval(data)
        for op, operand in zip(self.ops, self.operands[1:]):
            arith_fn = self.op_map[op]
            ret = arith_fn(ret, operand.eval(data))
        return ret

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
        return self.comparison_fn(self.op1.eval(data), self.op2.eval(data))

    def __repr__(self):
        return f"({repr(self.op1)}){self.op}({repr(self.op2)})"

    def referenced_fields(self):
        return self.op1.referenced_fields() | self.op2.referenced_fields()


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


def _identity(x):
    return x


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

    name_mapper = _identity
    if map_vcf_identifiers:
        name_mapper = functools.partial(vcf_name_to_vcz_name, all_fields)

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
    expr_list = pp.delimited_list(pp.Group(expr))
    lpar, rpar = map(pp.Suppress, "()")
    function = pp.common.identifier() + lpar - pp.Group(expr_list) + rpar
    function = function.set_parse_action(Function)

    comp_op = pp.oneOf("< = == > >= <= !=")
    filter_expression = pp.infix_notation(
        chrom_field_expr
        | filter_field_expr
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


class FilterExpression:
    def __init__(self, *, field_names=None, include=None, exclude=None):
        if field_names is None:
            field_names = set()
        self.parse_result = None
        self.referenced_fields = set()
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

    def evaluate(self, chunk_data):
        if self.parse_result is None:
            num_variants = len(next(iter(chunk_data.values())))
            return np.ones(num_variants, dtype=bool)

        result = self.parse_result[0].eval(chunk_data)
        if self.invert:
            result = np.logical_not(result)
        return result
