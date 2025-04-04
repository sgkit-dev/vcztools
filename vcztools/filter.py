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


class UnsupportedRegexError(UnsupportedFilteringFeatureError):
    issue = "174"
    feature = "Regular expressions"


class UnsupportedArraySubscriptError(UnsupportedFilteringFeatureError):
    issue = "167"
    feature = "Array subscripts"


class UnsupportedStringsError(UnsupportedFilteringFeatureError):
    issue = "189"
    feature = "String values temporarily removed"


class UnsupportedFileReferenceError(UnsupportedFilteringFeatureError):
    issue = "175"
    feature = "File references"


class UnsupportedSampleFilteringError(UnsupportedFilteringFeatureError):
    issue = "180"
    feature = "Per-sample filter expressions"


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
        raise UnsupportedStringsError()


class FileReference(Constant):
    def __init__(self, tokens):
        raise UnsupportedFileReferenceError()


class Function(EvaluationNode):
    def __init__(self, tokens):
        raise UnsupportedFunctionsError()


class Identifier(EvaluationNode):
    def __init__(self, mapper, tokens):
        self.field_name = mapper(tokens[0])
        if self.field_name.startswith("call_"):
            raise UnsupportedSampleFilteringError()
        logger.debug(f"Mapped {tokens[0]} to {self.field_name}")

    def eval(self, data):
        value = np.asarray(data[self.field_name])
        if len(value.shape) > 1:
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
# when doing things like referenced_fields. We should probably
# be extracting the operators and operands once.


class UnaryMinus(EvaluationNode):
    def eval(self, data):
        op, operand = self.tokens
        assert op == "-"
        return -1 * operand.eval(data)

    def __repr__(self):
        _, operand = self.tokens
        return f"-({repr(operand)})"

    def referenced_fields(self):
        _, operand = self.tokens
        return operand.referenced_fields()


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
        # As we're only supporting 1D values for now, these are the same thing
        "&&": np.logical_and,
        "||": np.logical_or,
    }

    def eval(self, data):
        # get the  operators and operands in pairs
        operands = self.tokens[0::2]
        ops = self.tokens[1::2]
        # start by eval()'ing the first operand
        ret = operands[0].eval(data)
        for op, operand in zip(ops, operands[1:]):
            arith_fn = self.op_map[op]
            ret = arith_fn(ret, operand.eval(data))
        return ret

    def __repr__(self):
        ops = self.tokens[1::2]
        operands = self.tokens[0::2]
        ret = f"({repr(operands[0])})"
        for op, operand in zip(ops, operands[1:]):
            ret += f"{op}({repr(operand)})"
        return ret

    def referenced_fields(self):
        operands = self.tokens[0::2]
        ret = operands[0].referenced_fields()
        for operand in operands[1:]:
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

    def eval(self, data):
        op1, op, op2 = self.tokens
        comparison_fn = self.op_map[op]
        return comparison_fn(op1.eval(data), op2.eval(data))

    def __repr__(self):
        op1, op, op2 = self.tokens
        return f"({repr(op1)}){op}({repr(op2)})"

    def referenced_fields(self):
        op1, _, op2 = self.tokens
        return op1.referenced_fields() | op2.referenced_fields()


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

    name_mapper = _identity
    if map_vcf_identifiers:
        name_mapper = functools.partial(vcf_name_to_vcz_name, all_fields)
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
        function | constant | indexed_identifier | identifier | file_expr,
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
