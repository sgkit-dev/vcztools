import functools
import logging
import operator

import numpy as np
import pyparsing as pp

from .utils import vcf_name_to_vcz_name

logger = logging.getLogger(__name__)


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


class Constant(EvaluationNode):
    def eval(self, data):
        return self.tokens


class Identifier(EvaluationNode):
    def __init__(self, mapper, tokens):
        self.field_name = mapper(tokens[0])
        logger.debug(f"Mapped {tokens[0]} to {self.field_name}")

    def eval(self, data):
        return data[self.field_name]


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
        # start by eval()'ing the first operand
        ret = self.tokens[0].eval(data)

        # get following operators and operands in pairs
        ops = self.tokens[1::2]
        operands = self.tokens[2::2]
        for op, operand in zip(ops, operands):
            # print(f"Eval {op}, {ret}, {operand}")
            # update cumulative value by add/subtract/mult/divide the next operand
            arith_fn = self.op_map[op]
            ret = arith_fn(ret, operand.eval(data))
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


def _identity(x):
    return x


def make_bcftools_filter_parser(all_fields=None, map_vcf_identifiers=True):
    if all_fields is None:
        all_fields = set()

    constant = (pp.common.number | pp.QuotedString('"')).set_parse_action(Constant)
    identifier = pp.common.identifier()

    vcf_prefixes = pp.Literal("INFO/") | pp.Literal("FORMAT/") | pp.Literal("FMT/")
    vcf_identifier = pp.Combine(vcf_prefixes + identifier) | identifier

    name_mapper = _identity
    if map_vcf_identifiers:
        name_mapper = functools.partial(vcf_name_to_vcz_name, all_fields)
    identifier = vcf_identifier.set_parse_action(
        functools.partial(Identifier, name_mapper)
    )
    comp_op = pp.oneOf("< = == > >= <= !=")
    filter_expression = pp.infix_notation(
        constant | identifier,
        [
            # ("-", 1, pp.OpAssoc.RIGHT, ),
            (pp.one_of("* /"), 2, pp.OpAssoc.LEFT, BinaryOperator),
            (pp.one_of("+ -"), 2, pp.OpAssoc.LEFT, BinaryOperator),
            (comp_op, 2, pp.OpAssoc.LEFT, ComparisonOperator),
            (pp.Keyword("&"), 2, pp.OpAssoc.LEFT, BinaryOperator),
            (pp.Keyword("&&"), 2, pp.OpAssoc.LEFT, BinaryOperator),
            (pp.Keyword("|"), 2, pp.OpAssoc.LEFT, BinaryOperator),
            (pp.Keyword("||"), 2, pp.OpAssoc.LEFT, BinaryOperator),
        ],
    )
    return filter_expression


class FilterExpression:
    def __init__(self, *, field_names=None, include=None, exclude=None):
        if field_names is None:
            field_names = set()
        self.parse_result = None
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
            self.parse_result = parser.parse_string(expr, parse_all=True)

        # Setting to None for now so that we retrieve all fields
        self.referenced_fields = None

    def evaluate(self, chunk_data):
        if self.parse_result is None:
            num_variants = len(next(iter(chunk_data.values())))
            return np.ones(num_variants, dtype=bool)

        result = self.parse_result[0].eval(chunk_data)
        if self.invert:
            result = np.logical_not(result)
        return result
