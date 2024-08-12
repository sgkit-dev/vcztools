import functools

import numpy as np
import pyparsing as pp


def search(a, v):
    """Like `np.searchsorted`, but array `a` does not have to be sorted."""
    sorter = np.argsort(a)
    rank = np.searchsorted(a, v, sorter=sorter)
    return sorter[rank]


class FilterExpressionParser:
    def __init__(self):
        constant_pattern = pp.common.number | pp.QuotedString('"')
        standard_tag_pattern = pp.Word(pp.srange("[A-Z]"))

        info_tag_pattern = pp.Combine(pp.Literal("INFO/") + standard_tag_pattern)
        format_tag_pattern = pp.Combine(
            (pp.Literal("FORMAT/") | pp.Literal("FMT/")) + standard_tag_pattern
        )
        tag_pattern = info_tag_pattern | format_tag_pattern | standard_tag_pattern

        identifier_pattern = tag_pattern | constant_pattern
        self._identifier_parser = functools.partial(
            identifier_pattern.parse_string, parse_all=True
        )

        comparison_pattern = pp.Group(
            tag_pattern + pp.one_of("== = != > >= < <=") + constant_pattern
        ).set_results_name("comparison")

        parentheses_pattern = pp.Forward()
        and_pattern = pp.Forward()
        or_pattern = pp.Forward()

        parentheses_pattern <<= (
            pp.Suppress("(") + or_pattern + pp.Suppress(")") | comparison_pattern
        )
        and_pattern <<= (
            pp.Group(
                parentheses_pattern + pp.Keyword("&&") + and_pattern
            ).set_results_name("and")
            | parentheses_pattern
        )
        or_pattern <<= (
            pp.Group(and_pattern + pp.Keyword("||") + or_pattern).set_results_name("or")
            | and_pattern
        )

        self._parser = functools.partial(or_pattern.parse_string, parse_all=True)

    def __call__(self, *args, **kwargs):
        assert args or kwargs

        if args:
            assert len(args) == 1
            assert not kwargs
            expression = args[0]
        else:
            assert len(kwargs) == 1
            assert "expression" in kwargs
            expression = kwargs["expression"]

        return self.parse(expression)

    def parse(self, expression: str):
        return self._parser(expression)
