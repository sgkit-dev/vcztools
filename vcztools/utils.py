import functools
import operator
from typing import Callable

import numpy as np
import pyparsing as pp

from vcztools.constants import RESERVED_VCF_FIELDS


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
                parentheses_pattern + (pp.Keyword("&&") | pp.Keyword("&")) + and_pattern
            ).set_results_name("and")
            | parentheses_pattern
        )
        or_pattern <<= (
            pp.Group(
                and_pattern + (pp.Keyword("||") | pp.Keyword("|")) + or_pattern
            ).set_results_name("or")
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


def vcf_to_vcz(vczs: set[str], vcf: str):
    split = vcf.split("/")
    assert 1 <= len(split) <= 2
    is_genotype_field = split[-1] == "GT"
    is_format_field = (
        split[0] in {"FORMAT", "FMT"}
        if len(split) > 1
        else is_genotype_field or f"call_{split[-1]}" in vczs
    )
    is_info_field = split[0] == "INFO" or split[-1] not in RESERVED_VCF_FIELDS

    if is_format_field:
        if split[-1] == "GT":
            return "call_genotype"
        else:
            return "call_" + split[-1]
    elif is_info_field:
        return f"variant_{split[-1]}"
    else:
        return RESERVED_VCF_FIELDS[vcf]


class FilterExpressionEvaluator:
    def __init__(self, parse_results: pp.ParseResults, *, invert=False):
        self._composers = {
            "comparison": self._compose_comparison_evaluator,
            "and": self._compose_and_evaluator,
            "or": self._compose_or_evaluator,
        }
        self._comparators = {
            "==": operator.eq,
            "=": operator.eq,
            "!=": operator.ne,
            ">": operator.gt,
            ">=": operator.ge,
            "<": operator.lt,
            "<=": operator.le,
        }
        evaluator = self._compose_evaluator(parse_results)

        if invert:

            def invert_evaluator(root, variant_chunk_index: int) -> np.ndarray:
                return np.logical_not(evaluator(root, variant_chunk_index))

            self._evaluator = invert_evaluator
        else:
            self._evaluator = evaluator

    def __call__(self, *args, **kwargs):
        assert len(args) == 2
        assert not kwargs

        return self._evaluator(*args)

    def _compose_comparison_evaluator(self, parse_results: pp.ParseResults) -> Callable:
        assert len(parse_results) == 3

        comparator = parse_results[1]
        comparator = self._comparators[comparator]

        def evaluator(root, variant_chunk_index: int) -> np.ndarray:
            vcf_name = parse_results[0]
            vcz_names = set(name for name, _array in root.items())
            vcz_name = vcf_to_vcz(vcz_names, vcf_name)
            zarray = root[vcz_name]
            variant_chunk_len = zarray.chunks[0]
            start = variant_chunk_len * variant_chunk_index
            end = start + variant_chunk_len
            array = zarray[start:end]

            if array.ndim > 1:
                axis = tuple(range(1, array.ndim))
                return np.any(comparator(array, parse_results[2]), axis=axis)
            else:
                return comparator(array, parse_results[2])

        return evaluator

    def _compose_and_evaluator(self, parse_results: pp.ParseResults) -> Callable:
        assert len(parse_results) == 3

        left_evaluator = self._compose_evaluator(parse_results[0])
        right_evaluator = self._compose_evaluator(parse_results[2])

        def evaluator(root, variant_chunk_index):
            left_array = left_evaluator(root, variant_chunk_index)
            right_array = right_evaluator(root, variant_chunk_index)
            return np.logical_and(left_array, right_array)

        return evaluator

    def _compose_or_evaluator(self, parse_results: pp.ParseResults) -> Callable:
        assert len(parse_results) == 3

        left_evaluator = self._compose_evaluator(parse_results[0])
        right_evaluator = self._compose_evaluator(parse_results[2])

        def evaluator(root, variant_chunk_index: int):
            left_array = left_evaluator(root, variant_chunk_index)
            right_array = right_evaluator(root, variant_chunk_index)
            return np.logical_or(left_array, right_array)

        return evaluator

    def _compose_evaluator(self, parse_results: pp.ParseResults) -> Callable:
        results_name = parse_results.get_name()
        return self._composers[results_name](parse_results)
