import functools
import operator
from typing import Callable

import numpy as np
import pyparsing as pp

from vcztools.utils import vcf_name_to_vcz_name


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
        base_evaluator = self._compose_evaluator(parse_results)

        def evaluator(root, variant_chunk_index: int) -> np.ndarray:
            base_array = base_evaluator(root, variant_chunk_index)
            return np.any(base_array, axis=tuple(range(1, base_array.ndim)))

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
            vcz_names = set(root.keys())
            vcz_name = vcf_name_to_vcz_name(vcz_names, vcf_name)
            zarray = root[vcz_name]
            variant_chunk_len = zarray.chunks[0]
            start = variant_chunk_len * variant_chunk_index
            end = start + variant_chunk_len
            # We load all samples (regardless of sample filtering)
            # to match bcftools' behavior.
            array = zarray[start:end]
            array = comparator(array, parse_results[2])

            if array.ndim > 2:
                return np.any(array, axis=tuple(range(2, array.ndim)))
            else:
                return array

        return evaluator

    def _compose_and_evaluator(self, parse_results: pp.ParseResults) -> Callable:
        assert len(parse_results) == 3
        assert parse_results[1] in {"&", "&&"}

        left_evaluator = self._compose_evaluator(parse_results[0])
        right_evaluator = self._compose_evaluator(parse_results[2])

        def evaluator(root, variant_chunk_index):
            left_array = left_evaluator(root, variant_chunk_index)
            right_array = right_evaluator(root, variant_chunk_index)

            if parse_results[1] == "&":
                return np.logical_and(left_array, right_array)
            else:
                left_array = np.any(left_array, axis=tuple(range(1, left_array.ndim)))
                right_array = np.any(
                    right_array, axis=tuple(range(1, right_array.ndim))
                )
                return np.logical_and(left_array, right_array)

        return evaluator

    def _compose_or_evaluator(self, parse_results: pp.ParseResults) -> Callable:
        assert len(parse_results) == 3
        assert parse_results[1] in {"|", "||"}

        left_evaluator = self._compose_evaluator(parse_results[0])
        right_evaluator = self._compose_evaluator(parse_results[2])

        def evaluator(root, variant_chunk_index: int):
            left_array = left_evaluator(root, variant_chunk_index)
            right_array = right_evaluator(root, variant_chunk_index)

            if parse_results[1] == "|":
                return np.logical_or(left_array, right_array)
            else:
                left_array = np.any(left_array, axis=tuple(range(1, left_array.ndim)))
                right_array = np.any(
                    right_array, axis=tuple(range(1, right_array.ndim))
                )
                return np.logical_or(left_array, right_array)

        return evaluator

    def _compose_evaluator(self, parse_results: pp.ParseResults) -> Callable:
        results_name = parse_results.get_name()
        return self._composers[results_name](parse_results)
