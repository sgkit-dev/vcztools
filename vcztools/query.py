import functools
import itertools
import math
from typing import Callable, Optional, Union

import numpy as np
import pyparsing as pp
import zarr

from vcztools import constants
from vcztools.filter import FilterExpressionEvaluator, FilterExpressionParser
from vcztools.utils import vcf_name_to_vcz_name


def list_samples(vcz_path, output):
    root = zarr.open(vcz_path, mode="r")

    sample_ids = root["sample_id"][:]
    print("\n".join(sample_ids), file=output)


class QueryFormatParser:
    def __init__(self):
        info_tag_pattern = pp.Combine(
            pp.Literal("%INFO/") + pp.Word(pp.srange("[A-Z]"))
        )
        tag_pattern = info_tag_pattern | pp.Combine(
            pp.Literal("%") + pp.Regex(r"[A-Z]+\d?")
        )
        subfield_pattern = pp.Group(
            tag_pattern
            + pp.Literal("{").suppress()
            + pp.common.integer
            + pp.Literal("}").suppress()
        ).set_results_name("subfield")
        newline_pattern = pp.Literal("\\n").set_parse_action(pp.replace_with("\n"))
        tab_pattern = pp.Literal("\\t").set_parse_action(pp.replace_with("\t"))
        format_pattern = pp.Forward()
        sample_loop_pattern = pp.Group(
            pp.Literal("[").suppress() + format_pattern + pp.Literal("]").suppress()
        ).set_results_name("sample loop")
        format_pattern <<= pp.ZeroOrMore(
            sample_loop_pattern
            | subfield_pattern
            | tag_pattern
            | newline_pattern
            | tab_pattern
            | pp.White()
            | pp.Word(pp.printables, exclude_chars=r"\{}[]%")
        ).leave_whitespace()

        self._parser = functools.partial(format_pattern.parse_string, parse_all=True)

    def __call__(self, *args, **kwargs):
        assert len(args) == 1
        assert not kwargs

        return self._parser(args[0])


class QueryFormatGenerator:
    def __init__(
        self,
        query_format: Union[str, pp.ParseResults],
        *,
        include: Optional[str] = None,
        exclude: Optional[str] = None,
    ):
        if isinstance(query_format, str):
            parser = QueryFormatParser()
            parse_results = parser(query_format)
        else:
            assert isinstance(query_format, pp.ParseResults)
            parse_results = query_format

        self._generator = self._compose_generator(
            parse_results, include=include, exclude=exclude
        )

    def __call__(self, *args, **kwargs):
        assert len(args) == 1
        assert not kwargs

        yield from self._generator(args[0])

    def _compose_gt_generator(self) -> Callable:
        def generate(root):
            gt_zarray = root["call_genotype"]
            v_chunk_size = gt_zarray.chunks[0]

            if "call_genotype_phased" in root:
                phase_zarray = root["call_genotype_phased"]
                assert gt_zarray.chunks[:2] == phase_zarray.chunks
                assert gt_zarray.shape[:2] == phase_zarray.shape

                for v_chunk_index in range(gt_zarray.cdata_shape[0]):
                    start = v_chunk_index * v_chunk_size
                    end = start + v_chunk_size

                    for gt_row, phase in zip(
                        gt_zarray[start:end], phase_zarray[start:end]
                    ):

                        def stringify(gt_and_phase: tuple):
                            gt, phase = gt_and_phase
                            gt = [
                                str(allele) if allele != constants.INT_MISSING else "."
                                for allele in gt
                                if allele != constants.INT_FILL
                            ]
                            separator = "|" if phase else "/"
                            return separator.join(gt)

                        gt_row = gt_row.tolist()
                        yield map(stringify, zip(gt_row, phase))
            else:
                # TODO: Support datasets without the phasing data
                raise NotImplementedError

        return generate

    def _compose_sample_ids_generator(self) -> Callable:
        def generate(root):
            variant_count = root["variant_position"].shape[0]
            sample_ids = root["sample_id"][:].tolist()
            yield from itertools.repeat(sample_ids, variant_count)

        return generate

    def _compose_tag_generator(
        self, tag: str, *, subfield=False, sample_loop=False
    ) -> Callable:
        assert tag.startswith("%")
        tag = tag[1:]

        if tag == "GT":
            return self._compose_gt_generator()

        if tag == "SAMPLE":
            return self._compose_sample_ids_generator()

        def generate(root):
            vcz_names = set(root.keys())
            vcz_name = vcf_name_to_vcz_name(vcz_names, tag)
            zarray = root[vcz_name]
            contig_ids = root["contig_id"][:] if tag == "CHROM" else None
            filter_ids = root["filter_id"][:] if tag == "FILTER" else None
            v_chunk_size = zarray.chunks[0]

            for v_chunk_index in range(zarray.cdata_shape[0]):
                start = v_chunk_index * v_chunk_size
                end = start + v_chunk_size

                for row in zarray[start:end]:
                    is_missing = np.any(row == -1)

                    if tag == "CHROM":
                        assert contig_ids is not None
                        row = contig_ids[row]
                    if tag == "REF":
                        row = row[0]
                    if tag == "ALT":
                        row = [allele for allele in row[1:] if allele] or "."
                    if tag == "FILTER":
                        assert filter_ids is not None

                        if np.any(row):
                            row = filter_ids[row]
                        else:
                            row = "."
                    if tag == "QUAL":
                        if math.isnan(row):
                            row = "."
                        else:
                            row = f"{row:g}"
                    if (
                        not subfield
                        and not sample_loop
                        and (isinstance(row, np.ndarray) or isinstance(row, list))
                    ):
                        row = ",".join(map(str, row))

                    if sample_loop:
                        sample_count = root["sample_id"].shape[0]

                        if isinstance(row, np.ndarray):
                            row = row.tolist()
                            row = [
                                str(element)
                                if element != constants.INT_MISSING
                                else "."
                                for element in row
                                if element != constants.INT_FILL
                            ]
                            yield row
                        else:
                            yield itertools.repeat(str(row), sample_count)
                    else:
                        yield row if not is_missing else "."

        return generate

    def _compose_subfield_generator(self, parse_results: pp.ParseResults) -> Callable:
        assert len(parse_results) == 2

        tag, subfield_index = parse_results
        tag_generator = self._compose_tag_generator(tag, subfield=True)

        def generate(root):
            for tag in tag_generator(root):
                if isinstance(tag, str):
                    assert tag == "."
                    yield "."
                else:
                    if subfield_index < len(tag):
                        yield tag[subfield_index]
                    else:
                        yield "."

        return generate

    def _compose_sample_loop_generator(
        self, parse_results: pp.ParseResults
    ) -> Callable:
        generators = map(
            functools.partial(self._compose_element_generator, sample_loop=True),
            parse_results,
        )

        def generate(root):
            iterables = (generator(root) for generator in generators)
            zipped = zip(*iterables)
            zipped_zipped = (zip(*element) for element in zipped)
            flattened_zipped_zipped = (
                (
                    subsubelement
                    for subelement in element  # sample-wise
                    for subsubelement in subelement
                )
                for element in zipped_zipped  # variant-wise
            )
            yield from map("".join, flattened_zipped_zipped)

        return generate

    def _compose_element_generator(
        self, element: Union[str, pp.ParseResults], *, sample_loop=False
    ) -> Callable:
        if isinstance(element, pp.ParseResults):
            if element.get_name() == "subfield":
                return self._compose_subfield_generator(element)
            elif element.get_name() == "sample loop":
                return self._compose_sample_loop_generator(element)

        assert isinstance(element, str)

        if element.startswith("%"):
            return self._compose_tag_generator(element, sample_loop=sample_loop)
        else:

            def generate(root):
                nonlocal element
                variant_count = root["variant_position"].shape[0]
                if sample_loop:
                    sample_count = root["sample_id"].shape[0]
                    for _ in range(variant_count):
                        yield itertools.repeat(element, sample_count)
                else:
                    yield from itertools.repeat(element, variant_count)

            return generate

    def _compose_filter_generator(
        self, *, include: Optional[str] = None, exclude: Optional[str] = None
    ) -> Callable:
        assert not (include and exclude)

        if not include and not exclude:

            def generate(root):
                variant_count = root["variant_position"].shape[0]
                yield from itertools.repeat(True, variant_count)

            return generate

        parser = FilterExpressionParser()
        parse_results = parser(include or exclude)[0]
        filter_evaluator = FilterExpressionEvaluator(
            parse_results, invert=bool(exclude)
        )

        def generate(root):
            nonlocal filter_evaluator

            filter_evaluator = functools.partial(filter_evaluator, root)
            variant_chunk_count = root["variant_position"].cdata_shape[0]

            for variant_chunk_index in range(variant_chunk_count):
                yield from filter_evaluator(variant_chunk_index)

        return generate

    def _compose_generator(
        self,
        parse_results: pp.ParseResults,
        *,
        include: Optional[str] = None,
        exclude: Optional[str] = None,
    ) -> Callable:
        generators = (
            self._compose_element_generator(element) for element in parse_results
        )
        filter_generator = self._compose_filter_generator(
            include=include, exclude=exclude
        )

        def generate(root) -> str:
            iterables = (generator(root) for generator in generators)
            filter_iterable = filter_generator(root)
            for results, filter_indicator in zip(zip(*iterables), filter_iterable):
                if filter_indicator:
                    results = map(str, results)
                    yield "".join(results)

        return generate


def write_query(
    vcz,
    output,
    *,
    query_format: str,
    include: Optional[str] = None,
    exclude: Optional[str] = None,
):
    if include and exclude:
        raise ValueError(
            "Cannot handle both an include expression and an exclude expression."
        )

    root = zarr.open(vcz, mode="r")
    generator = QueryFormatGenerator(query_format, include=include, exclude=exclude)

    for result in generator(root):
        print(result, sep="", end="", file=output)
