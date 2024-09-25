import functools
import itertools
import math
from typing import Callable, Optional, Union

import numpy as np
import pyparsing as pp
import zarr

from vcztools.utils import open_file_like, vcf_name_to_vcz_name


def list_samples(vcz_path, output=None):
    root = zarr.open(vcz_path, mode="r")

    with open_file_like(output) as output:
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
        subscript_pattern = pp.Group(
            tag_pattern
            + pp.Literal("{").suppress()
            + pp.common.integer
            + pp.Literal("}").suppress()
        )
        newline_pattern = pp.Literal("\\n").set_parse_action(pp.replace_with("\n"))
        tab_pattern = pp.Literal("\\t").set_parse_action(pp.replace_with("\t"))
        pattern = pp.ZeroOrMore(
            subscript_pattern
            | tag_pattern
            | newline_pattern
            | tab_pattern
            | pp.White()
            | pp.Word(pp.printables, exclude_chars=r"\{}[]%")
        )
        pattern = pattern.leave_whitespace()

        self._parser = functools.partial(pattern.parse_string, parse_all=True)

    def __call__(self, *args, **kwargs):
        assert len(args) == 1
        assert not kwargs

        return self._parser(args[0])


class QueryFormatGenerator:
    def __init__(self, query_format: Union[str, pp.ParseResults]):
        if isinstance(query_format, str):
            parser = QueryFormatParser()
            parse_results = parser(query_format)
        else:
            assert isinstance(query_format, pp.ParseResults)
            parse_results = query_format

        self._generator = self._compose_generator(parse_results)

    def __call__(self, *args, **kwargs):
        assert len(args) == 1
        assert not kwargs

        yield from self._generator(args[0])

    def _compose_tag_generator(self, tag: str, *, subscript=False) -> Callable:
        assert tag.startswith("%")
        tag = tag[1:]

        def generate(root):
            vcz_names = set(name for name, _zarray in root.items())
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
                        row = [allele for allele in row[1:] if allele]
                        row = row or "."
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
                    if not subscript and (
                        isinstance(row, np.ndarray) or isinstance(row, list)
                    ):
                        row = ",".join(map(str, row))

                    yield row if not is_missing else "."

        return generate

    def _compose_subscript_generator(self, parse_results: pp.ParseResults) -> Callable:
        assert len(parse_results) == 2

        tag, subscript_index = parse_results
        tag_generator = self._compose_tag_generator(tag, subscript=True)

        def generate(root):
            for tag in tag_generator(root):
                if isinstance(tag, str):
                    assert tag == "."
                    yield "."
                else:
                    if subscript_index < len(tag):
                        yield tag[subscript_index]
                    else:
                        yield "."

        return generate

    def _compose_element_generator(
        self, element: Union[str, pp.ParseResults]
    ) -> Callable:
        if isinstance(element, pp.ParseResults):
            return self._compose_subscript_generator(element)

        assert isinstance(element, str)

        if element.startswith("%"):
            return self._compose_tag_generator(element)
        else:

            def generate(root):
                variant_count = root["variant_position"].shape[0]
                yield from itertools.repeat(element, variant_count)

            return generate

    def _compose_generator(self, parse_results: pp.ParseResults) -> Callable:
        generators = (
            self._compose_element_generator(element) for element in parse_results
        )

        def generate(root) -> str:
            iterables = (generator(root) for generator in generators)

            for results in zip(*iterables):
                results = map(str, results)
                yield "".join(results)

        return generate


def write_query(
    vcz,
    output=None,
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
    generator = QueryFormatGenerator(query_format)

    with open_file_like(output) as output:
        for result in generator(root):
            print(result, sep="", end="", file=output)
