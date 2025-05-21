import functools
import itertools
import math
from collections.abc import Callable

import numpy as np
import pyparsing as pp
import zarr

from vcztools import constants, retrieval
from vcztools.samples import parse_samples
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
    def __init__(self, query_format, sample_ids, contigs, filters):
        self.sample_ids = sample_ids
        self.sample_count = len(self.sample_ids)
        self.contig_ids = contigs
        self.filter_ids = filters
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

    def _compose_gt_generator(self) -> Callable:
        def generate(chunk_data):
            gt_array = chunk_data["call_genotype"]

            if "call_genotype_phased" in chunk_data:
                phase_array = chunk_data["call_genotype_phased"]
                assert gt_array.shape[:2] == phase_array.shape

                for gt_row, phase in zip(gt_array, phase_array):

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
        def generate(chunk_data):
            variant_count = chunk_data["variant_position"].shape[0]
            yield from itertools.repeat(self.sample_ids, variant_count)

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

        def generate(chunk_data):
            vcz_names = set(chunk_data.keys())
            vcz_name = vcf_name_to_vcz_name(vcz_names, tag)
            array = chunk_data[vcz_name]
            for row in array:
                is_missing = np.any(row == -1)
                sep = ","

                if tag == "CHROM":
                    row = self.contig_ids[row]
                if tag == "REF":
                    row = row[0]
                if tag == "ALT":
                    row = [allele for allele in row[1:] if allele] or "."
                if tag == "FILTER":
                    if np.any(row):
                        row = self.filter_ids[row]
                    else:
                        row = "."
                    sep = ";"
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
                    row = sep.join(map(str, row))

                if sample_loop:
                    if isinstance(row, np.ndarray):
                        row = row.tolist()
                        row = [
                            (str(element) if element != constants.INT_MISSING else ".")
                            for element in row
                            if element != constants.INT_FILL
                        ]
                        yield row
                    else:
                        yield itertools.repeat(str(row), self.sample_count)
                else:
                    yield row if not is_missing else "."

        return generate

    def _compose_subfield_generator(self, parse_results: pp.ParseResults) -> Callable:
        assert len(parse_results) == 2

        tag, subfield_index = parse_results
        tag_generator = self._compose_tag_generator(tag, subfield=True)

        def generate(chunk_data):
            for tag in tag_generator(chunk_data):
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

        def generate(chunk_data):
            iterables = (generator(chunk_data) for generator in generators)
            zipped = zip(*iterables)
            zipped_zipped = (zip(*element) for element in zipped)
            if "call_mask" not in chunk_data:
                flattened_zipped_zipped = (
                    (
                        subsubelement
                        for subelement in element  # sample-wise
                        for subsubelement in subelement
                    )
                    for element in zipped_zipped  # variant-wise
                )
            else:
                call_mask = chunk_data["call_mask"]
                flattened_zipped_zipped = (
                    (
                        subsubelement
                        for j, subelement in enumerate(element)  # sample-wise
                        if call_mask[i, j]
                        for subsubelement in subelement
                    )
                    for i, element in enumerate(zipped_zipped)  # variant-wise
                )
            yield from map("".join, flattened_zipped_zipped)

        return generate

    def _compose_element_generator(
        self, element: str | pp.ParseResults, *, sample_loop=False
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

            def generate(chunk_data):
                nonlocal element
                variant_count = chunk_data["variant_position"].shape[0]
                if sample_loop:
                    for _ in range(variant_count):
                        yield itertools.repeat(element, self.sample_count)
                else:
                    yield from itertools.repeat(element, variant_count)

            return generate

    def _compose_generator(
        self,
        parse_results,
    ) -> Callable:
        generators = (
            self._compose_element_generator(element) for element in parse_results
        )

        def generate(chunk_data) -> str:
            iterables = (generator(chunk_data) for generator in generators)
            for results in zip(*iterables):
                results = map(str, results)
                yield "".join(results)

        return generate


def write_query(
    vcz,
    output,
    *,
    query_format: str,
    regions=None,
    targets=None,
    samples=None,
    force_samples: bool = False,
    include: str | None = None,
    exclude: str | None = None,
):
    root = zarr.open(vcz, mode="r")

    all_samples = root["sample_id"][:]
    sample_ids, samples_selection = parse_samples(
        samples, all_samples, force_samples=force_samples
    )
    contigs = root["contig_id"][:]
    filters = root["filter_id"][:]

    generator = QueryFormatGenerator(query_format, sample_ids, contigs, filters)

    for chunk_data in retrieval.variant_chunk_iter(
        root,
        variant_regions=regions,
        variant_targets=targets,
        include=include,
        exclude=exclude,
        samples_selection=samples_selection,
    ):
        for result in generator(chunk_data):
            print(result, sep="", end="", file=output)
