import functools
import math

import numpy as np
import pyparsing as pp

from vcztools import constants
from vcztools.utils import (
    missing,
    vcf_name_to_vcz_names,
)


def list_samples(reader, output):
    sample_ids = reader.all_sample_ids
    # don't show missing samples
    print("\n".join(sample_ids[sample_ids != ""]), file=output)


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


class QueryFormatter:
    """Format variant data according to a bcftools-style query format string.

    Each variant is formatted independently — pass a single-variant dict
    (as yielded by VczReader.variants) to format_variant().
    """

    def __init__(self, query_format, reader):
        self.sample_ids = reader.sample_ids
        self.sample_count = len(self.sample_ids)
        self.contig_ids = reader.contig_ids
        self.filter_ids = reader.filter_ids
        if isinstance(query_format, str):
            parser = QueryFormatParser()
            parse_results = parser(query_format)
        else:
            assert isinstance(query_format, pp.ParseResults)
            parse_results = query_format

        self._formatters = [self._build_formatter(element) for element in parse_results]

    def format_variant(self, variant_data):
        return "".join(str(formatter(variant_data)) for formatter in self._formatters)

    def _build_formatter(self, element, *, sample_loop=False):
        if isinstance(element, pp.ParseResults):
            name = element.get_name()
            if name == "subfield":
                return self._build_subfield_formatter(element)
            elif name == "sample loop":
                return self._build_sample_loop_formatter(element)

        assert isinstance(element, str)

        if element.startswith("%"):
            return self._build_tag_formatter(element[1:], sample_loop=sample_loop)

        if sample_loop:
            count = self.sample_count
            return lambda variant_data, e=element, n=count: [e] * n
        return lambda variant_data, e=element: e

    def _build_tag_formatter(self, tag, *, sample_loop=False, subfield=False):
        if tag == "GT":
            if not sample_loop:
                raise ValueError(
                    "no such tag defined: INFO/GT. "
                    "FORMAT fields must be enclosed in square brackets,"
                    ' e.g. "[ %GT]"'
                )
            return self._format_gt

        if tag == "SAMPLE":
            if not sample_loop:
                raise ValueError("no such tag defined: INFO/SAMPLE")
            return lambda variant_data: self.sample_ids

        def format_tag(variant_data):
            vcz_names = set(variant_data.keys())
            vcz_name_matches = vcf_name_to_vcz_names(vcz_names, tag)
            if len(vcz_name_matches) == 0:
                raise ValueError(f"No mapping found for '{tag}'")
            if sample_loop:
                # FORMAT fields have precedence over INFO fields
                vcz_name = vcz_name_matches[0]
            else:
                # FORMAT fields are not allowed
                vcz_name = vcz_name_matches[-1]
                if vcz_name.startswith("call_"):
                    raise ValueError(
                        f"no such tag defined: INFO/{tag}. "
                        "FORMAT fields must be enclosed in square brackets, "
                        f'e.g. "[ %{tag}]"'
                    )

            value = variant_data[vcz_name]
            is_missing = False if isinstance(value, str) else np.all(missing(value))
            sep = ","

            if tag == "CHROM":
                value = self.contig_ids[value]
            if tag == "REF":
                value = value[0]
            if tag == "ALT":
                value = [allele for allele in value[1:] if allele] or "."
            if tag == "FILTER":
                if np.any(value):
                    value = self.filter_ids[value]
                else:
                    value = "."
                sep = ";"
            if tag == "QUAL":
                if math.isnan(value):
                    value = "."
                else:
                    value = f"{value:g}"

            if sample_loop:
                if isinstance(value, np.ndarray):
                    if value.ndim == 1:
                        # Scalar per sample (e.g., DP)
                        return [
                            (str(v) if v != constants.INT_MISSING else ".")
                            for v in value.tolist()
                            if v != constants.INT_FILL
                        ]
                    # Multi-valued per sample (e.g., HQ with shape
                    # [samples, dim])
                    result = []
                    for sample_values in value.tolist():
                        formatted = [
                            (str(v) if v != constants.INT_MISSING else ".")
                            for v in sample_values
                            if v != constants.INT_FILL
                        ]
                        result.append(",".join(formatted) if formatted else ".")
                    return result
                return [str(value)] * self.sample_count
            elif subfield:
                # Return raw value for subfield indexing
                if is_missing:
                    return "."
                return value
            else:
                if is_missing:
                    return "."
                if isinstance(value, (np.ndarray, list)):
                    return sep.join(map(str, value))
                return value

        return format_tag

    def _format_gt(self, variant_data):
        gt_row = variant_data["call_genotype"]
        if "call_genotype_phased" not in variant_data:
            # TODO: Support datasets without the phasing data
            raise NotImplementedError

        phase = variant_data["call_genotype_phased"]
        result = []
        for g, p in zip(gt_row.tolist(), phase):
            alleles = [
                str(a) if a != constants.INT_MISSING else "."
                for a in g
                if a != constants.INT_FILL
            ]
            separator = "|" if p else "/"
            result.append(separator.join(alleles))
        return result

    def _build_subfield_formatter(self, parse_results):
        assert len(parse_results) == 2
        tag_str, subfield_index = parse_results
        tag_formatter = self._build_tag_formatter(tag_str[1:], subfield=True)

        def format_subfield(variant_data):
            value = tag_formatter(variant_data)
            if isinstance(value, str):
                assert value == "."
                return "."
            if subfield_index < len(value):
                return value[subfield_index]
            return "."

        return format_subfield

    def _build_sample_loop_formatter(self, parse_results):
        formatters = [
            self._build_formatter(element, sample_loop=True)
            for element in parse_results
        ]

        def format_sample_loop(variant_data):
            sample_values = [f(variant_data) for f in formatters]
            zipped = zip(*sample_values)
            if "sample_filter_pass" not in variant_data:
                parts = (str(part) for sample in zipped for part in sample)
            else:
                sample_filter_pass = variant_data["sample_filter_pass"]
                parts = (
                    str(part)
                    for j, sample in enumerate(zipped)
                    if sample_filter_pass[j]
                    for part in sample
                )
            return "".join(parts)

        return format_sample_loop


def write_query(
    reader,
    output,
    *,
    query_format: str,
    disable_automatic_newline: bool = False,
):
    if "\\n" not in query_format and not disable_automatic_newline:
        query_format = query_format + "\\n"

    formatter = QueryFormatter(query_format, reader)

    for variant in reader.variants():
        print(formatter.format_variant(variant), sep="", end="", file=output)
