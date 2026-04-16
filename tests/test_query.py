import re
from io import StringIO

import numpy as np
import pyparsing as pp
import pytest

from tests import vcz_builder
from vcztools.query import (
    QueryFormatParser,
    QueryFormatter,
    list_samples,
    write_query,
)
from vcztools.retrieval import VczReader


def test_list_samples(fx_sample_vcz):
    expected_output = "NA00001\nNA00002\nNA00003\n"
    reader = VczReader(fx_sample_vcz.group)
    with StringIO() as output:
        list_samples(reader, output)
        assert output.getvalue() == expected_output


def test_list_samples__missing(fx_sample_vcz):
    mutated = vcz_builder.copy_vcz(fx_sample_vcz.group)
    # delete sample NA00002 at index 1
    mutated["sample_id"][1] = ""

    reader = VczReader(mutated)
    expected_output = "NA00001\nNA00003\n"
    with StringIO() as output:
        list_samples(reader, output)
        assert output.getvalue() == expected_output


class TestQueryFormatParser:
    @pytest.fixture
    def fx_parser(self):
        return QueryFormatParser()

    @pytest.mark.parametrize(
        ("expression", "expected_result"),
        [
            ("%CHROM", ["%CHROM"]),
            (r"\n", ["\n"]),
            (r"\t", ["\t"]),
            (r"%CHROM\n", ["%CHROM", "\n"]),
            ("%CHROM  %POS  %REF", ["%CHROM", "  ", "%POS", "  ", "%REF"]),
            (r"%CHROM  %POS0  %REF\n", ["%CHROM", "  ", "%POS0", "  ", "%REF", "\n"]),
            (
                r"%CHROM\t%POS\t%REF\t%ALT{0}\n",
                ["%CHROM", "\t", "%POS", "\t", "%REF", "\t", ["%ALT", 0], "\n"],
            ),
            (
                r"%CHROM\t%POS0\t%END\t%ID\n",
                ["%CHROM", "\t", "%POS0", "\t", "%END", "\t", "%ID", "\n"],
            ),
            (r"%CHROM:%POS\n", ["%CHROM", ":", "%POS", "\n"]),
            (r"%AC{1}\n", [["%AC", 1], "\n"]),
            (
                r"Read depth: %INFO/DP\n",
                ["Read", " ", "depth:", " ", "%INFO/DP", "\n"],
            ),
            (
                r"%CHROM\t%POS\t%REF\t%ALT[\t%SAMPLE=%GT]\n",
                [
                    "%CHROM",
                    "\t",
                    "%POS",
                    "\t",
                    "%REF",
                    "\t",
                    "%ALT",
                    ["\t", "%SAMPLE", "=", "%GT"],
                    "\n",
                ],
            ),
            (
                r"%CHROM\t%POS\t%REF\t%ALT[\t%SAMPLE=%GT{0}]\n",
                [
                    "%CHROM",
                    "\t",
                    "%POS",
                    "\t",
                    "%REF",
                    "\t",
                    "%ALT",
                    ["\t", "%SAMPLE", "=", ["%GT", 0]],
                    "\n",
                ],
            ),
            (
                r"GQ:[ %GQ] \t GT:[ %GT]\n",
                ["GQ:", [" ", "%GQ"], " ", "\t", " ", "GT:", [" ", "%GT"], "\n"],
            ),
            (
                r"[%SAMPLE %GT %DP\n]",
                [["%SAMPLE", " ", "%GT", " ", "%DP", "\n"]],
            ),
        ],
    )
    def test_valid_expressions(self, fx_parser, expression, expected_result):
        assert fx_parser(expression).as_list() == expected_result

    @pytest.mark.parametrize(
        "expression",
        [
            "%ac",
            "%AC {1}",
            "% CHROM",
        ],
    )
    def test_invalid_expressions(self, fx_parser, expression):
        with pytest.raises(pp.ParseException):
            fx_parser(expression)


class TestQueryFormatEvaluator:
    @pytest.fixture
    def fx_root(self, fx_sample_vcz):
        return fx_sample_vcz.group

    @pytest.fixture
    def fx_reader(self, fx_root):
        return VczReader(fx_root)

    @pytest.mark.parametrize(
        ("query_format", "expected_result"),
        [
            (r"A\t", "A\t" * 9),
            (r"CHROM", "CHROM" * 9),
            (
                r"%CHROM:%POS\n",
                "19:111\n19:112\n20:14370\n20:17330\n20:1110696\n20:1230237\n20:1234567\n20:1235237\nX:10\n",
            ),
            (
                r"%REF\t%ALT\n",
                "A\tC\nA\tG\nG\tA\nT\tA\nA\tG,T\nT\t.\nG\tGA,GAC\nT\t.\nAC\tA,ATG,C\n",
            ),
            (r"%ID\n", ".\n.\nrs6054257\n.\nrs6040355\n.\nmicrosat1\n.\nrsTest\n"),
            (r"%FILTER\n", ".\n.\nPASS\nq10\nPASS\nPASS\nPASS\n.\nPASS\n"),
            (r"%INFO/DP\n", ".\n.\n14\n11\n10\n13\n9\n.\n.\n"),
            (r"%AC\n", ".\n.\n.\n.\n.\n.\n1,1\n.\n.\n"),
            (r"%AC{0}\n", ".\n.\n.\n.\n.\n.\n1\n.\n.\n"),
            (r"%AF{0}\n", ".\n.\n0.5\n0.017\n0.333\n.\n.\n.\n.\n"),
        ],
    )
    def test(self, fx_reader, query_format, expected_result):
        formatter = QueryFormatter(query_format, fx_reader)
        result = ""
        for variant in fx_reader.variants():
            result += formatter.format_variant(variant)
        assert result == expected_result

    # fmt: off
    @pytest.mark.parametrize(
        ("query_format", "call_mask", "expected_result"),
        [
            (
                r"[%DP ]\n",
                None,
                ". . . \n. . . \n1 8 5 \n3 5 3 \n6 0 4 \n. 4 2 \n4 2 3 \n. . . \n. . . \n",  # noqa: E501
            ),
            (
                r"[%DP ]\n",
                np.array(
                    [
                        [1, 1, 1,],
                        [1, 1, 1,],
                        [1, 0, 1,],
                        [1, 1, 1,],
                        [1, 1, 1,],
                        [1, 1, 1,],
                        [1, 1, 1,],
                        [1, 1, 1,],
                        [1, 1, 1,],
                    ]
                ),
                ". . . \n. . . \n1 5 \n3 5 3 \n6 0 4 \n. 4 2 \n4 2 3 \n. . . \n. . . \n",  # noqa: E501
            ),
        ],
    )
    # fmt: on
    def test_call_mask(self, fx_reader, query_format, call_mask, expected_result):
        formatter = QueryFormatter(query_format, fx_reader)
        result = ""
        for i, variant in enumerate(fx_reader.variants()):
            if call_mask is not None:
                variant["call_mask"] = call_mask[i]
            result += formatter.format_variant(variant)
        assert result == expected_result

    @pytest.mark.parametrize(
        ("query_format", "expected_result"),
        [(r"%QUAL\n", "9.6\n10\n29\n3\n67\n47\n50\n.\n10\n")],
    )
    def test_with_parse_results(self, fx_reader, query_format, expected_result):
        parser = QueryFormatParser()
        parse_results = parser(query_format)
        formatter = QueryFormatter(parse_results, fx_reader)
        result = ""
        for variant in fx_reader.variants():
            result += formatter.format_variant(variant)
        assert result == expected_result


class TestQueryFormatterErrors:
    """Test error paths in QueryFormatter."""

    @pytest.fixture
    def fx_reader(self, fx_sample_vcz):
        return VczReader(fx_sample_vcz.group)

    def test_gt_outside_sample_loop(self, fx_reader):
        with pytest.raises(ValueError, match="no such tag defined: INFO/GT"):
            QueryFormatter(r"%GT\n", fx_reader)

    def test_sample_outside_sample_loop(self, fx_reader):
        with pytest.raises(ValueError, match="no such tag defined: INFO/SAMPLE"):
            QueryFormatter(r"%SAMPLE\n", fx_reader)

    def test_unknown_tag(self, fx_reader):
        formatter = QueryFormatter(r"%NONEXISTENT\n", fx_reader)
        variant = next(fx_reader.variants())
        with pytest.raises(ValueError, match="No mapping found for 'NONEXISTENT'"):
            formatter.format_variant(variant)

    def test_format_field_outside_sample_loop(self, fx_reader):
        """GQ is FORMAT-only, so %GQ outside [] should error."""
        formatter = QueryFormatter(r"%GQ\n", fx_reader)
        variant = next(fx_reader.variants())
        with pytest.raises(ValueError, match="no such tag defined: INFO/GQ"):
            formatter.format_variant(variant)


class TestQueryFormatterSampleLoop:
    """Test sample loop formatting paths."""

    @pytest.fixture
    def fx_reader(self, fx_sample_vcz):
        return VczReader(fx_sample_vcz.group)

    def test_gt_in_sample_loop(self, fx_reader):
        formatter = QueryFormatter(r"[\t%GT]\n", fx_reader)
        # Third variant (20:14370) has GT 0|0, 1|0, 1/1
        variants = list(fx_reader.variants())
        result = formatter.format_variant(variants[2])
        assert result == "\t0|0\t1|0\t1/1\n"

    def test_sample_in_sample_loop(self, fx_reader):
        formatter = QueryFormatter(r"[%SAMPLE ]\n", fx_reader)
        variant = next(fx_reader.variants())
        result = formatter.format_variant(variant)
        assert result == "NA00001 NA00002 NA00003 \n"

    def test_scalar_in_sample_loop(self, fx_reader):
        """INFO-level tag inside [] broadcasts to all samples."""
        formatter = QueryFormatter(r"[%CHROM ]\n", fx_reader)
        variant = next(fx_reader.variants())
        result = formatter.format_variant(variant)
        assert result == "19 19 19 \n"

    def test_multivalued_format_in_sample_loop(self, fx_reader):
        """Multi-valued FORMAT field (HQ, 2 values per sample) is comma-joined."""
        formatter = QueryFormatter(r"[%HQ ]\n", fx_reader)
        variants = list(fx_reader.variants())
        # First variant (19:111) has HQ 10,15 / 10,10 / 3,3
        assert formatter.format_variant(variants[0]) == "10,15 10,10 3,3 \n"
        # Third variant (20:14370) has HQ 51,51 / 51,51 / .,. (all-missing)
        assert formatter.format_variant(variants[2]) == "51,51 51,51 .,. \n"
        # Last variants have HQ absent — stored as all INT_MISSING in zarr
        assert formatter.format_variant(variants[6]) == ".,. .,. .,. \n"


class TestQueryFormatterSubfield:
    """Test subfield indexing paths."""

    @pytest.fixture
    def fx_reader(self, fx_sample_vcz):
        return VczReader(fx_sample_vcz.group)

    def test_subfield_out_of_bounds(self, fx_reader):
        """AC{5} on a variant with fewer elements returns '.'."""
        formatter = QueryFormatter(r"%AC{5}\n", fx_reader)
        # Variant at 20:1234567 (index 6) has AC=1,1 — index 5 is out of bounds
        variants = list(fx_reader.variants())
        result = formatter.format_variant(variants[6])
        assert result == ".\n"


class TestWriteQuery:
    """Test write_query integration."""

    def test_write_query(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        output = StringIO()
        write_query(reader, output, query_format=r"%CHROM:%POS")
        result = output.getvalue()
        lines = result.strip().split("\n")
        assert len(lines) == 9
        assert lines[0] == "19:111"
        assert lines[-1] == "X:10"

    def test_write_query_disable_automatic_newline(self, fx_sample_vcz):
        reader = VczReader(fx_sample_vcz.group)
        output = StringIO()
        write_query(
            reader,
            output,
            query_format=r"%POS ",
            disable_automatic_newline=True,
        )
        result = output.getvalue()
        assert "\n" not in result
        assert result.startswith("111 112 ")


def test_write_query__include_exclude(tmp_path, fx_sample_vcz):
    output = tmp_path.joinpath("output.vcf")

    query_format = r"%POS"
    variant_site_filter = "POS > 1"

    reader = VczReader(fx_sample_vcz.group)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot handle both an include expression and an exclude expression."
        ),
    ):
        write_query(
            reader,
            output,
            query_format=query_format,
            include=variant_site_filter,
            exclude=variant_site_filter,
        )
