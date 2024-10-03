import pathlib
import re
from io import StringIO

import pyparsing as pp
import pytest
import zarr

from tests.utils import vcz_path_cache
from vcztools.query import (
    QueryFormatGenerator,
    QueryFormatParser,
    list_samples,
    write_query,
)


def test_list_samples(tmp_path):
    vcf_path = pathlib.Path("tests/data/vcf") / "sample.vcf.gz"
    vcz_path = vcz_path_cache(vcf_path)
    expected_output = "NA00001\nNA00002\nNA00003\n"

    with StringIO() as output:
        list_samples(vcz_path, output)
        assert output.getvalue() == expected_output


class TestQueryFormatParser:
    @pytest.fixture()
    def parser(self):
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
    def test_valid_expressions(self, parser, expression, expected_result):
        assert parser(expression).as_list() == expected_result

    @pytest.mark.parametrize(
        "expression",
        [
            "%ac",
            "%AC {1}",
            "% CHROM",
        ],
    )
    def test_invalid_expressions(self, parser, expression):
        with pytest.raises(pp.ParseException):
            parser(expression)


class TestQueryFormatEvaluator:
    @pytest.fixture()
    def root(self):
        vcf_path = pathlib.Path("tests/data/vcf/sample.vcf.gz")
        vcz_path = vcz_path_cache(vcf_path)
        return zarr.open(vcz_path, mode="r")

    @pytest.mark.parametrize(
        ("query_format", "expected_result"),
        [
            (r"A\t", "A\t" * 9),
            (r"CHROM", "CHROM" * 9),
            (
                r"%CHROM:%POS\n",
                "19:111\n19:112\n20:14370\n20:17330\n20:1110696\n20:1230237\n20:1234567\n20:1235237\nX:10\n",
            ),
            (r"%INFO/DP\n", ".\n.\n14\n11\n10\n13\n9\n.\n.\n"),
            (r"%AC\n", ".\n.\n.\n.\n.\n.\n1,1\n.\n.\n"),
            (r"%AC{0}\n", ".\n.\n.\n.\n.\n.\n1\n.\n.\n"),
        ],
    )
    def test(self, root, query_format, expected_result):
        generator = QueryFormatGenerator(query_format)
        result = "".join(generator(root))
        assert result == expected_result

    @pytest.mark.parametrize(
        ("query_format", "expected_result"),
        [(r"%QUAL\n", "9.6\n10\n29\n3\n67\n47\n50\n.\n10\n")],
    )
    def test_with_parse_results(self, root, query_format, expected_result):
        parser = QueryFormatParser()
        parse_results = parser(query_format)
        generator = QueryFormatGenerator(parse_results)
        result = "".join(generator(root))
        assert result == expected_result


def test_write_query__include_exclude(tmp_path):
    original = pathlib.Path("tests/data/vcf") / "sample.vcf.gz"
    vcz = vcz_path_cache(original)
    output = tmp_path.joinpath("output.vcf")

    query_format = r"%POS\n"
    variant_site_filter = "POS > 1"

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot handle both an include expression and an exclude expression."
        ),
    ):
        write_query(
            vcz,
            output,
            query_format=query_format,
            include=variant_site_filter,
            exclude=variant_site_filter,
        )
