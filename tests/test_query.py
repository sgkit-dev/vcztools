import pathlib
from io import StringIO

import pyparsing as pp
import pytest

from tests.utils import vcz_path_cache
from vcztools.query import QueryFormatParser, list_samples


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
                r"Allelic depth: %INFO/AD\n",
                ["Allelic", " ", "depth:", " ", "%INFO/AD", "\n"],
            ),
        ],
    )
    def test_valid_expressions(self, parser, expression, expected_result):
        assert parser(expression).as_list() == expected_result

    @pytest.mark.parametrize(
        "expression",
        [
            ("%ac",),
            ("%AC {1}",),
            ("% CHROM",),
        ],
    )
    def test_invalid_expressions(self, parser, expression):
        with pytest.raises(pp.ParseException):
            parser(expression)
