import re
from io import StringIO

import numpy as np
import pyparsing as pp
import pytest

from tests import vcz_builder
from tests.utils import make_reader
from vcztools import constants
from vcztools.query import (
    QueryFormatParser,
    QueryFormatter,
    _format_sample_loop_tag,
    write_query,
)
from vcztools.retrieval import VczReader


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
        ("query_format", "sample_filter_pass", "expected_result"),
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
    def test_sample_filter_pass(
        self, fx_reader, query_format, sample_filter_pass, expected_result
    ):
        formatter = QueryFormatter(query_format, fx_reader)
        result = ""
        for i, variant in enumerate(fx_reader.variants()):
            if sample_filter_pass is not None:
                variant["sample_filter_pass"] = sample_filter_pass[i]
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

    def test_alt_subfield(self, fx_reader):
        """ALT subfield indexing into the trimmed ALT alleles."""
        formatter = QueryFormatter(r"%ALT{1}\n", fx_reader)
        # Variant at 20:1110696 (index 4) has ALT=G,T.
        variants = list(fx_reader.variants())
        assert formatter.format_variant(variants[4]) == "T\n"

    def test_alt_subfield_no_alt(self, fx_reader):
        """ALT subfield on a variant with no ALT allele returns '.'."""
        formatter = QueryFormatter(r"%ALT{0}\n", fx_reader)
        # Variant at 20:1230237 (index 5) has ALT=. (no alternate allele).
        variants = list(fx_reader.variants())
        assert formatter.format_variant(variants[5]) == ".\n"


class TestFormatSampleLoopTag:
    """Unit tests for the per-sample tag renderer used inside ``[...]``."""

    def test_scalar_per_sample_int(self):
        value = np.array([0, constants.INT_MISSING, 5], dtype=np.int32)
        # One value per sample; the missing sentinel renders as ".".
        assert _format_sample_loop_tag(value, sample_count=3) == ["0", ".", "5"]

    def test_scalar_per_sample_float_preserves_repr(self):
        value = np.array([1.5, constants.FLOAT32_MISSING], dtype=np.float32)
        assert _format_sample_loop_tag(value, sample_count=2) == ["1.5", "."]

    def test_multivalued_per_sample(self):
        value = np.array([[1, 2], [3, constants.INT_FILL]], dtype=np.int32)
        # Trailing fill in the second sample is trimmed to a single value.
        assert _format_sample_loop_tag(value, sample_count=2) == ["1,2", "3"]

    def test_multivalued_all_fill_row(self):
        value = np.array(
            [[1, 2], [constants.INT_FILL, constants.INT_FILL]], dtype=np.int32
        )
        # An all-fill sample row trims to empty and renders as ".".
        assert _format_sample_loop_tag(value, sample_count=2) == ["1,2", "."]

    def test_multivalued_all_missing_row(self):
        value = np.array(
            [[constants.INT_MISSING, constants.INT_MISSING]], dtype=np.int32
        )
        # An all-missing row keeps both elements, each rendered as ".".
        assert _format_sample_loop_tag(value, sample_count=1) == [".,."]

    def test_scalar_broadcast(self):
        # A non-array value (e.g. an INFO tag inside []) broadcasts.
        assert _format_sample_loop_tag("19", sample_count=3) == ["19", "19", "19"]


class TestQueryFillTrimming:
    """Integer FORMAT/INFO multi-valued fields drop trailing fill and
    render per-element missing as ``.`` in the query path."""

    def _info_store(self, data):
        return vcz_builder.make_vcz(
            variant_contig=[0, 0],
            variant_position=[10, 20],
            alleles=[["A", "C"], ["G", "T"]],
            num_samples=1,
            call_genotype=np.zeros((2, 1, 2), dtype=np.int8),
            info_fields={"XX": np.asarray(data, dtype=np.int32)},
        )

    def _query(self, group, query_format):
        reader = VczReader(group)
        formatter = QueryFormatter(query_format, reader)
        return "".join(formatter.format_variant(v) for v in reader.variants())

    def test_info_multivalued_trailing_fill(self):
        group = self._info_store([[1, 2], [3, constants.INT_FILL]])
        assert self._query(group, "%INFO/XX\n") == "1,2\n3\n"

    def test_info_subfield_index_into_fill(self):
        group = self._info_store([[1, constants.INT_FILL], [5, 6]])
        # Row 0 has only one element after trimming; index 1 points at the
        # trimmed fill and renders ".".
        assert self._query(group, "%XX{1}\n") == ".\n6\n"
        assert self._query(group, "%XX{0}\n") == "1\n5\n"

    def test_format_multivalued_trailing_fill(self):
        ad = np.array([[[1, 2], [3, constants.INT_FILL]]], dtype=np.int32)
        group = vcz_builder.make_vcz(
            variant_contig=[0],
            variant_position=[10],
            alleles=[["A", "C"]],
            num_samples=2,
            call_genotype=np.zeros((1, 2, 2), dtype=np.int8),
            call_fields={"AD": ad},
        )
        # Sample 0 is full [1, 2]; sample 1 has a trailing fill, trimmed to [3].
        assert self._query(group, "[%AD ]\n") == "1,2 3 \n"

    def test_format_multivalued_all_fill_sample(self):
        ad = np.array(
            [[[1, 2], [constants.INT_FILL, constants.INT_FILL]]], dtype=np.int32
        )
        group = vcz_builder.make_vcz(
            variant_contig=[0],
            variant_position=[10],
            alleles=[["A", "C"]],
            num_samples=2,
            call_genotype=np.zeros((1, 2, 2), dtype=np.int8),
            call_fields={"AD": ad},
        )
        # Sample 1 is all-fill, trimmed to empty, rendered as ".".
        assert self._query(group, "[%AD ]\n") == "1,2 . \n"

    def test_gt_mixed_ploidy(self):
        # Sample 0 is haploid (second allele is INT_FILL); sample 1 diploid.
        gt = np.array([[[0, constants.INT_FILL], [1, 1]]], dtype=np.int8)
        group = vcz_builder.make_vcz(
            variant_contig=[0],
            variant_position=[10],
            alleles=[["A", "C"]],
            num_samples=2,
            call_genotype=gt,
        )
        reader = VczReader(group)
        formatter = QueryFormatter(r"[%GT ]\n", reader)
        variant = next(reader.variants())
        variant["call_genotype_phased"] = np.array([True, False])
        assert formatter.format_variant(variant) == "0 1/1 \n"


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


def test_write_query__include_exclude(fx_sample_vcz):
    variant_site_filter = "POS > 1"
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot handle both an include expression and an exclude expression."
        ),
    ):
        make_reader(
            fx_sample_vcz.group,
            include=variant_site_filter,
            exclude=variant_site_filter,
        )


class TestDtypeMatrixQuery:
    """Querying the dtype / shape matrix through a reordered sample subset
    (a FORMAT loop) renders identically at every width."""

    @staticmethod
    def _query(reader, query_format):
        formatter = QueryFormatter(query_format, reader)
        return "".join(formatter.format_variant(v) for v in reader.variants())

    def test_sample_subset_parity(self, fx_dtype_matrix, fx_dtype_matrix_reference):
        query_format = "%POS\t[%IAS %FAR %SAR ]\n"
        samples = ["sample_2", "sample_0"]
        native = self._query(
            make_reader(fx_dtype_matrix, samples=samples), query_format
        )
        reference = self._query(
            make_reader(fx_dtype_matrix_reference, samples=samples), query_format
        )
        assert native == reference
