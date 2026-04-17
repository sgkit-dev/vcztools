import logging

import numpy as np
import numpy.testing as nt
import pytest

from vcztools.retrieval import VczReader
from vcztools.samples import parse_samples, read_samples_file
from vcztools.vcf_writer import write_vcf

from .vcz_builder import make_vcz

ALL_SAMPLES = np.array(["NA00001", "NA00002", "NA00003"])


class TestParseSamplesNone:
    def test_default_returns_all_samples(self):
        ids, selection = parse_samples(None, ALL_SAMPLES)
        nt.assert_array_equal(ids, ["NA00001", "NA00002", "NA00003"])
        nt.assert_array_equal(selection, [0, 1, 2])

    def test_complement_has_no_effect(self):
        # samples=None exits before the complement branch; the flag is a no-op.
        ids, selection = parse_samples(None, ALL_SAMPLES, complement=True)
        nt.assert_array_equal(ids, ["NA00001", "NA00002", "NA00003"])
        nt.assert_array_equal(selection, [0, 1, 2])

    def test_skips_missing_header_samples(self):
        all_samples = np.array(["NA00001", "", "NA00003"])
        ids, selection = parse_samples(None, all_samples)
        nt.assert_array_equal(ids, ["NA00001", "NA00003"])
        nt.assert_array_equal(selection, [0, 2])


class TestParseSamplesList:
    def test_single_sample(self):
        ids, selection = parse_samples(["NA00002"], ALL_SAMPLES)
        nt.assert_array_equal(ids, ["NA00002"])
        nt.assert_array_equal(selection, [1])

    def test_subset_in_input_order(self):
        # Order of the input list is preserved in the output.
        ids, selection = parse_samples(["NA00003", "NA00001"], ALL_SAMPLES)
        nt.assert_array_equal(ids, ["NA00003", "NA00001"])
        nt.assert_array_equal(selection, [2, 0])

    def test_full_list(self):
        ids, selection = parse_samples(["NA00001", "NA00002", "NA00003"], ALL_SAMPLES)
        nt.assert_array_equal(ids, ["NA00001", "NA00002", "NA00003"])
        nt.assert_array_equal(selection, [0, 1, 2])

    def test_empty_list(self):
        ids, selection = parse_samples([], ALL_SAMPLES)
        nt.assert_array_equal(ids, [])
        nt.assert_array_equal(selection, [])

    def test_empty_string_element_raises(self):
        # bcftools rejects "" as an unknown sample; vcztools matches.
        with pytest.raises(
            ValueError, match=r"subset called for sample\(s\) not in header: \."
        ):
            parse_samples([""], ALL_SAMPLES)

    def test_empty_string_element_with_ignore_missing(self):
        # With ignore_missing_samples=True, "" is dropped silently and we
        # end up with no samples selected (matches bcftools -s '' --force-samples).
        ids, selection = parse_samples([""], ALL_SAMPLES, ignore_missing_samples=True)
        nt.assert_array_equal(ids, [])
        nt.assert_array_equal(selection, [])


class TestParseSamplesDuplicates:
    def test_duplicates_raise(self):
        with pytest.raises(ValueError, match=r'Duplicate sample name "NA00001"'):
            parse_samples(["NA00001", "NA00001"], ALL_SAMPLES)

    def test_duplicates_raise_with_ignore_missing(self):
        # The duplicate check runs after unknown-sample removal, so a list
        # that contains both unknowns and duplicates still trips on the dup
        # even when ignore_missing_samples=True (matches bcftools 1.19).
        with pytest.raises(ValueError, match=r'Duplicate sample name "NA00001"'):
            parse_samples(
                ["NA00001", "NA00001", "UNKNOWN"],
                ALL_SAMPLES,
                ignore_missing_samples=True,
            )

    def test_duplicates_collapsed_under_complement(self):
        ids, selection = parse_samples(
            ["NA00002", "NA00002"], ALL_SAMPLES, complement=True
        )
        nt.assert_array_equal(ids, ["NA00001", "NA00003"])
        nt.assert_array_equal(selection, [0, 2])

    def test_many_duplicates_collapsed_under_complement(self):
        ids, selection = parse_samples(
            ["NA00002", "NA00002", "NA00002"], ALL_SAMPLES, complement=True
        )
        nt.assert_array_equal(ids, ["NA00001", "NA00003"])
        nt.assert_array_equal(selection, [0, 2])


class TestParseSamplesComplement:
    def test_excludes_one(self):
        ids, selection = parse_samples(["NA00002"], ALL_SAMPLES, complement=True)
        nt.assert_array_equal(ids, ["NA00001", "NA00003"])
        nt.assert_array_equal(selection, [0, 2])

    def test_excludes_multiple(self):
        ids, selection = parse_samples(
            ["NA00001", "NA00003"], ALL_SAMPLES, complement=True
        )
        nt.assert_array_equal(ids, ["NA00002"])
        nt.assert_array_equal(selection, [1])

    def test_exclude_all_returns_empty(self):
        ids, selection = parse_samples(
            ["NA00001", "NA00002", "NA00003"], ALL_SAMPLES, complement=True
        )
        nt.assert_array_equal(ids, [])
        nt.assert_array_equal(selection, [])

    def test_empty_list_complement_returns_all(self):
        ids, selection = parse_samples([], ALL_SAMPLES, complement=True)
        nt.assert_array_equal(ids, ["NA00001", "NA00002", "NA00003"])
        nt.assert_array_equal(selection, [0, 1, 2])

    def test_output_follows_all_samples_order(self):
        # Input order is irrelevant under complement; output follows header order.
        ids_reversed, selection_reversed = parse_samples(
            ["NA00003", "NA00002"], ALL_SAMPLES, complement=True
        )
        ids_forward, selection_forward = parse_samples(
            ["NA00002", "NA00003"], ALL_SAMPLES, complement=True
        )
        nt.assert_array_equal(ids_reversed, ["NA00001"])
        nt.assert_array_equal(ids_forward, ["NA00001"])
        nt.assert_array_equal(selection_reversed, [0])
        nt.assert_array_equal(selection_forward, [0])


class TestParseSamplesUnknown:
    def test_one_unknown_raises(self):
        with pytest.raises(
            ValueError,
            match=(
                r"subset called for sample\(s\) not in header: UNKNOWN\. "
                r'Use "--force-samples" to ignore this error\.'
            ),
        ):
            parse_samples(["NA00001", "UNKNOWN"], ALL_SAMPLES)

    def test_message_lists_every_unknown(self):
        with pytest.raises(ValueError, match="UNKNOWN1") as excinfo:
            parse_samples(["UNKNOWN1", "UNKNOWN2", "NA00001"], ALL_SAMPLES)
        assert "UNKNOWN2" in str(excinfo.value)

    def test_ignore_missing_samples_drops_unknowns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="vcztools.samples"):
            ids, selection = parse_samples(
                ["NA00001", "UNKNOWN", "NA00003"],
                ALL_SAMPLES,
                ignore_missing_samples=True,
            )
        nt.assert_array_equal(ids, ["NA00001", "NA00003"])
        nt.assert_array_equal(selection, [0, 2])
        assert "UNKNOWN" in caplog.text

    def test_ignore_missing_samples_all_unknown(self, caplog):
        with caplog.at_level(logging.WARNING, logger="vcztools.samples"):
            ids, selection = parse_samples(
                ["UNKNOWN1", "UNKNOWN2"], ALL_SAMPLES, ignore_missing_samples=True
            )
        nt.assert_array_equal(ids, [])
        nt.assert_array_equal(selection, [])


class TestParseSamplesMissingHeader:
    ALL_SAMPLES_WITH_MISSING = np.array(["NA00001", "", "NA00003"])

    def test_none_skips_missing(self):
        ids, selection = parse_samples(None, self.ALL_SAMPLES_WITH_MISSING)
        nt.assert_array_equal(ids, ["NA00001", "NA00003"])
        nt.assert_array_equal(selection, [0, 2])

    def test_explicit_list_with_missing_header(self):
        ids, selection = parse_samples(
            ["NA00001", "NA00003"], self.ALL_SAMPLES_WITH_MISSING
        )
        nt.assert_array_equal(ids, ["NA00001", "NA00003"])
        nt.assert_array_equal(selection, [0, 2])

    def test_explicit_single_sample_with_missing_header(self):
        ids, selection = parse_samples(["NA00003"], self.ALL_SAMPLES_WITH_MISSING)
        nt.assert_array_equal(ids, ["NA00003"])
        nt.assert_array_equal(selection, [2])

    def test_explicit_list_plus_complement_with_missing_header(self):
        # Exclude NA00001; the masked "" entry must also be dropped from
        # the resulting selection.
        ids, selection = parse_samples(
            ["NA00001"], self.ALL_SAMPLES_WITH_MISSING, complement=True
        )
        nt.assert_array_equal(ids, ["NA00003"])
        nt.assert_array_equal(selection, [2])


class TestParseSamplesEdgeCases:
    """Probe handling of unusual-but-valid sample IDs.

    bcftools (htslib) treats sample IDs on the ``#CHROM`` line as opaque
    byte strings. The VCF 4.3 spec constrains them only via the field
    structure of the line: tabs, newlines and carriage returns are
    forbidden because they delimit fields and records; every other
    character is legal.

    Valid in bcftools 1.19 (confirmed empirically):
      - ASCII letters, digits, and most punctuation.
      - Dots, dashes, underscores, forward and back slashes.
      - Embedded spaces (``"a b"``).
      - Embedded commas (``"a,b"``) — preserved on the ``#CHROM`` line
        by ``bcftools view``, but cannot be targeted by ``-s`` because
        the CLI splits that argument on commas. Same constraint applies
        to vcztools' CLI; ``parse_samples`` itself treats the comma as
        an opaque character.
      - Numeric-only names (``"123"``, ``"0"``).
      - Unicode / non-ASCII (accented letters, CJK glyphs).
      - Very long names (no practical limit observed).
      - Case-sensitive (``"x"`` and ``"X"`` are distinct).

    Invalid:
      - Tab, newline, carriage return (break VCF structure).
      - Empty string ``""`` (silently dropped from header parse by
        bcftools; rejected as a ``-s`` target — see ``TestParseSamples*``
        tests for vcztools' matching behaviour).
      - Duplicates in the header (htslib errors at parse time).

    These tests pin down that ``parse_samples`` treats IDs as opaque
    strings across the categories bcftools accepts.
    """

    def test_embedded_space(self):
        all_samples = np.array(["a b", "c d", "e f"])
        ids, selection = parse_samples(["a b"], all_samples)
        nt.assert_array_equal(ids, ["a b"])
        nt.assert_array_equal(selection, [0])

    def test_embedded_comma(self):
        # Commas are valid in the VCF file (bcftools view preserves them
        # byte-for-byte on #CHROM) but break CLI -s splitting. At the
        # parse_samples layer the comma is just another character.
        all_samples = np.array(["a,b", "c,d", "plain"])
        ids, selection = parse_samples(["a,b"], all_samples)
        nt.assert_array_equal(ids, ["a,b"])
        nt.assert_array_equal(selection, [0])

    def test_dots_and_dashes(self):
        all_samples = np.array(["sample.1", "sample-1", "sample_1"])
        ids, selection = parse_samples(["sample.1", "sample-1"], all_samples)
        nt.assert_array_equal(ids, ["sample.1", "sample-1"])
        nt.assert_array_equal(selection, [0, 1])

    def test_slashes(self):
        all_samples = np.array(["s/1", "s\\2", "s|3"])
        ids, selection = parse_samples(["s\\2"], all_samples)
        nt.assert_array_equal(ids, ["s\\2"])
        nt.assert_array_equal(selection, [1])

    def test_numeric_only(self):
        all_samples = np.array(["0", "1", "123"])
        ids, selection = parse_samples(["123", "0"], all_samples)
        nt.assert_array_equal(ids, ["123", "0"])
        nt.assert_array_equal(selection, [2, 0])

    def test_single_character(self):
        all_samples = np.array(["A", "B", "C"])
        ids, selection = parse_samples(["B"], all_samples)
        nt.assert_array_equal(ids, ["B"])
        nt.assert_array_equal(selection, [1])

    def test_unicode_accents(self):
        all_samples = np.array(["sampléé", "sample", "café"])
        ids, selection = parse_samples(["sampléé", "café"], all_samples)
        nt.assert_array_equal(ids, ["sampléé", "café"])
        nt.assert_array_equal(selection, [0, 2])

    def test_unicode_cjk(self):
        all_samples = np.array(["sample_中文", "sample_日本", "sample"])
        ids, selection = parse_samples(["sample_日本"], all_samples)
        nt.assert_array_equal(ids, ["sample_日本"])
        nt.assert_array_equal(selection, [1])

    def test_very_long_names(self):
        long_name = "x" * 300
        all_samples = np.array(["short", long_name, "other"])
        ids, selection = parse_samples([long_name], all_samples)
        nt.assert_array_equal(ids, [long_name])
        nt.assert_array_equal(selection, [1])

    def test_case_sensitive(self):
        # bcftools is case-sensitive; "x" is unknown when header only has "X".
        all_samples = np.array(["X", "Y", "Z"])
        with pytest.raises(
            ValueError, match=r"subset called for sample\(s\) not in header: x"
        ):
            parse_samples(["x"], all_samples)

    def test_complement_with_unicode(self):
        all_samples = np.array(["sampléé", "sample_中文", "ascii"])
        ids, selection = parse_samples(["sampléé"], all_samples, complement=True)
        nt.assert_array_equal(ids, ["sample_中文", "ascii"])
        nt.assert_array_equal(selection, [1, 2])

    def test_duplicate_detection_with_unicode(self):
        all_samples = np.array(["sampléé", "café"])
        with pytest.raises(ValueError, match=r'Duplicate sample name "sampléé"'):
            parse_samples(["sampléé", "sampléé"], all_samples)

    def test_ignore_missing_with_unicode_unknown(self, caplog):
        all_samples = np.array(["sampléé", "café"])
        with caplog.at_level(logging.WARNING, logger="vcztools.samples"):
            ids, selection = parse_samples(
                ["sampléé", "日本"],
                all_samples,
                ignore_missing_samples=True,
            )
        nt.assert_array_equal(ids, ["sampléé"])
        nt.assert_array_equal(selection, [0])
        assert "日本" in caplog.text


class TestReadSamplesFile:
    def test_basic(self):
        assert read_samples_file("tests/data/txt/samples.txt") == [
            "NA00001",
            "NA00003",
        ]

    def test_ignores_blank_lines(self, tmp_path):
        f = tmp_path / "samples.txt"
        f.write_text("NA00001\n\nNA00003\n\n")
        assert read_samples_file(str(f)) == ["NA00001", "NA00003"]

    def test_strips_whitespace(self, tmp_path):
        f = tmp_path / "samples.txt"
        f.write_text("  NA00001\nNA00003  \n")
        assert read_samples_file(str(f)) == ["NA00001", "NA00003"]

    def test_preserves_unicode_and_embedded_spaces(self, tmp_path):
        # strip() only touches leading/trailing whitespace, so internal
        # spaces survive. UTF-8 is the default open() encoding.
        f = tmp_path / "samples.txt"
        f.write_text("sampléé\na b\nsample_中文\n", encoding="utf-8")
        assert read_samples_file(str(f)) == ["sampléé", "a b", "sample_中文"]


class TestRoundTripEdgeCaseSamples:
    """Confirm edge-case sample IDs survive zarr → VCF → parse end-to-end.

    Covers the categories enumerated in ``TestParseSamplesEdgeCases`` in
    one shot, to confirm the writer path (``#CHROM`` line emission)
    preserves them byte-for-byte and a real VCF parser can read them
    back.
    """

    def test_writer_preserves_edge_case_ids(self, tmp_path):
        cyvcf2 = pytest.importorskip("cyvcf2")
        sample_ids = [
            "a b",
            "a,b",
            "sample.1",
            "sample-1",
            "123",
            "sampléé",
            "sample_中文",
            "X",
        ]
        num_samples = len(sample_ids)
        root = make_vcz(
            variant_contig=[0],
            variant_position=[1],
            alleles=[["A", "T"]],
            num_samples=num_samples,
            sample_id=sample_ids,
            call_genotype=np.zeros((1, num_samples, 2), dtype=np.int8),
        )
        reader = VczReader(root)
        output_path = tmp_path / "out.vcf"
        write_vcf(reader, output_path, no_version=True)

        v = cyvcf2.VCF(output_path)
        assert v.samples == sample_ids


def test_parse_samples_returns_numpy_arrays():
    """sample_ids is a numpy array; selection is always a numpy int array."""
    sample_ids, selection = parse_samples(["NA00001"], ALL_SAMPLES)
    assert isinstance(sample_ids, np.ndarray)
    assert isinstance(selection, np.ndarray)
    assert np.issubdtype(selection.dtype, np.integer)

    _, selection_default = parse_samples(None, ALL_SAMPLES)
    assert isinstance(selection_default, np.ndarray)
    assert np.issubdtype(selection_default.dtype, np.integer)


def test_parse_samples_accepts_numpy_string_dtype():
    """all_samples may arrive as a numpy StringDType array (zarr output)."""
    all_samples = np.array(
        ["NA00001", "NA00002", "NA00003"], dtype=np.dtypes.StringDType()
    )
    ids, selection = parse_samples(["NA00002"], all_samples)
    nt.assert_array_equal(ids, ["NA00002"])
    nt.assert_array_equal(selection, [1])
