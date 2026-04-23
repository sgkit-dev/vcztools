import logging

import numpy as np
import numpy.testing as nt
import pytest

from vcztools.retrieval import VczReader
from vcztools.samples import (
    build_chunk_plan,
    read_samples_file,
    resolve_sample_selection,
)
from vcztools.vcf_writer import write_vcf

from .vcz_builder import make_vcz

ALL_SAMPLES = np.array(["NA00001", "NA00002", "NA00003"])


class TestResolveSampleSelection:
    """Bcftools-layer helper: turns ``-s``/``-S``/``--force-samples``
    into the integer-index list that VczReader consumes."""

    def test_none_and_no_complement_returns_none(self):
        assert resolve_sample_selection(None, ALL_SAMPLES) is None

    def test_none_and_complement_returns_all_non_null_samples(self):
        assert resolve_sample_selection(None, ALL_SAMPLES, complement=True) == [
            0,
            1,
            2,
        ]

    def test_none_and_complement_skips_masked(self):
        raw_sample_ids = np.array(["NA00001", "", "NA00003"])
        assert resolve_sample_selection(None, raw_sample_ids, complement=True) == [0, 2]

    def test_single_sample(self):
        assert resolve_sample_selection(["NA00002"], ALL_SAMPLES) == [1]

    def test_subset_in_input_order(self):
        assert resolve_sample_selection(["NA00003", "NA00001"], ALL_SAMPLES) == [2, 0]

    def test_full_list(self):
        assert resolve_sample_selection(
            ["NA00001", "NA00002", "NA00003"], ALL_SAMPLES
        ) == [0, 1, 2]

    def test_empty_list(self):
        assert resolve_sample_selection([], ALL_SAMPLES) == []

    def test_empty_string_element_raises(self):
        with pytest.raises(ValueError, match=r"sample\(s\) not in header: "):
            resolve_sample_selection([""], ALL_SAMPLES)

    def test_complement_excludes_one(self):
        assert resolve_sample_selection(["NA00002"], ALL_SAMPLES, complement=True) == [
            0,
            2,
        ]

    def test_complement_excludes_multiple(self):
        assert resolve_sample_selection(
            ["NA00001", "NA00003"], ALL_SAMPLES, complement=True
        ) == [1]

    def test_complement_exclude_all_returns_empty(self):
        assert (
            resolve_sample_selection(
                ["NA00001", "NA00002", "NA00003"], ALL_SAMPLES, complement=True
            )
            == []
        )

    def test_complement_empty_list_returns_all(self):
        assert resolve_sample_selection([], ALL_SAMPLES, complement=True) == [
            0,
            1,
            2,
        ]

    def test_complement_output_follows_header_order(self):
        assert resolve_sample_selection(
            ["NA00003", "NA00002"], ALL_SAMPLES, complement=True
        ) == [0]
        assert resolve_sample_selection(
            ["NA00002", "NA00003"], ALL_SAMPLES, complement=True
        ) == [0]

    def test_duplicates_raise_without_complement(self):
        with pytest.raises(ValueError, match=r'Duplicate sample name "NA00001"'):
            resolve_sample_selection(["NA00001", "NA00001"], ALL_SAMPLES)

    def test_duplicates_collapsed_under_complement(self):
        assert resolve_sample_selection(
            ["NA00002", "NA00002"], ALL_SAMPLES, complement=True
        ) == [0, 2]

    def test_many_duplicates_collapsed_under_complement(self):
        assert resolve_sample_selection(
            ["NA00002", "NA00002", "NA00002"], ALL_SAMPLES, complement=True
        ) == [0, 2]

    def test_unknown_raises(self):
        with pytest.raises(
            ValueError,
            match=(
                r"subset called for sample\(s\) not in header: UNKNOWN\. "
                r'Use "--force-samples" to ignore this error\.'
            ),
        ):
            resolve_sample_selection(["NA00001", "UNKNOWN"], ALL_SAMPLES)

    def test_message_lists_every_unknown(self):
        with pytest.raises(ValueError, match="UNKNOWN1") as excinfo:
            resolve_sample_selection(["UNKNOWN1", "UNKNOWN2", "NA00001"], ALL_SAMPLES)
        assert "UNKNOWN2" in str(excinfo.value)

    def test_ignore_missing_drops_unknowns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="vcztools.samples"):
            result = resolve_sample_selection(
                ["NA00001", "UNKNOWN", "NA00003"],
                ALL_SAMPLES,
                ignore_missing_samples=True,
            )
        assert result == [0, 2]
        assert "UNKNOWN" in caplog.text

    def test_ignore_missing_all_unknown(self, caplog):
        with caplog.at_level(logging.WARNING, logger="vcztools.samples"):
            result = resolve_sample_selection(
                ["UNKNOWN1", "UNKNOWN2"],
                ALL_SAMPLES,
                ignore_missing_samples=True,
            )
        assert result == []

    def test_duplicates_raise_even_with_ignore_missing(self):
        # Duplicates are a hard error regardless of force-samples.
        with pytest.raises(ValueError, match=r'Duplicate sample name "NA00001"'):
            resolve_sample_selection(
                ["NA00001", "NA00001", "UNKNOWN"],
                ALL_SAMPLES,
                ignore_missing_samples=True,
            )

    def test_empty_string_element_with_ignore_missing(self):
        # bcftools -s '' --force-samples → no samples.
        result = resolve_sample_selection(
            [""], ALL_SAMPLES, ignore_missing_samples=True
        )
        assert result == []

    def test_complement_with_masked_header(self):
        # Masked "" entries never appear in the complement output.
        raw_sample_ids = np.array(["NA00001", "", "NA00003"])
        assert resolve_sample_selection(
            ["NA00001"], raw_sample_ids, complement=True
        ) == [2]

    def test_explicit_list_with_masked_header(self):
        raw_sample_ids = np.array(["NA00001", "", "NA00003"])
        assert resolve_sample_selection(["NA00003"], raw_sample_ids) == [2]


class TestResolveSampleSelectionEdgeCases:
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
        to vcztools' CLI; the resolver itself treats the comma as an
        opaque character.
      - Numeric-only names (``"123"``, ``"0"``).
      - Unicode / non-ASCII (accented letters, CJK glyphs).
      - Very long names (no practical limit observed).
      - Case-sensitive (``"x"`` and ``"X"`` are distinct).

    Invalid:
      - Tab, newline, carriage return (break VCF structure).
      - Empty string ``""`` (silently dropped from header parse by
        bcftools; rejected as a ``-s`` target — see
        ``test_empty_string_element_raises`` above).
      - Duplicates in the header (htslib errors at parse time).
    """

    def test_embedded_space(self):
        raw_sample_ids = np.array(["a b", "c d", "e f"])
        assert resolve_sample_selection(["a b"], raw_sample_ids) == [0]

    def test_embedded_comma(self):
        # Commas are valid in the VCF file (bcftools view preserves them
        # byte-for-byte on #CHROM) but break CLI -s splitting. At the
        # resolver layer the comma is just another character.
        raw_sample_ids = np.array(["a,b", "c,d", "plain"])
        assert resolve_sample_selection(["a,b"], raw_sample_ids) == [0]

    def test_dots_and_dashes(self):
        raw_sample_ids = np.array(["sample.1", "sample-1", "sample_1"])
        assert resolve_sample_selection(["sample.1", "sample-1"], raw_sample_ids) == [
            0,
            1,
        ]

    def test_slashes(self):
        raw_sample_ids = np.array(["s/1", "s\\2", "s|3"])
        assert resolve_sample_selection(["s\\2"], raw_sample_ids) == [1]

    def test_numeric_only(self):
        raw_sample_ids = np.array(["0", "1", "123"])
        assert resolve_sample_selection(["123", "0"], raw_sample_ids) == [2, 0]

    def test_single_character(self):
        raw_sample_ids = np.array(["A", "B", "C"])
        assert resolve_sample_selection(["B"], raw_sample_ids) == [1]

    def test_unicode_accents(self):
        raw_sample_ids = np.array(["sampléé", "sample", "café"])
        assert resolve_sample_selection(["sampléé", "café"], raw_sample_ids) == [0, 2]

    def test_unicode_cjk(self):
        raw_sample_ids = np.array(["sample_中文", "sample_日本", "sample"])
        assert resolve_sample_selection(["sample_日本"], raw_sample_ids) == [1]

    def test_very_long_names(self):
        long_name = "x" * 300
        raw_sample_ids = np.array(["short", long_name, "other"])
        assert resolve_sample_selection([long_name], raw_sample_ids) == [1]

    def test_case_sensitive(self):
        raw_sample_ids = np.array(["X", "Y", "Z"])
        with pytest.raises(
            ValueError, match=r"subset called for sample\(s\) not in header: x"
        ):
            resolve_sample_selection(["x"], raw_sample_ids)

    def test_complement_with_unicode(self):
        raw_sample_ids = np.array(["sampléé", "sample_中文", "ascii"])
        assert resolve_sample_selection(
            ["sampléé"], raw_sample_ids, complement=True
        ) == [
            1,
            2,
        ]

    def test_duplicate_detection_with_unicode(self):
        raw_sample_ids = np.array(["sampléé", "café"])
        with pytest.raises(ValueError, match=r'Duplicate sample name "sampléé"'):
            resolve_sample_selection(["sampléé", "sampléé"], raw_sample_ids)

    def test_ignore_missing_with_unicode_unknown(self, caplog):
        raw_sample_ids = np.array(["sampléé", "café"])
        with caplog.at_level(logging.WARNING, logger="vcztools.samples"):
            result = resolve_sample_selection(
                ["sampléé", "日本"],
                raw_sample_ids,
                ignore_missing_samples=True,
            )
        assert result == [0]
        assert "日本" in caplog.text

    def test_accepts_numpy_string_dtype(self):
        raw_sample_ids = np.array(
            ["NA00001", "NA00002", "NA00003"], dtype=np.dtypes.StringDType()
        )
        assert resolve_sample_selection(["NA00002"], raw_sample_ids) == [1]


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

    def test_preserves_embedded_commas(self, tmp_path):
        # The samples-file reader is line-oriented (no comma splitting),
        # so "a,b" on a line survives as a single ID — unlike the CLI's
        # -s argument, which does split on commas.
        f = tmp_path / "samples.txt"
        f.write_text("a,b\nc,d,e\nplain\n")
        assert read_samples_file(str(f)) == ["a,b", "c,d,e", "plain"]


class TestRoundTripEdgeCaseSamples:
    """Confirm edge-case sample IDs survive zarr → VCF → parse end-to-end.

    Covers the categories enumerated in
    ``TestResolveSampleSelectionEdgeCases`` in one shot, to confirm the
    writer path (``#CHROM`` line emission) preserves them byte-for-byte
    and a real VCF parser can read them back.
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


class TestBuildChunkPlan:
    """Translate a global sample selection into a sample-chunk read plan."""

    @staticmethod
    def _chunk_indexes(plan):
        return [cr.index for cr in plan.chunk_reads]

    def test_single_sample_middle_chunk(self):
        # num_samples=10, chunks of 3: [0-2][3-5][6-8][9]. Sample 4 is in chunk 1.
        plan = build_chunk_plan(np.array([4]), samples_chunk_size=3)
        assert self._chunk_indexes(plan) == [1]
        nt.assert_array_equal(plan.chunk_reads[0].selection, [1])
        assert plan.permutation is None

    def test_samples_spanning_two_chunks_with_gap(self):
        # Samples 1 and 7 sit in chunks 0 and 2; chunk 1 is skipped.
        # Sorted input → no permutation needed.
        plan = build_chunk_plan(np.array([1, 7]), samples_chunk_size=3)
        assert self._chunk_indexes(plan) == [0, 2]
        nt.assert_array_equal(plan.chunk_reads[0].selection, [1])
        nt.assert_array_equal(plan.chunk_reads[1].selection, [1])
        assert plan.permutation is None

    def test_partial_final_chunk(self):
        # num_samples=10, chunk size 3 → last chunk starts at 9.
        plan = build_chunk_plan(np.array([9]), samples_chunk_size=3)
        assert self._chunk_indexes(plan) == [3]
        nt.assert_array_equal(plan.chunk_reads[0].selection, [0])
        assert plan.permutation is None

    def test_preserves_user_order(self):
        # selection order [7, 1] must round-trip through the plan.
        # Chunks 0 (sample 1) and 2 (sample 7); concat in chunk order
        # gives [sample_1, sample_7]. Permutation [1, 0] flips back.
        plan = build_chunk_plan(np.array([7, 1]), samples_chunk_size=3)
        assert self._chunk_indexes(plan) == [0, 2]
        nt.assert_array_equal(plan.chunk_reads[0].selection, [1])
        nt.assert_array_equal(plan.chunk_reads[1].selection, [1])
        nt.assert_array_equal(plan.permutation, [1, 0])

    def test_within_chunk_user_order_uses_no_permutation(self):
        # Reversed samples that live in the same chunk: per-chunk
        # selection captures the order, permutation stays None.
        plan = build_chunk_plan(np.array([2, 0]), samples_chunk_size=3)
        assert self._chunk_indexes(plan) == [0]
        nt.assert_array_equal(plan.chunk_reads[0].selection, [2, 0])
        assert plan.permutation is None

    def test_chunk_size_one(self):
        # Each sample is its own chunk.
        plan = build_chunk_plan(np.array([0, 3, 5]), samples_chunk_size=1)
        assert self._chunk_indexes(plan) == [0, 3, 5]
        for cr in plan.chunk_reads:
            nt.assert_array_equal(cr.selection, [0])
        assert plan.permutation is None

    def test_single_chunk_covers_raw_sample_ids(self):
        # samples_chunk_size >= num_samples → one chunk.
        plan = build_chunk_plan(np.array([0, 2]), samples_chunk_size=3)
        assert self._chunk_indexes(plan) == [0]
        nt.assert_array_equal(plan.chunk_reads[0].selection, [0, 2])
        assert plan.permutation is None

    def test_empty_selection(self):
        plan = build_chunk_plan(np.array([], dtype=np.int64), samples_chunk_size=3)
        assert plan.chunk_reads == []
        assert plan.permutation is None

    def test_duplicate_samples_preserved(self):
        # Duplicate global indices are passed through as duplicate local indices.
        plan = build_chunk_plan(np.array([1, 1]), samples_chunk_size=3)
        assert self._chunk_indexes(plan) == [0]
        nt.assert_array_equal(plan.chunk_reads[0].selection, [1, 1])
        assert plan.permutation is None
