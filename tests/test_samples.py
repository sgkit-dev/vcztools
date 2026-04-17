import logging

import numpy as np
import numpy.testing as nt
import pytest

from vcztools.samples import parse_samples, read_samples_file

ALL_SAMPLES = np.array(["NA00001", "NA00002", "NA00003"])


def _call(samples, all_samples=ALL_SAMPLES, **kwargs):
    """Invoke ``parse_samples`` and return ``(list[str], selection)``.

    ``selection`` is returned as a list for readable assertions unless it is
    ``None`` (the "no subsetting" signal).
    """
    sample_ids, selection = parse_samples(samples, all_samples, **kwargs)
    return list(sample_ids), None if selection is None else list(selection)


class TestParseSamplesNone:
    def test_default_returns_all_samples(self):
        ids, selection = _call(None)
        assert ids == ["NA00001", "NA00002", "NA00003"]
        assert selection is None

    def test_complement_has_no_effect(self):
        # samples=None exits before the complement branch; the flag is a no-op.
        ids, selection = _call(None, complement=True)
        assert ids == ["NA00001", "NA00002", "NA00003"]
        assert selection is None

    def test_skips_missing_header_samples(self):
        all_samples = np.array(["NA00001", "", "NA00003"])
        ids, selection = _call(None, all_samples=all_samples)
        assert ids == ["NA00001", "NA00003"]
        assert selection == [0, 2]


class TestParseSamplesList:
    def test_single_sample(self):
        ids, selection = _call(["NA00002"])
        assert ids == ["NA00002"]
        assert selection == [1]

    def test_subset_in_input_order(self):
        # Order of the input list is preserved in the output.
        ids, selection = _call(["NA00003", "NA00001"])
        assert ids == ["NA00003", "NA00001"]
        assert selection == [2, 0]

    def test_full_list(self):
        ids, selection = _call(["NA00001", "NA00002", "NA00003"])
        assert ids == ["NA00001", "NA00002", "NA00003"]
        assert selection == [0, 1, 2]

    def test_empty_list(self):
        ids, selection = _call([])
        assert ids == []
        assert selection == []

    def test_empty_string_element(self):
        # [""] is a special input representing "no samples".
        ids, selection = _call([""])
        assert ids == []
        assert selection == []


class TestParseSamplesDuplicates:
    def test_duplicates_preserved(self):
        ids, selection = _call(["NA00001", "NA00001"])
        assert ids == ["NA00001", "NA00001"]
        assert selection == [0, 0]

    def test_duplicates_collapsed_under_complement(self):
        ids, selection = _call(["NA00002", "NA00002"], complement=True)
        assert ids == ["NA00001", "NA00003"]
        assert selection == [0, 2]


class TestParseSamplesComplement:
    def test_excludes_one(self):
        ids, selection = _call(["NA00002"], complement=True)
        assert ids == ["NA00001", "NA00003"]
        assert selection == [0, 2]

    def test_excludes_multiple(self):
        ids, selection = _call(["NA00001", "NA00003"], complement=True)
        assert ids == ["NA00002"]
        assert selection == [1]

    def test_exclude_all_returns_empty(self):
        ids, selection = _call(["NA00001", "NA00002", "NA00003"], complement=True)
        assert ids == []
        assert selection == []

    def test_empty_list_complement_returns_all(self):
        ids, selection = _call([], complement=True)
        assert ids == ["NA00001", "NA00002", "NA00003"]
        assert selection == [0, 1, 2]

    def test_output_follows_all_samples_order(self):
        # Input order is irrelevant under complement; output follows header order.
        ids_reversed, selection_reversed = _call(
            ["NA00003", "NA00002"], complement=True
        )
        ids_forward, selection_forward = _call(["NA00002", "NA00003"], complement=True)
        assert ids_reversed == ids_forward == ["NA00001"]
        assert selection_reversed == selection_forward == [0]


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

    def test_force_samples_drops_unknowns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="vcztools.samples"):
            ids, selection = _call(
                ["NA00001", "UNKNOWN", "NA00003"], force_samples=True
            )
        assert ids == ["NA00001", "NA00003"]
        assert selection == [0, 2]
        assert "UNKNOWN" in caplog.text

    def test_force_samples_all_unknown(self, caplog):
        with caplog.at_level(logging.WARNING, logger="vcztools.samples"):
            ids, selection = _call(["UNKNOWN1", "UNKNOWN2"], force_samples=True)
        assert ids == []
        assert selection == []


class TestParseSamplesMissingHeader:
    ALL_SAMPLES_WITH_MISSING = np.array(["NA00001", "", "NA00003"])

    def test_none_skips_missing(self):
        ids, selection = _call(None, all_samples=self.ALL_SAMPLES_WITH_MISSING)
        assert ids == ["NA00001", "NA00003"]
        assert selection == [0, 2]

    def test_explicit_list_with_missing_header(self):
        ids, selection = _call(
            ["NA00001", "NA00003"], all_samples=self.ALL_SAMPLES_WITH_MISSING
        )
        assert ids == ["NA00001", "NA00003"]
        assert selection == [0, 2]

    def test_explicit_single_sample_with_missing_header(self):
        ids, selection = _call(["NA00003"], all_samples=self.ALL_SAMPLES_WITH_MISSING)
        assert ids == ["NA00003"]
        assert selection == [2]

    def test_explicit_list_plus_complement_with_missing_header(self):
        # Exclude NA00001; the masked "" entry must also be dropped from
        # the resulting selection.
        ids, selection = _call(
            ["NA00001"],
            all_samples=self.ALL_SAMPLES_WITH_MISSING,
            complement=True,
        )
        assert ids == ["NA00003"]
        assert selection == [2]


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


def test_parse_samples_returns_numpy_arrays():
    """sample_ids is a numpy array; selection is a numpy int array (or None)."""
    sample_ids, selection = parse_samples(["NA00001"], ALL_SAMPLES)
    assert isinstance(sample_ids, np.ndarray)
    assert isinstance(selection, np.ndarray)
    assert np.issubdtype(selection.dtype, np.integer)

    _, selection_none = parse_samples(None, ALL_SAMPLES)
    assert selection_none is None


def test_parse_samples_accepts_numpy_string_dtype():
    """all_samples may arrive as a numpy StringDType array (zarr output)."""
    all_samples = np.array(
        ["NA00001", "NA00002", "NA00003"], dtype=np.dtypes.StringDType()
    )
    ids, selection = parse_samples(["NA00002"], all_samples)
    nt.assert_array_equal(ids, ["NA00002"])
    nt.assert_array_equal(selection, [1])
