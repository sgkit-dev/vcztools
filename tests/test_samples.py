import numpy as np
import numpy.testing as nt
import pytest

from vcztools.samples import parse_samples, read_samples_file


@pytest.mark.parametrize(
    (
        "samples",
        "complement",
        "all_samples",
        "expected_sample_ids",
        "expected_samples_selection",
    ),
    [
        (
            None,
            False,  # return all samples
            np.array(["NA00001", "NA00002", "NA00003"]),
            ["NA00001", "NA00002", "NA00003"],
            None,
        ),
        (
            ["NA00002", "NA00003"],
            False,  # samples to include
            np.array(["NA00001", "NA00002", "NA00003"]),
            ["NA00002", "NA00003"],
            [1, 2],
        ),
        (
            ["NA00002", "NA00003"],
            True,  # samples to exclude
            np.array(["NA00001", "NA00002", "NA00003"]),
            ["NA00001"],
            [0],
        ),
        (
            None,
            False,
            np.array(["NA00001", "", "NA00003"]),  # ignore missing
            ["NA00001", "NA00003"],
            [0, 2],
        ),
    ],
)
def test_parse_samples(
    samples, complement, all_samples, expected_sample_ids, expected_samples_selection
):
    sample_ids, samples_selection = parse_samples(
        samples, all_samples, complement=complement
    )

    nt.assert_array_equal(sample_ids, expected_sample_ids)
    nt.assert_array_equal(samples_selection, expected_samples_selection)


def test_read_samples_file():
    assert read_samples_file("tests/data/txt/samples.txt") == ["NA00001", "NA00003"]


def test_read_samples_file_ignores_blank_lines(tmp_path):
    f = tmp_path / "samples.txt"
    f.write_text("NA00001\n\nNA00003\n\n")
    assert read_samples_file(str(f)) == ["NA00001", "NA00003"]
