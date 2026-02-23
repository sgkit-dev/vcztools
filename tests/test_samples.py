import numpy as np
import numpy.testing as nt
import pytest

from vcztools.samples import parse_samples, parse_samples_file


@pytest.mark.parametrize(
    ("samples", "all_samples", "expected_sample_ids", "expected_samples_selection"),
    [
        (
            None,  # return all samples
            np.array(["NA00001", "NA00002", "NA00003"]),
            ["NA00001", "NA00002", "NA00003"],
            None,
        ),
        (
            "NA00002,NA00003",  # samples to include
            np.array(["NA00001", "NA00002", "NA00003"]),
            ["NA00002", "NA00003"],
            [1, 2],
        ),
        (
            "^NA00002,NA00003",  # samples to exclude
            np.array(["NA00001", "NA00002", "NA00003"]),
            ["NA00001"],
            [0],
        ),
        (
            None,
            np.array(["NA00001", "", "NA00003"]),  # ignore missing
            ["NA00001", "NA00003"],
            [0, 2],
        ),
    ],
)
def test_parse_samples(
    samples, all_samples, expected_sample_ids, expected_samples_selection
):
    sample_ids, samples_selection = parse_samples(samples, all_samples)

    nt.assert_array_equal(sample_ids, expected_sample_ids)
    nt.assert_array_equal(samples_selection, expected_samples_selection)


def test_parse_samples_file():
    nt.assert_array_equal(
        parse_samples_file("tests/data/txt/samples.txt"), "NA00001,NA00003"
    )
    nt.assert_array_equal(
        parse_samples_file("^tests/data/txt/samples.txt"), "^NA00001,NA00003"
    )
