import pathlib
import re

import pytest

from tests.test_bcftools_validation import run_vcztools
from tests.utils import vcz_path_cache


@pytest.fixture()
def vcz_path():
    vcf_path = pathlib.Path("tests/data/vcf/sample.vcf.gz")
    return vcz_path_cache(vcf_path)


def test_version_header(vcz_path):
    output = run_vcztools(f"view {vcz_path}")
    assert output.find("##vcztools_viewCommand=") >= 0
    assert output.find("Date=") >= 0


def test_view_bad_output(tmp_path, vcz_path):
    bad_output = tmp_path / "output.vcf.gz"

    with pytest.raises(
        ValueError, match=re.escape("Output file extension must be .vcf, got: .gz")
    ):
        run_vcztools(f"view --no-version {vcz_path} -o {bad_output}")


def test_excluding_and_including_samples(vcz_path):
    samples_file_path = pathlib.Path("tests/data/txt/samples.txt")
    error_message = re.escape("vcztools does not support combining -s and -S")

    with pytest.raises(AssertionError, match=error_message):
        run_vcztools(f"view {vcz_path} -s NA00001 -S ^{samples_file_path}")
    with pytest.raises(AssertionError, match=error_message):
        run_vcztools(f"view {vcz_path} -s ^NA00001 -S {samples_file_path}")
