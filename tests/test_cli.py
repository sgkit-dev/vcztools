import pathlib
import re
from unittest import mock

import click.testing as ct
import pytest

import vcztools.cli as cli
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


class TestOutput:
    def test_view_unsupported_output(self, tmp_path, vcz_path):
        bad_output = tmp_path / "output.vcf.gz"

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Only uncompressed VCF output supported, suffix .gz not allowed"
            ),
        ):
            run_vcztools(f"view --no-version {vcz_path} -o {bad_output}")

    @pytest.mark.parametrize("suffix", ["gz", "bgz", "bcf"])
    def test_view_unsupported_output_suffix(self, tmp_path, vcz_path, suffix):
        bad_output = tmp_path / f"output.vcf.{suffix}"

        with pytest.raises(ValueError, match=f".{suffix} not allowed"):
            run_vcztools(f"view --no-version {vcz_path} -o {bad_output}")

    def test_view_good_path(self, tmp_path, vcz_path):
        output_path = tmp_path / "tmp.vcf"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcztools_main,
            f"view --no-version {vcz_path} -o {output_path}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert output_path.exists()

    def test_view_write_directory(self, tmp_path, vcz_path):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcztools_main,
            f"view --no-version {vcz_path} -o {tmp_path}",
            catch_exceptions=False,
        )
        assert result.exit_code == 1
        assert len(result.stdout) == 0
        assert "Is a directory" in result.stderr

    def test_view_write_pipe(self, tmp_path, vcz_path):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.vcztools_main,
            f"view --no-version {vcz_path} -o {tmp_path}",
            catch_exceptions=False,
        )
        assert result.exit_code == 1
        assert len(result.stdout) == 0
        assert "Is a directory" in result.stderr


def test_excluding_and_including_samples(vcz_path):
    samples_file_path = pathlib.Path("tests/data/txt/samples.txt")
    error_message = re.escape("vcztools does not support combining -s and -S")

    with pytest.raises(AssertionError, match=error_message):
        run_vcztools(f"view {vcz_path} -s NA00001 -S ^{samples_file_path}")
    with pytest.raises(AssertionError, match=error_message):
        run_vcztools(f"view {vcz_path} -s ^NA00001 -S {samples_file_path}")


@mock.patch("sys.exit")
@mock.patch("os.dup2")
def test_broken_pipe(mocked_dup2, mocked_exit, tmp_path):
    with open(tmp_path / "tmp.txt", "w") as output:
        with cli.handle_broken_pipe(output):
            raise BrokenPipeError()
        mocked_dup2.assert_called_once()
        mocked_exit.assert_called_once_with(1)
