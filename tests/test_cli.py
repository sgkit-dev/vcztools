import pathlib
from unittest import mock

import click.testing as ct
import pytest

import vcztools.cli as cli
from tests.test_bcftools_validation import run_vcztools
from tests.utils import vcz_path_cache
from vcztools import provenance


@pytest.fixture()
def vcz_path():
    vcf_path = pathlib.Path("tests/data/vcf/sample.vcf.gz")
    return vcz_path_cache(vcf_path)


def test_version_header(vcz_path):
    output, _ = run_vcztools(f"view {vcz_path}")
    assert output.find("##vcztools_viewCommand=") >= 0
    assert output.find("Date=") >= 0


class TestOutput:
    def test_view_unsupported_output(self, tmp_path, vcz_path):
        bad_output = tmp_path / "output.vcf.gz"

        _, vcztools_error = run_vcztools(
            f"view --no-version {vcz_path} -o {bad_output}", expect_error=True
        )
        assert (
            "Only uncompressed VCF output supported, suffix .gz not allowed"
            in vcztools_error
        )

    @pytest.mark.parametrize("suffix", ["gz", "bgz", "bcf"])
    def test_view_unsupported_output_suffix(self, tmp_path, vcz_path, suffix):
        bad_output = tmp_path / f"output.vcf.{suffix}"

        _, vcztools_error = run_vcztools(
            f"view --no-version {vcz_path} -o {bad_output}", expect_error=True
        )
        assert f".{suffix} not allowed" in vcztools_error

    def test_view_good_path(self, tmp_path, vcz_path):
        output_path = tmp_path / "tmp.vcf"
        runner = ct.CliRunner()
        result = runner.invoke(
            cli.vcztools_main,
            f"view --no-version {vcz_path} -o {output_path}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert output_path.exists()

    def test_view_write_directory(self, tmp_path, vcz_path):
        runner = ct.CliRunner()
        result = runner.invoke(
            cli.vcztools_main,
            f"view --no-version {vcz_path} -o {tmp_path}",
            catch_exceptions=False,
        )
        assert result.exit_code == 1
        assert len(result.stdout) == 0
        assert "Is a directory" in result.stderr

    def test_view_write_pipe(self, tmp_path, vcz_path):
        runner = ct.CliRunner()
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
    error_message = "vcztools does not support combining -s and -S"

    _, vcztools_error = run_vcztools(
        f"view {vcz_path} -s NA00001 -S ^{samples_file_path}", expect_error=True
    )
    assert error_message in vcztools_error
    _, vcztools_error = run_vcztools(
        f"view {vcz_path} -s ^NA00001 -S {samples_file_path}", expect_error=True
    )
    assert error_message in vcztools_error


@mock.patch("sys.exit")
@mock.patch("os.dup2")
def test_broken_pipe(mocked_dup2, mocked_exit, tmp_path):
    with open(tmp_path / "tmp.txt", "w") as output:
        with cli.handle_broken_pipe(output):
            raise BrokenPipeError()
        mocked_dup2.assert_called_once()
        mocked_exit.assert_called_once_with(1)


class TestQuery:
    def test_format_required(self, vcz_path):
        runner = ct.CliRunner()
        result = runner.invoke(
            cli.vcztools_main,
            f"query {vcz_path} ",
            catch_exceptions=False,
        )
        assert result.exit_code != 0
        assert len(result.stdout) == 0
        assert len(result.stderr) > 0

    def test_path_required(self):
        runner = ct.CliRunner()
        result = runner.invoke(
            cli.vcztools_main,
            "query --format=POS ",
            catch_exceptions=False,
        )
        assert result.exit_code != 0
        assert len(result.stdout) == 0
        assert len(result.stderr) > 0

    def test_list(self, vcz_path):
        result, _ = run_vcztools(f"query -l {vcz_path}")
        assert list(result.splitlines()) == ["NA00001", "NA00002", "NA00003"]

    def test_list_ignores_output(self, vcz_path, tmp_path):
        output = tmp_path / "tmp.txt"
        result, _ = run_vcztools(f"query -l {vcz_path} -o {output}")
        assert list(result.splitlines()) == ["NA00001", "NA00002", "NA00003"]
        assert not output.exists()

    def test_output(self, vcz_path, tmp_path):
        output = tmp_path / "tmp.txt"
        result, _ = run_vcztools(f"query -f '%POS\n' {vcz_path} -o {output}")
        assert list(result.splitlines()) == []
        assert output.exists()


class TestIndex:
    def test_stats(self, vcz_path):
        result, _ = run_vcztools(f"index -s {vcz_path}")
        assert list(result.splitlines()) == ["19\t.\t2", "20\t.\t6", "X\t.\t1"]

    def test_nrecords(self, vcz_path):
        result, _ = run_vcztools(f"index -n {vcz_path}")
        assert list(result.splitlines()) == ["9"]

    def test_stats_and_nrecords(self, vcz_path):
        runner = ct.CliRunner()
        result = runner.invoke(
            cli.vcztools_main,
            f"index -ns {vcz_path}",
            catch_exceptions=False,
        )
        assert result.exit_code != 0
        assert len(result.stdout) == 0
        assert len(result.stderr) > 0
        assert "Expected only one of --stats or --nrecords options" in result.stderr

    def test_no_stats_or_nrecords(self, vcz_path):
        runner = ct.CliRunner()
        result = runner.invoke(
            cli.vcztools_main,
            f"index {vcz_path}",
            catch_exceptions=False,
        )
        assert result.exit_code != 0
        assert len(result.stdout) == 0
        assert len(result.stderr) > 0
        assert "Error: Building region indexes is not supported" in result.stderr


def test_top_level():
    runner = ct.CliRunner()
    result = runner.invoke(
        cli.vcztools_main,
        catch_exceptions=False,
    )
    assert result.exit_code != 0
    assert len(result.stdout) == 0
    assert len(result.stderr) > 0


def test_version():
    runner = ct.CliRunner()
    result = runner.invoke(cli.vcztools_main, ["--version"], catch_exceptions=False)
    s = f"version {provenance.__version__}\n"
    assert result.stdout.endswith(s)
