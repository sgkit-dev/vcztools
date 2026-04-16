import pathlib
import sys
from unittest import mock

import click.testing as ct
import pytest

import vcztools.cli as cli
from tests.test_bcftools_validation import run_vcztools
from vcztools import provenance

IS_WINDOWS = sys.platform == "win32"


@pytest.fixture
def fx_vcz_path(fx_sample_vcz):
    # Return a posix-form string so f-string interpolation into CLI
    # arg strings is safe on Windows (CliRunner shlex-splits with
    # posix=True, which treats backslashes as escape characters).
    return fx_sample_vcz.zip_path.as_posix()


def test_version_header(fx_vcz_path):
    output, _ = run_vcztools(f"view {fx_vcz_path}")
    assert output.find("##vcztools_viewCommand=") >= 0
    assert output.find("Date=") >= 0


class TestOutput:
    def test_view_unsupported_output(self, tmp_path, fx_vcz_path):
        bad_output = tmp_path / "output.vcf.gz"

        _, vcztools_error = run_vcztools(
            f"view --no-version {fx_vcz_path} -o {bad_output.as_posix()}",
            expect_error=True,
        )
        assert (
            "Only uncompressed VCF output supported, suffix .gz not allowed"
            in vcztools_error
        )

    @pytest.mark.parametrize("suffix", ["gz", "bgz", "bcf"])
    def test_view_unsupported_output_suffix(self, tmp_path, fx_vcz_path, suffix):
        bad_output = tmp_path / f"output.vcf.{suffix}"

        _, vcztools_error = run_vcztools(
            f"view --no-version {fx_vcz_path} -o {bad_output.as_posix()}",
            expect_error=True,
        )
        assert f".{suffix} not allowed" in vcztools_error

    def test_view_good_path(self, tmp_path, fx_vcz_path):
        output_path = tmp_path / "tmp.vcf"
        runner = ct.CliRunner()
        result = runner.invoke(
            cli.vcztools_main,
            f"view --no-version {fx_vcz_path} -o {output_path.as_posix()}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert output_path.exists()

    def test_view_write_directory(self, tmp_path, fx_vcz_path):
        runner = ct.CliRunner()
        result = runner.invoke(
            cli.vcztools_main,
            f"view --no-version {fx_vcz_path} -o {tmp_path.as_posix()}",
            catch_exceptions=False,
        )
        assert result.exit_code == 1
        assert len(result.stdout) == 0
        expected = "Permission denied" if IS_WINDOWS else "Is a directory"
        assert expected in result.stderr

    def test_view_write_pipe(self, tmp_path, fx_vcz_path):
        runner = ct.CliRunner()
        result = runner.invoke(
            cli.vcztools_main,
            f"view --no-version {fx_vcz_path} -o {tmp_path.as_posix()}",
            catch_exceptions=False,
        )
        assert result.exit_code == 1
        assert len(result.stdout) == 0
        expected = "Permission denied" if IS_WINDOWS else "Is a directory"
        assert expected in result.stderr


def test_excluding_and_including_samples(fx_vcz_path):
    samples_file_path = pathlib.Path("tests/data/txt/samples.txt")
    error_message = "Cannot specify both a samples string (-s) and a samples file (-S)"

    _, vcztools_error = run_vcztools(
        f"view {fx_vcz_path} -s NA00001 -S ^{samples_file_path.as_posix()}",
        expect_error=True,
    )
    assert error_message in vcztools_error
    _, vcztools_error = run_vcztools(
        f"view {fx_vcz_path} -s ^NA00001 -S {samples_file_path.as_posix()}",
        expect_error=True,
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
    def test_format_required(self, fx_vcz_path):
        runner = ct.CliRunner()
        result = runner.invoke(
            cli.vcztools_main,
            f"query {fx_vcz_path} ",
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

    def test_list(self, fx_vcz_path):
        result, _ = run_vcztools(f"query -l {fx_vcz_path}")
        assert list(result.splitlines()) == ["NA00001", "NA00002", "NA00003"]

    def test_list_ignores_output(self, fx_vcz_path, tmp_path):
        output = tmp_path / "tmp.txt"
        result, _ = run_vcztools(f"query -l {fx_vcz_path} -o {output.as_posix()}")
        assert list(result.splitlines()) == ["NA00001", "NA00002", "NA00003"]
        assert not output.exists()

    def test_output(self, fx_vcz_path, tmp_path):
        output = tmp_path / "tmp.txt"
        result, _ = run_vcztools(
            f"query -f '%POS\n' {fx_vcz_path} -o {output.as_posix()}"
        )
        assert list(result.splitlines()) == []
        assert output.exists()


class TestIndex:
    def test_stats(self, fx_vcz_path):
        result, _ = run_vcztools(f"index -s {fx_vcz_path}")
        assert list(result.splitlines()) == ["19\t.\t2", "20\t.\t6", "X\t.\t1"]

    def test_nrecords(self, fx_vcz_path):
        result, _ = run_vcztools(f"index -n {fx_vcz_path}")
        assert list(result.splitlines()) == ["9"]

    def test_stats_and_nrecords(self, fx_vcz_path):
        runner = ct.CliRunner()
        result = runner.invoke(
            cli.vcztools_main,
            f"index -ns {fx_vcz_path}",
            catch_exceptions=False,
        )
        assert result.exit_code != 0
        assert len(result.stdout) == 0
        assert len(result.stderr) > 0
        assert "Expected only one of --stats or --nrecords options" in result.stderr

    def test_no_stats_or_nrecords(self, fx_vcz_path):
        runner = ct.CliRunner()
        result = runner.invoke(
            cli.vcztools_main,
            f"index {fx_vcz_path}",
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


@pytest.mark.parametrize("backend", ["zip", "directory"])
class TestBackends:
    """Smoke tests verifying each CLI command works with both storage backends."""

    def test_view(self, fx_sample_vcz, backend):
        vcz = fx_sample_vcz.path(backend).as_posix()
        output, _ = run_vcztools(f"view --no-version {vcz}")
        assert output.startswith("##fileformat=VCF")
        assert "NA00001" in output

    def test_query(self, fx_sample_vcz, backend):
        vcz = fx_sample_vcz.path(backend).as_posix()
        output, _ = run_vcztools(f"query -f '%POS\\n' {vcz}")
        positions = output.strip().splitlines()
        assert len(positions) == 9
        assert positions[0] == "111"

    def test_index_nrecords(self, fx_sample_vcz, backend):
        vcz = fx_sample_vcz.path(backend).as_posix()
        output, _ = run_vcztools(f"index -n {vcz}")
        assert output.strip() == "9"

    def test_index_stats(self, fx_sample_vcz, backend):
        vcz = fx_sample_vcz.path(backend).as_posix()
        output, _ = run_vcztools(f"index -s {vcz}")
        assert output.strip().splitlines() == ["19\t.\t2", "20\t.\t6", "X\t.\t1"]
