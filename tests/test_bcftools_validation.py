import pathlib
import subprocess

import click.testing as ct
import pytest

import vcztools.cli as cli

from .utils import assert_vcfs_close, vcz_path_cache


def run_bcftools(args: str) -> str:
    """Run bcftools (which must be on the PATH) and return stdout as a string."""
    completed = subprocess.run(
        ["bcftools", *(args.split(" "))], capture_output=True, check=True
    )
    return completed.stdout.decode("utf-8")


def run_vcztools(args: str) -> str:
    """Run run_vcztools and return stdout as a string."""
    runner = ct.CliRunner(mix_stderr=False)
    result = runner.invoke(
        cli.vcztools_main,
        args,
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    return result.stdout


# fmt: off
@pytest.mark.parametrize(
    ("args", "vcf_file"),
    [
        ("view --no-version", "sample.vcf.gz"),
    ]
)
# fmt: on
def test(tmp_path, args, vcf_file):
    original = pathlib.Path("tests/data/vcf") / vcf_file
    vcz = vcz_path_cache(original)

    bcftools_out = run_bcftools(f"{args} {original}")
    bcftools_out_file = tmp_path.joinpath("bcftools_out.vcf")
    with open(bcftools_out_file, "w") as f:
        f.write(bcftools_out)

    vcztools_out = run_vcztools(f"{args} {vcz}")
    vcztools_out_file = tmp_path.joinpath("vcztools_out.vcf")
    with open(vcztools_out_file, "w") as f:
        f.write(vcztools_out)

    assert_vcfs_close(bcftools_out_file, vcztools_out_file)
