import pathlib
import subprocess

import click.testing as ct
import pytest

import vcztools.cli as cli

from .utils import assert_vcfs_close, vcz_path_cache


def run_bcftools(args: str) -> str:
    """Run bcftools (which must be on the PATH) and return stdout as a string."""
    completed = subprocess.run(
        f"bcftools {args}", capture_output=True, check=True, shell=True
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
        ("view --no-version -i 'INFO/DP > 10'", "sample.vcf.gz"),
        ("view --no-version -i 'FMT/DP >= 5 && FMT/GQ > 10'", "sample.vcf.gz"),
        ("view --no-version -i 'FMT/DP >= 5 & FMT/GQ>10'", "sample.vcf.gz"),
        (
                "view --no-version -i '(QUAL > 10 || FMT/GQ>10) && POS > 100000'",
                "sample.vcf.gz"
        ),
        (
                "view --no-version -i '(FMT/DP >= 8 | FMT/GQ>40) && POS > 100000'",
                "sample.vcf.gz"
        ),
        (
                "view --no-version -e '(FMT/DP >= 8 | FMT/GQ>40) && POS > 100000'",
                "sample.vcf.gz"
        ),
    ]
)
# fmt: on
def test_vcf_output(tmp_path, args, vcf_file):
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


@pytest.mark.parametrize(
    ("args", "vcf_name"),
    [
        ("index -n", "sample.vcf.gz"),
        ("index --nrecords", "1kg_2020_chrM.vcf.gz"),
        ("query -l", "sample.vcf.gz"),
        ("query --list-samples", "1kg_2020_chrM.vcf.gz"),
    ],
)
def test_output(tmp_path, args, vcf_name):
    vcf_path = pathlib.Path("tests/data/vcf") / vcf_name
    vcz_path = vcz_path_cache(vcf_path)

    bcftools_output = run_bcftools(f"{args} {vcf_path}")
    vcztools_output = run_vcztools(f"{args} {vcz_path}")

    assert vcztools_output == bcftools_output
