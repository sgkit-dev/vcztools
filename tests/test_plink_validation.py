import os
import pathlib
import subprocess

import click.testing as ct
import pytest

import vcztools.cli as cli

from . import utils


def assert_files_identical(path1, path2):
    """
    Asserts the files are byte-for-byte identical.
    """
    with open(path1, "rb") as f:
        b1 = f.read()
    with open(path2, "rb") as f:
        b2 = f.read()
    assert b1 == b2


@pytest.mark.skip("Removing plink from CLI for bugfix release")
# fmt: off
@pytest.mark.parametrize(
    ("args", "vcf_file"),
    [
        ("", "sample.vcf.gz"),
        ("", "chr22.vcf.gz"),
        ("", "1kg_2020_chrM.vcf.gz"),
        # FIXME this needs some extra args to deal with sample ID format
        # ("", "msprime_diploid.vcf.gz"),
    ],
)
# fmt: on
def test_conversion_identical(tmp_path, args, vcf_file):
    original = pathlib.Path("tests/data/vcf") / vcf_file
    vcz = utils.vcz_path_cache(original)

    plink_workdir = tmp_path / "plink1.9"
    plink_workdir.mkdir()
    plink_bin = os.environ.get("PLINK_BIN", "plink")
    cmd = f"{plink_bin} --vcf {original.absolute()} {args}"
    result = subprocess.run(cmd, shell=True, cwd=plink_workdir, capture_output=True)
    assert result.returncode == 0

    cmd = f"view-plink1 {vcz.absolute()} {args}"
    runner = ct.CliRunner()
    with runner.isolated_filesystem(tmp_path) as working_dir:
        vcz_workdir = pathlib.Path(working_dir)
        result = runner.invoke(cli.vcztools_main, cmd, catch_exceptions=False)
        for filename in ["plink.fam", "plink.bim", "plink.bed"]:
            assert_files_identical(vcz_workdir / filename, plink_workdir / filename)
