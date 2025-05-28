import os
import pathlib
import subprocess

import pytest

from vcztools.plink import write_plink

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
    tmp_path = pathlib.Path("tmp/plink")

    original = pathlib.Path("tests/data/vcf") / vcf_file
    vcz = utils.vcz_path_cache(original)

    plink_prefix = str(tmp_path / "plink")
    plink_bin = os.environ.get("PLINK_BIN", "plink")
    cmd = f"{plink_bin} --out {plink_prefix} --vcf {original.absolute()} {args}"
    result = subprocess.run(cmd, shell=True, capture_output=True)
    assert result.returncode == 0

    vcztools_prefix = str(tmp_path / "vcztools")
    write_plink(vcz, vcztools_prefix)

    for suffix in [".bed", ".fam", ".bim"]:
        plink_file = plink_prefix + suffix
        vcztools_file = vcztools_prefix + suffix
        assert_files_identical(plink_file, vcztools_file)
