import pathlib
from io import StringIO

import pytest
from bio2zarr import vcf2zarr

from vcztools.stats import nrecords, stats

from .utils import vcz_path_cache


def test_nrecords():
    original = pathlib.Path("tests/data/vcf") / "sample.vcf.gz"
    vcz = vcz_path_cache(original)

    output_str = StringIO()
    nrecords(vcz, output_str)
    assert output_str.getvalue() == "9\n"


def test_stats():
    original = pathlib.Path("tests/data/vcf") / "sample.vcf.gz"
    vcz = vcz_path_cache(original)

    output_str = StringIO()
    stats(vcz, output_str)

    assert (
        output_str.getvalue()
        == """19	.	2
20	.	6
X	.	1
"""
    )


def test_stats__no_index(tmp_path):
    original = pathlib.Path("tests/data/vcf") / "sample.vcf.gz"
    # don't use cache here since we want to make sure vcz is not indexed
    vcz = tmp_path.joinpath("intermediate.vcz")
    vcf2zarr.convert([original], vcz, worker_processes=0, local_alleles=False)

    with pytest.raises(ValueError, match="Could not load 'region_index' variable."):
        stats(vcz, StringIO())
