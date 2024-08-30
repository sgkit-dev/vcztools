import pathlib

import pytest

from tests.utils import vcz_path_cache
from vcztools.vcf_writer import write_vcf

from .utils import assert_vcfs_close


@pytest.mark.parametrize(
    "vcf_file",
    [
        "sample.vcf.gz",
        "1kg_2020_chr20_annotations.bcf",
        "1kg_2020_chrM.vcf.gz",
        "field_type_combos.vcf.gz",
    ],
)
def test_vcf_to_zarr_to_vcf__real_files(tmp_path, vcf_file):
    original = pathlib.Path("tests/data/vcf") / vcf_file
    vcz = vcz_path_cache(original)
    generated = tmp_path.joinpath("output.vcf")
    write_vcf(vcz, generated, no_version=True)
    assert_vcfs_close(original, generated)
