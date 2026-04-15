import sys

import pytest

from vcztools.vcf_writer import write_vcf

from .utils import assert_vcfs_close

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="Not supported on Windows"
)


@pytest.mark.parametrize(
    "vcf_file",
    [
        "sample.vcf.gz",
        "1kg_2020_chr20_annotations.bcf",
        "1kg_2020_chrM.vcf.gz",
        "field_type_combos.vcf.gz",
    ],
)
def test_vcf_to_zarr_to_vcf__real_files(tmp_path, fx_all_vcz, vcf_file):
    fx = fx_all_vcz[vcf_file]
    generated = tmp_path.joinpath("output.vcf")
    write_vcf(fx.group, generated, no_version=True)
    assert_vcfs_close(fx.vcf_path, generated)
