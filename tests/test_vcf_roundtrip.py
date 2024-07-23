import pathlib

import pytest
from bio2zarr import vcf2zarr

from vcztools.vcf_writer import write_vcf

from .utils import assert_vcfs_close


def vcz_path_cache(vcf_path):
    """
    Store converted files in a cache to speed up tests. We're not testing
    vcf2zarr here, so no point in running over and over again.
    """
    cache_path = pathlib.Path("vcz_test_cache")
    if not cache_path.exists():
        cache_path.mkdir()
    cached_vcz_path = (cache_path / vcf_path.name).with_suffix(".vcz")
    if not cached_vcz_path.exists():
        vcf2zarr.convert(
            [vcf_path], cached_vcz_path, worker_processes=0, local_alleles=False
        )
    return cached_vcz_path


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
    write_vcf(vcz, generated)
    assert_vcfs_close(original, generated)
