import pytest

from bio2zarr import vcf2zarr
from vcztools.vcf_writer import write_vcf
from .utils import assert_vcfs_close


@pytest.mark.parametrize(
    "vcf_file", ["sample.vcf.gz"]
)
@pytest.mark.parametrize("implementation", ["c", "numba"])
def test_vcf_to_zarr_to_vcf__real_files(shared_datadir, tmp_path, vcf_file, implementation):
    path = shared_datadir / "vcf" / vcf_file
    intermediate_icf = tmp_path.joinpath("intermediate.icf")
    intermediate_vcz = tmp_path.joinpath("intermediate.vcz")
    output = tmp_path.joinpath("output.vcf")

    vcf2zarr.convert(
        [path], intermediate_vcz, icf_path=intermediate_icf, worker_processes=0
    )

    write_vcf(intermediate_vcz, output, implementation=implementation)

    assert_vcfs_close(path, output)
