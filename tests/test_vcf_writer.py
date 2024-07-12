from io import StringIO

import pytest
from cyvcf2 import VCF
from numpy.testing import assert_array_equal

from bio2zarr import vcf2zarr
from vcztools.vcf_writer import write_vcf

from .utils import assert_vcfs_close


@pytest.mark.parametrize("output_is_path", [True, False])
@pytest.mark.parametrize("implementation", ["c", "numba"])
def test_write_vcf(shared_datadir, tmp_path, output_is_path, implementation):
    path = shared_datadir / "vcf" / "sample.vcf.gz"
    intermediate_icf = tmp_path.joinpath("intermediate.icf")
    intermediate_vcz = tmp_path.joinpath("intermediate.vcz")
    output = tmp_path.joinpath("output.vcf")

    vcf2zarr.convert(
        [path], intermediate_vcz, icf_path=intermediate_icf, worker_processes=0
    )

    if output_is_path:
        write_vcf(intermediate_vcz, output, implementation=implementation)
    else:
        output_str = StringIO()
        write_vcf(intermediate_vcz, output_str, implementation=implementation)
        with open(output, "w") as f:
            f.write(output_str.getvalue())

    v = VCF(output)

    assert v.samples == ["NA00001", "NA00002", "NA00003"]

    variant = next(v)

    assert variant.CHROM == "19"
    assert variant.POS == 111
    assert variant.ID is None
    assert variant.REF == "A"
    assert variant.ALT == ["C"]
    assert variant.QUAL == pytest.approx(9.6)
    assert variant.FILTER is None

    assert variant.genotypes == [[0, 0, True], [0, 0, True], [0, 1, False]]

    assert_array_equal(
        variant.format("HQ"),
        [[10, 15], [10, 10], [3, 3]],
    )

    # check headers are the same
    assert_vcfs_close(path, output)


@pytest.mark.parametrize("implementation", ["c", "numba"])
def test_write_vcf__targets(shared_datadir, tmp_path, implementation):
    path = shared_datadir / "vcf" / "sample.vcf.gz"
    intermediate_icf = tmp_path.joinpath("intermediate.icf")
    intermediate_vcz = tmp_path.joinpath("intermediate.vcz")
    output = tmp_path.joinpath("output.vcf")

    vcf2zarr.convert(
        [path], intermediate_vcz, icf_path=intermediate_icf, worker_processes=0
    )

    write_vcf(intermediate_vcz, output, variant_targets="20", implementation=implementation)

    v = VCF(output)

    assert v.samples == ["NA00001", "NA00002", "NA00003"]

    count = 0
    for variant in v:
        assert variant.CHROM == "20"
        count += 1

    assert count == 6


def test_write_vcf__set_header(shared_datadir, tmp_path):
    path = shared_datadir / "vcf" / "sample.vcf.gz"
    intermediate_icf = tmp_path.joinpath("intermediate.icf")
    intermediate_vcz = tmp_path.joinpath("intermediate.vcz")
    output = tmp_path.joinpath("output.vcf")

    vcf2zarr.convert(
        [path], intermediate_vcz, icf_path=intermediate_icf, worker_processes=0
    )

    # specified header drops NS and HQ fields,
    # and adds H3 and GL fields (which are not in the data)
    vcf_header = """##fileformat=VCFv4.3
##INFO=<ID=AN,Number=1,Type=Integer,Description="Total number of alleles in called genotypes">
##INFO=<ID=AC,Number=.,Type=Integer,Description="Allele count in genotypes, for each ALT allele, in the same order as listed">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##INFO=<ID=AF,Number=.,Type=Float,Description="Allele Frequency">
##INFO=<ID=AA,Number=1,Type=String,Description="Ancestral Allele">
##INFO=<ID=DB,Number=0,Type=Flag,Description="dbSNP membership, build 129">
##INFO=<ID=H2,Number=0,Type=Flag,Description="HapMap2 membership">
##INFO=<ID=H3,Number=0,Type=Flag,Description="HapMap3 membership">
##FILTER=<ID=s50,Description="Less than 50% of samples have data">
##FILTER=<ID=q10,Description="Quality below 10">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">
##FORMAT=<ID=GL,Number=G,Type=Float,Description="Genotype likelihoods">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	NA00001	NA00002	NA00003
"""

    write_vcf(intermediate_vcz, output, vcf_header=vcf_header)

    v = VCF(output)
    # check dropped fields are not present in VCF header
    assert "##INFO=<ID=NS" not in v.raw_header
    assert "##FORMAT=<ID=HQ" not in v.raw_header
    # check added fields are present in VCF header
    assert "##INFO=<ID=H3" in v.raw_header
    assert "##FORMAT=<ID=GL" in v.raw_header
    count = 0
    for variant in v:
        # check dropped fields are not present in VCF data
        assert "NS" not in dict(variant.INFO).keys()
        assert "HQ" not in variant.FORMAT
        # check added fields are not present in VCF data
        assert "H3" not in dict(variant.INFO).keys()
        assert "GL" not in variant.FORMAT

        assert variant.genotypes is not None
        count += 1
    assert count == 9
