import pathlib
import re
import sys
from io import StringIO

import numpy as np
import pytest
import zarr
from bio2zarr import vcf2zarr
from cyvcf2 import VCF
from numpy.testing import assert_array_equal

from vcztools import filter as filter_mod
from vcztools.constants import INT_FILL, INT_MISSING
from vcztools.vcf_writer import _compute_info_fields, write_vcf

from .utils import assert_vcfs_close, vcz_path_cache


@pytest.mark.parametrize("output_is_path", [True, False])
def test_write_vcf(tmp_path, output_is_path):
    original = pathlib.Path("tests/data/vcf") / "sample.vcf.gz"
    vcz = vcz_path_cache(original)
    output = tmp_path.joinpath("output.vcf")

    if output_is_path:
        write_vcf(vcz, output, no_version=True)
    else:
        output_str = StringIO()
        write_vcf(vcz, output_str, no_version=True)
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
    assert_vcfs_close(original, output)


@pytest.mark.parametrize(
    ("include", "exclude", "expected_chrom_pos"),
    [
        ("POS < 1000", None, [("19", 111), ("19", 112), ("X", 10)]),
        (
            None,
            "POS < 1000",
            [
                ("20", 14370),
                ("20", 17330),
                ("20", 1110696),
                ("20", 1230237),
                ("20", 1234567),
                ("20", 1235237),
            ],
        ),
    ],
)
def test_write_vcf__filtering(tmp_path, include, exclude, expected_chrom_pos):
    original = pathlib.Path("tests/data/vcf") / "sample.vcf.gz"
    vcz = vcz_path_cache(original)
    output = tmp_path.joinpath("output.vcf")

    write_vcf(vcz, output, include=include, exclude=exclude)

    v = VCF(str(output))
    variants = list(v)

    assert len(variants) == len(expected_chrom_pos)
    assert v.samples == ["NA00001", "NA00002", "NA00003"]

    for variant, chrom_pos in zip(variants, expected_chrom_pos):
        chrom, pos = chrom_pos
        assert variant.CHROM == chrom
        assert variant.POS == pos


# fmt: off
@pytest.mark.parametrize(
    ("regions", "targets", "expected_chrom_pos"),
    [
        # regions only
        ("19", None, [("19", 111), ("19", 112)]),
        ("19:112", None, [("19", 112)]),
        ("20:1230236-", None, [("20", 1230237), ("20", 1234567), ("20", 1235237)]),
        ("20:1230237-", None, [("20", 1230237), ("20", 1234567), ("20", 1235237)]),
        ("20:1230238-", None, [("20", 1234567), ("20", 1235237)]),
        ("20:1230237-1235236", None, [("20", 1230237), ("20", 1234567)]),
        ("20:1230237-1235237", None, [("20", 1230237), ("20", 1234567), ("20", 1235237)]),  # noqa: E501
        ("20:1230237-1235238", None, [("20", 1230237), ("20", 1234567), ("20", 1235237)]),  # noqa: E501
        ("19,X", None, [("19", 111), ("19", 112), ("X", 10)]),
        ("X:11", None, [("X", 10)]),  # note differs from targets

        # targets only
        (None, "19", [("19", 111), ("19", 112)]),
        (None, "19:112", [("19", 112)]),
        (None, "20:1230236-", [("20", 1230237), ("20", 1234567), ("20", 1235237)]),
        (None, "20:1230237-", [("20", 1230237), ("20", 1234567), ("20", 1235237)]),
        (None, "20:1230238-", [("20", 1234567), ("20", 1235237)]),
        (None, "20:1230237-1235236", [("20", 1230237), ("20", 1234567)]),
        (None, "20:1230237-1235237", [("20", 1230237), ("20", 1234567), ("20", 1235237)]),  # noqa: E501
        (None, "20:1230237-1235238", [("20", 1230237), ("20", 1234567), ("20", 1235237)]),  # noqa: E501
        (None, "19,X", [("19", 111), ("19", 112), ("X", 10)]),
        (None, "X:11", []),
        (None, "^19,20:1-1234567", [("20", 1235237), ("X", 10)]),  # complement

        # regions and targets
        ("20", "^20:1110696-", [("20", 14370), ("20", 17330)])
    ]
)
# fmt: on
def test_write_vcf__regions(tmp_path, regions, targets, expected_chrom_pos):

    original = pathlib.Path("tests/data/vcf") / "sample.vcf.gz"
    vcz = vcz_path_cache(original)
    output = tmp_path.joinpath("output.vcf")

    write_vcf(vcz, output, variant_regions=regions, variant_targets=targets)

    v = VCF(output)
    variants = list(v)
    assert len(variants) == len(expected_chrom_pos)

    assert v.samples == ["NA00001", "NA00002", "NA00003"]

    for variant, chrom_pos in zip(variants, expected_chrom_pos):
        chrom, pos = chrom_pos
        assert variant.CHROM == chrom
        assert variant.POS == pos


@pytest.mark.parametrize(
    ("samples", "force_samples", "expected_samples", "expected_genotypes"),
    [
        ("NA00001", False, ["NA00001"], [[0, 0, True]]),
        (
            "NA00001,NA00003",
            False,
            ["NA00001", "NA00003"],
            [[0, 0, True], [0, 1, False]],
        ),
        (
            "NA00003,NA00001",
            False,
            ["NA00003", "NA00001"],
            [[0, 1, False], [0, 0, True]],
        ),
        ("^NA00002", False, ["NA00001", "NA00003"], [[0, 0, True], [0, 1, False]]),
        ("^NA00003,NA00002", False, ["NA00001"], [[0, 0, True]]),
        ("^NA00003,NA00002,NA00003", False, ["NA00001"], [[0, 0, True]]),
        ("NO_SAMPLE", True, [], None),
    ],
)
def test_write_vcf__samples(
    tmp_path, samples, force_samples, expected_samples, expected_genotypes
):
    original = pathlib.Path("tests/data/vcf") / "sample.vcf.gz"
    vcz = vcz_path_cache(original)
    output = tmp_path.joinpath("output.vcf")

    write_vcf(vcz, output, samples=samples, force_samples=force_samples)

    v = VCF(output)

    assert v.samples == expected_samples

    variant = next(v)

    assert variant.CHROM == "19"
    assert variant.POS == 111
    assert variant.ID is None
    assert variant.REF == "A"
    assert variant.ALT == ["C"]
    assert variant.QUAL == pytest.approx(9.6)
    assert variant.FILTER is None

    assert variant.genotypes == expected_genotypes


def test_write_vcf__non_existent_sample(tmp_path):
    original = pathlib.Path("tests/data/vcf") / "sample.vcf.gz"
    vcz = vcz_path_cache(original)
    output = tmp_path.joinpath("output.vcf")

    with pytest.raises(
        ValueError,
        match=re.escape(
            "subset called for sample(s) not in header: NO_SAMPLE. "
            'Use "--force-samples" to ignore this error.'
        ),
    ):
        write_vcf(vcz, output, samples="NO_SAMPLE")


def test_write_vcf__no_samples(tmp_path):
    original = pathlib.Path("tests/data/vcf") / "sample.vcf.gz"
    vcz = vcz_path_cache(original)
    output = tmp_path.joinpath("output.vcf")

    write_vcf(vcz, output, drop_genotypes=True)

    v = VCF(output)

    assert v.samples == []


# @pytest.mark.skip("Sample filtering not implemented: #180")
@pytest.mark.parametrize(
    ("regions", "targets", "samples", "include", "expected_chrom_pos"),
    [
        # Test that sample filtering takes place after include filtering.
        # Not supporting format fields in filtering for now: #180
        # ("20", None, "NA00001", "FMT/GQ > 60", [("20", 1230237)]),
        # Test that region filtering and include expression are combined.
        ("19", None, "NA00001", "POS > 200", []),
        # Test that target filtering and include expression are combined.
        (None, "19", "NA00001", "POS > 200", []),
        # Test that empty output in the no-regions cases works
        (None, None, "NA00001", "POS < 1", []),
        # Test that empty output in the no-regions cases works
        (None, None, None, "POS < 1", []),
    ],
)
def test_write_vcf__regions_samples_filtering(
    tmp_path, regions, targets, samples, include, expected_chrom_pos
):
    original = pathlib.Path("tests/data/vcf") / "sample.vcf.gz"
    vcz = vcz_path_cache(original)
    output = tmp_path.joinpath("output.vcf")

    write_vcf(
        vcz,
        output,
        variant_regions=regions,
        variant_targets=targets,
        samples=samples,
        include=include,
    )

    v = VCF(str(output))
    variants = list(v)

    assert len(variants) == len(expected_chrom_pos)
    if samples is not None:
        assert v.samples == [samples]

    for variant, chrom_pos in zip(variants, expected_chrom_pos):
        chrom, pos = chrom_pos
        assert variant.CHROM == chrom
        assert variant.POS == pos


def test_write_vcf__include_exclude(tmp_path):
    original = pathlib.Path("tests/data/vcf") / "sample.vcf.gz"
    vcz = vcz_path_cache(original)
    output = tmp_path.joinpath("output.vcf")

    variant_site_filter = "POS > 1"

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot handle both an include expression and an exclude expression."
        ),
    ):
        write_vcf(vcz, output, include=variant_site_filter, exclude=variant_site_filter)


def test_write_vcf__header_flags(tmp_path):
    original = pathlib.Path("tests/data/vcf") / "sample.vcf.gz"
    vcz = vcz_path_cache(original)
    output = tmp_path.joinpath("output.vcf")

    output_header = StringIO()
    write_vcf(vcz, output_header, header_only=True, no_version=True)

    output_no_header = StringIO()
    write_vcf(vcz, output_no_header, no_header=True, no_version=True)
    assert not output_no_header.getvalue().startswith("#")

    # combine outputs and check VCFs match
    output_str = output_header.getvalue() + output_no_header.getvalue()
    with open(output, "w") as f:
        f.write(output_str)
    assert_vcfs_close(original, output)


def test_write_vcf__generate_header(tmp_path):
    original = pathlib.Path("tests/data/vcf") / "sample.vcf.gz"
    # don't use cache here since we mutate the vcz
    vcz = tmp_path.joinpath("intermediate.vcz")
    vcf2zarr.convert([original], vcz, worker_processes=0, local_alleles=False)

    # remove vcf_header
    root = zarr.open(vcz, mode="r+")
    del root.attrs["vcf_header"]

    output_header = StringIO()
    write_vcf(vcz, output_header, header_only=True, no_version=True)

    expected_vcf_header = """##fileformat=VCFv4.3
##source={}
##INFO=<ID=AA,Number=1,Type=String,Description="Ancestral Allele">
##INFO=<ID=AC,Number=2,Type=Integer,Description="Allele count in genotypes, for each ALT allele, in the same order as listed">
##INFO=<ID=AF,Number=2,Type=Float,Description="Allele Frequency">
##INFO=<ID=AN,Number=1,Type=Integer,Description="Total number of alleles in called genotypes">
##INFO=<ID=DB,Number=0,Type=Flag,Description="dbSNP membership, build 129">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##INFO=<ID=H2,Number=0,Type=Flag,Description="HapMap2 membership">
##INFO=<ID=NS,Number=1,Type=Integer,Description="Number of Samples With Data">
##FILTER=<ID=PASS,Description="">
##FILTER=<ID=s50,Description="">
##FILTER=<ID=q10,Description="">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
##FORMAT=<ID=HQ,Number=2,Type=Integer,Description="Haplotype Quality">
##contig=<ID=19>
##contig=<ID=20>
##contig=<ID=X>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	NA00001	NA00002	NA00003
"""  # noqa: E501

    # substitute value of source
    expected_vcf_header = expected_vcf_header.format(root.attrs["source"])

    assert output_header.getvalue() == expected_vcf_header


def test_compute_info_fields():
    gt = np.array(
        [
            [[0, 0], [0, 1], [1, 1]],
            [[0, 0], [0, 2], [2, 2]],
            [[0, 1], [1, 2], [2, 2]],
            [
                [INT_MISSING, INT_MISSING],
                [INT_MISSING, INT_MISSING],
                [INT_FILL, INT_FILL],
            ],
            [[INT_MISSING, INT_MISSING], [0, 3], [INT_FILL, INT_FILL]],
        ]
    )
    alt = np.array(
        [
            [b"A", b"B", b""],
            [b"A", b"B", b"C"],
            [b"A", b"B", b"C"],
            [b"", b"", b""],
            [b"A", b"B", b"C"],
        ]
    )
    expected_result = {
        "AC": np.array(
            [
                [3, 0, INT_FILL],
                [0, 3, 0],
                [2, 3, 0],
                [INT_FILL, INT_FILL, INT_FILL],
                [0, 0, 1],
            ]
        ),
        "AN": np.array([6, 6, 6, 0, 2]),
    }

    computed_info_fields = _compute_info_fields(gt, alt)

    assert expected_result.keys() == computed_info_fields.keys()

    for key in expected_result.keys():
        np.testing.assert_array_equal(expected_result[key], computed_info_fields[key])



class TestApiErrors:

    @pytest.fixture()
    def vcz(self):
        original = pathlib.Path("tests/data/vcf") / "sample.vcf.gz"
        return vcz_path_cache(original)

    def test_samples_and_drop_genotypes(self, vcz):
        with pytest.raises(
            ValueError, match="Cannot select samples and drop genotypes"
        ):
            write_vcf(vcz, sys.stdout, samples=["NA00001"], drop_genotypes=True)


    def test_no_output_filter_parse_error(self, vcz):
        output = StringIO()
        with pytest.raises(filter_mod.ParseError):
            write_vcf(vcz, output, include="Not a valid expression")
        assert output.getvalue() == ""
