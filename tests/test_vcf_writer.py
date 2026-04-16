import re
from io import StringIO

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from vcztools.constants import INT_FILL, INT_MISSING
from vcztools.retrieval import VczReader
from vcztools.vcf_writer import _compute_info_fields, c_chunk_to_vcf, write_vcf

from .utils import assert_vcfs_close, to_vcz_icechunk
from .vcz_builder import copy_vcz, make_vcz

cyvcf2 = pytest.importorskip("cyvcf2")
VCF = cyvcf2.VCF


@pytest.mark.parametrize(
    ("output_is_path", "zarr_backend_storage"),
    [
        (True, None),
        (True, "obstore"),
        (False, "fsspec"),
    ],
)
def test_write_vcf(
    tmp_path,
    fx_sample_vcz,
    output_is_path,
    zarr_backend_storage,
):
    if zarr_backend_storage == "obstore":
        pytest.importorskip(zarr_backend_storage)
    # obstore cannot read a ZipStore, so we need a directory VCZ.
    vcz = fx_sample_vcz.directory_path
    output = tmp_path.joinpath("output.vcf")

    reader = VczReader(vcz, zarr_backend_storage=zarr_backend_storage)
    if output_is_path:
        write_vcf(reader, output, no_version=True)
    else:
        output_str = StringIO()
        write_vcf(reader, output_str, no_version=True)
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
    assert_vcfs_close(fx_sample_vcz.vcf_path, output)


def test_write_vcf__icechunk(tmp_path, fx_sample_vcz):
    pytest.importorskip("icechunk")

    vcz_icechunk = to_vcz_icechunk(fx_sample_vcz.directory_path, tmp_path)
    output = tmp_path.joinpath("output.vcf")

    reader = VczReader(vcz_icechunk, zarr_backend_storage="icechunk")
    write_vcf(reader, output, no_version=True)

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
    assert_vcfs_close(fx_sample_vcz.vcf_path, output)


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
def test_write_vcf__filtering(
    tmp_path, fx_sample_vcz, include, exclude, expected_chrom_pos
):
    output = tmp_path.joinpath("output.vcf")
    reader = VczReader(fx_sample_vcz.group)
    write_vcf(reader, output, include=include, exclude=exclude)

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
        (None, "19:111", [("19", 111)]),
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
@pytest.mark.parametrize("variants_chunk_size", [None, 1, 2, 3, 7, 100])
def test_write_vcf__regions(
        tmp_path, fx_sample_vcz, regions, targets, expected_chrom_pos,
        variants_chunk_size):
    # Copy the sample fixture into an in-memory VCZ so we can sweep chunk
    # sizes without re-opening disk fixtures. Region/target chunk-boundary
    # behaviour is what this test covers; VCF-level parity is covered
    # elsewhere.
    vcz = copy_vcz(fx_sample_vcz.group, variants_chunk_size=variants_chunk_size)
    output = tmp_path.joinpath("output.vcf")
    reader = VczReader(vcz, regions=regions, targets=targets)
    write_vcf(reader, output)

    v = VCF(str(output))
    variants = list(v)
    assert len(variants) == len(expected_chrom_pos)
    for variant, (chrom, pos) in zip(variants, expected_chrom_pos):
        assert variant.CHROM == chrom
        assert variant.POS == pos


class TestSmallChunks:
    """
    Chunk-boundary coverage for filter/region code paths, built on the
    synthetic in-memory VCZ so we can freely sweep chunk sizes without
    re-running bio2zarr.
    """

    AC_VALUES = [0, 1, 2, 3, 4, 5, 1, 2, 3, 6, 0, 4, 2, 3, 1, 5, 7, 2, 3, 0]
    NUM_VARIANTS = len(AC_VALUES)

    def _build(self, variants_chunk_size):
        positions = list(range(1, self.NUM_VARIANTS + 1))
        return make_vcz(
            variant_contig=[0] * self.NUM_VARIANTS,
            variant_position=positions,
            alleles=[["A", "T"]] * self.NUM_VARIANTS,
            contigs=("chr1",),
            variants_chunk_size=variants_chunk_size,
            info_fields={"AC": np.array(self.AC_VALUES, dtype=np.int32)},
        )

    @pytest.mark.parametrize("variants_chunk_size", [1, 2, 3, 5, 7, 10, None])
    def test_info_ac_filter(self, tmp_path, variants_chunk_size):
        vcz = self._build(variants_chunk_size)
        output = tmp_path / "out.vcf"
        reader = VczReader(vcz)
        write_vcf(reader, output, include="INFO/AC>2", no_version=True)

        expected_positions = [
            i + 1 for i, ac in enumerate(self.AC_VALUES) if ac > 2
        ]
        v = VCF(str(output))
        got_positions = [variant.POS for variant in v]
        assert got_positions == expected_positions

    @pytest.mark.parametrize("variants_chunk_size", [1, 2, 3, 5, 7, 10, None])
    def test_region_filter(self, tmp_path, variants_chunk_size):
        vcz = self._build(variants_chunk_size)
        output = tmp_path / "out.vcf"
        reader = VczReader(vcz, regions="chr1:5-12")
        write_vcf(reader, output, no_version=True)

        v = VCF(str(output))
        got_positions = [variant.POS for variant in v]
        assert got_positions == list(range(5, 13))


@pytest.mark.parametrize("variants_chunk_size", [3, 4, 5])
def test_write_vcf__regions_split_alleles(
    tmp_path, fx_sample_split_alleles_vcz, variants_chunk_size
):
    vcz = copy_vcz(
        fx_sample_split_alleles_vcz.group, variants_chunk_size=variants_chunk_size
    )
    output = tmp_path.joinpath("output.vcf")
    reader = VczReader(vcz, regions="20:1234567")
    write_vcf(reader, output)

    v = VCF(output)
    variants = list(v)
    assert len(variants) == 2

    variant = variants[0]
    assert variant.CHROM == "20"
    assert variant.POS == 1234567
    assert variant.REF == "G"
    assert variant.ALT == ["GA"]

    variant = variants[1]
    assert variant.CHROM == "20"
    assert variant.POS == 1234567
    assert variant.REF == "G"
    assert variant.ALT == ["GAC"]


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
    tmp_path,
    fx_sample_vcz,
    samples,
    force_samples,
    expected_samples,
    expected_genotypes,
):
    output = tmp_path.joinpath("output.vcf")
    reader = VczReader(
        fx_sample_vcz.group, samples=samples, force_samples=force_samples
    )
    write_vcf(reader, output)

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


def test_write_vcf__non_existent_sample(fx_sample_vcz):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "subset called for sample(s) not in header: NO_SAMPLE. "
            'Use "--force-samples" to ignore this error.'
        ),
    ):
        VczReader(fx_sample_vcz.group, samples="NO_SAMPLE")


def test_write_vcf__no_samples(tmp_path, fx_sample_vcz):
    output = tmp_path.joinpath("output.vcf")
    reader = VczReader(fx_sample_vcz.group, drop_genotypes=True)
    write_vcf(reader, output, drop_genotypes=True)

    v = VCF(output)
    assert v.samples == []


def test_write_vcf__missing_samples(tmp_path, fx_sample_vcz):
    mutated = copy_vcz(fx_sample_vcz.group)
    # delete samples NA00001 and NA00002 at index 0 and 1
    mutated["sample_id"][:2] = ""

    output = tmp_path.joinpath("output.vcf")
    reader = VczReader(mutated)
    write_vcf(reader, output)

    v = VCF(output)
    assert v.samples == ["NA00003"]



@pytest.mark.parametrize(
    ("regions", "targets", "samples", "include", "expected_chrom_pos"),
    [
        # Test that sample filtering takes place after include filtering.
        ("20", None, "NA00001", "FMT/GQ > 60", [("20", 1230237)]),
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
    tmp_path, fx_sample_vcz, regions, targets, samples, include, expected_chrom_pos
):
    output = tmp_path.joinpath("output.vcf")
    reader = VczReader(
        fx_sample_vcz.group,
        regions=regions,
        targets=targets,
        samples=samples,
    )
    write_vcf(reader, output, include=include)

    v = VCF(str(output))
    variants = list(v)

    assert len(variants) == len(expected_chrom_pos)
    if samples is not None:
        assert v.samples == [samples]

    for variant, chrom_pos in zip(variants, expected_chrom_pos):
        chrom, pos = chrom_pos
        assert variant.CHROM == chrom
        assert variant.POS == pos


def test_write_vcf__include_exclude(tmp_path, fx_sample_vcz):
    output = tmp_path.joinpath("output.vcf")
    variant_site_filter = "POS > 1"

    reader = VczReader(fx_sample_vcz.group)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot handle both an include expression and an exclude expression."
        ),
    ):
        write_vcf(
            reader,
            output,
            include=variant_site_filter,
            exclude=variant_site_filter,
        )


def test_write_vcf__header_flags(tmp_path, fx_sample_vcz):
    output = tmp_path.joinpath("output.vcf")

    reader = VczReader(fx_sample_vcz.group)

    output_header = StringIO()
    write_vcf(reader, output_header, header_only=True, no_version=True)

    output_no_header = StringIO()
    write_vcf(reader, output_no_header, no_header=True, no_version=True)
    assert not output_no_header.getvalue().startswith("#")

    # combine outputs and check VCFs match
    output_str = output_header.getvalue() + output_no_header.getvalue()
    with open(output, "w") as f:
        f.write(output_str)
    assert_vcfs_close(fx_sample_vcz.vcf_path, output)


def test_write_vcf__generate_header(fx_sample_vcz):
    output_header = StringIO()
    reader = VczReader(fx_sample_vcz.group)
    write_vcf(reader, output_header, header_only=True, no_version=True)

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
##FILTER=<ID=PASS,Description="All filters passed">
##FILTER=<ID=s50,Description="Less than 50% of samples have data">
##FILTER=<ID=q10,Description="Quality below 10">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
##FORMAT=<ID=HQ,Number=2,Type=Integer,Description="Haplotype Quality">
##contig=<ID=19>
##contig=<ID=20>
##contig=<ID=X>
##fileDate=20090805
##reference=1000GenomesPilot-NCBI36
##phasing=partial
##ALT=<ID=DEL:ME:ALU,Description="Deletion of ALU element">
##ALT=<ID=CNV,Description="Copy number variable region">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	NA00001	NA00002	NA00003
"""  # noqa: E501

    # substitute value of source
    source_attr = fx_sample_vcz.group.attrs["source"]
    expected_vcf_header = expected_vcf_header.format(source_attr)

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

    @pytest.fixture
    def fx_vcz(self, fx_sample_vcz):
        return fx_sample_vcz.group

    def test_no_output_filter_parse_error(self, fx_vcz):
        output = StringIO()
        reader = VczReader(fx_vcz)
        with pytest.raises(ValueError, match='the tag "Not" is not defined'):
            write_vcf(reader, output, include="Not a valid expression")
        assert output.getvalue() == ""


def minimal_vcf_chunk(num_variants, num_samples, ploidy=2):
    return {
        "variant_position": 1 + np.arange(num_variants, dtype=np.int32),
        "variant_contig": np.zeros(num_variants, dtype=np.int32),
        # "variant_id": np.array(["."] * num_variants, dtype="S1"),
        "variant_id": np.array(["."] * num_variants, dtype="S").reshape(
            (num_variants, 1)
        ),
        "variant_allele": np.array([("A", "T")] * num_variants),
        "variant_quality": np.zeros(num_variants, dtype=np.float32),
        "variant_filter": np.ones(num_variants, dtype=bool).reshape((num_variants, 1)),
        "call_genotype": np.zeros((num_variants, num_samples, ploidy), dtype=np.int8),
    }


def chunk_to_vcf(chunk):
    filters = np.array([b"PASS"])
    contigs = np.array([b"chr1"])
    output = StringIO()
    c_chunk_to_vcf(
        chunk,
        samples_selection=None,
        contigs=contigs,
        filters=filters,
        output=output,
        drop_genotypes=False,
        no_update=False,
    )
    return output.getvalue()


def chunk_to_vcf_file(chunk):
    """
    Simple function just to get the data out to a minimal file for
    testing and evaluation
    """
    num_samples = chunk["call_genotype"].shape[1]

    output = StringIO()
    print("##fileformat=VCFv4.3", file=output)
    print("##contig=<ID=chr1>", file=output)
    print(
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        file=output,
    )
    print(
        "#CHROM",
        "POS",
        "ID",
        "REF",
        "ALT",
        "QUAL",
        "FILTER",
        "INFO",
        sep="\t",
        end="",
        file=output,
    )
    print(end="\t", file=output)
    sample_ids = [f"x{j}" for j in range(num_samples)]
    print("FORMAT", *sample_ids, sep="\t", file=output)
    return output.getvalue() + chunk_to_vcf(chunk)


class TestEncoding:

    def test_basic_example(self):
        chunk = minimal_vcf_chunk(1, 2)
        out = chunk_to_vcf(chunk)
        line = "\t".join(
            ["chr1", "1", ".", "A", "T", "0", "PASS", ".", "GT", "0/0", "0/0"]
        )
        assert out == line + "\n"

    def test_mixed_ploidy(self):
        chunk = minimal_vcf_chunk(2, 2)
        chunk["call_genotype"][0, 0, 1] = -2
        chunk["call_genotype"][1, 1, 1] = -2
        out = chunk_to_vcf(chunk)
        lines = [
            ["chr1", "1", ".", "A", "T", "0", "PASS", ".", "GT", "0", "0/0"],
            ["chr1", "2", ".", "A", "T", "0", "PASS", ".", "GT", "0/0", "0"],
        ]
        lines = "\n".join("\t".join(line) for line in lines)
        assert out == lines + "\n"

    def test_zero_ploidy(self):
        chunk = minimal_vcf_chunk(2, 2)
        chunk["call_genotype"][0, 0] = -2
        chunk["call_genotype"][1, 1] = -2
        out = chunk_to_vcf(chunk)
        lines = [
            ["chr1", "1", ".", "A", "T", "0", "PASS", ".", "GT", "", "0/0"],
            ["chr1", "2", ".", "A", "T", "0", "PASS", ".", "GT", "0/0", ""],
        ]
        lines = "\n".join("\t".join(line) for line in lines)
        assert out == lines + "\n"

        # NOTE bcftools/htslib doesn't like this
        # [E::vcf_parse_format] Couldn't read GT data:
        #  value not a number or '.' at chr1:1

        # with open("zero-ploidy.vcf", "w") as f:
        #     print(chunk_to_vcf_file(chunk), file=f, end="")
