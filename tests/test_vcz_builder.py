import pathlib

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from tests.utils import open_vcz, vcz_path_cache
from tests.vcz_builder import copy_vcz, make_vcz

SAMPLE_VCF = pathlib.Path("tests/data/vcf/sample.vcf.gz")


def _values_equal(a_arr, b_arr):
    a = np.asarray(a_arr)
    b = np.asarray(b_arr)
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    if a.dtype.kind in "OUT" or b.dtype.kind in "OUT":
        a = np.array([str(x) for x in a.ravel()]).reshape(a.shape)
        b = np.array([str(x) for x in b.ravel()]).reshape(b.shape)
    assert_array_equal(a, b)


class TestCopyVcz:
    @pytest.fixture
    def source(self):
        return open_vcz(vcz_path_cache(SAMPLE_VCF))

    def test_full_copy_matches_source(self, source):
        copy = copy_vcz(source)
        src_keys = sorted(source.array_keys())
        dst_keys = sorted(copy.array_keys())
        assert src_keys == dst_keys
        for name in src_keys:
            _values_equal(source[name][...], copy[name][...])

    def test_variants_chunk_size_override(self, source):
        copy = copy_vcz(source, variants_chunk_size=3)
        assert copy["variant_position"].chunks[0] == 3
        assert copy["variant_allele"].chunks[0] == 3
        assert copy["call_genotype"].chunks[0] == 3
        # values are still correct under the new chunking
        _values_equal(source["variant_position"][...], copy["variant_position"][...])
        _values_equal(source["call_genotype"][...], copy["call_genotype"][...])

    def test_samples_chunk_size_override(self, source):
        copy = copy_vcz(source, samples_chunk_size=2)
        assert copy["call_genotype"].chunks[1] == 2
        assert copy["call_DP"].chunks[1] == 2
        _values_equal(source["call_DP"][...], copy["call_DP"][...])

    def test_unknown_array_raises(self):
        # Build a minimal in-memory group with an extra top-level array
        # that copy_vcz doesn't know how to round-trip.
        group = make_vcz(
            variant_contig=[0, 0],
            variant_position=[1, 2],
            alleles=[["A", "T"], ["A", "G"]],
            num_samples=1,
        )
        group.create_array(
            name="contig_length",
            shape=(1,),
            chunks=(1,),
            dtype="int64",
        )
        group["contig_length"][:] = [42]
        with pytest.raises(ValueError, match="contig_length"):
            copy_vcz(group)


class TestMakeVczCallFields:
    def test_call_fields_2d(self):
        group = make_vcz(
            variant_contig=[0, 0],
            variant_position=[1, 2],
            alleles=[["A", "T"], ["A", "G"]],
            num_samples=3,
            call_fields={"DP": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)},
        )
        assert group["call_DP"].shape == (2, 3)
        assert_array_equal(group["call_DP"][...], [[1, 2, 3], [4, 5, 6]])

    def test_call_fields_3d(self):
        data = np.arange(2 * 3 * 2, dtype=np.int8).reshape(2, 3, 2)
        group = make_vcz(
            variant_contig=[0, 0],
            variant_position=[1, 2],
            alleles=[["A", "T"], ["A", "G"]],
            num_samples=3,
            call_fields={"HQ": data},
        )
        assert group["call_HQ"].shape == (2, 3, 2)
        assert_array_equal(group["call_HQ"][...], data)

    def test_call_fields_invalid_ndim(self):
        with pytest.raises(ValueError, match="2- or 3-dimensional"):
            make_vcz(
                variant_contig=[0],
                variant_position=[1],
                alleles=[["A", "T"]],
                num_samples=2,
                call_fields={"X": np.zeros((1, 2, 3, 4), dtype=np.int8)},
            )

    def test_filter_descriptions(self):
        group = make_vcz(
            variant_contig=[0],
            variant_position=[1],
            alleles=[["A", "T"]],
            filters=("PASS", "q10"),
            filter_descriptions=("All filters passed", "Quality below 10"),
        )
        assert group["filter_description"].shape == (2,)
        assert [str(x) for x in group["filter_description"][...]] == [
            "All filters passed",
            "Quality below 10",
        ]

    def test_filter_descriptions_length_mismatch(self):
        with pytest.raises(ValueError, match="filter_descriptions length"):
            make_vcz(
                variant_contig=[0],
                variant_position=[1],
                alleles=[["A", "T"]],
                filters=("PASS", "q10"),
                filter_descriptions=("only one",),
            )

    def test_sample_id_passthrough(self):
        group = make_vcz(
            variant_contig=[0],
            variant_position=[1],
            alleles=[["A", "T"]],
            num_samples=3,
            sample_id=["NA00001", "NA00002", "NA00003"],
        )
        assert [str(x) for x in group["sample_id"][...]] == [
            "NA00001",
            "NA00002",
            "NA00003",
        ]
