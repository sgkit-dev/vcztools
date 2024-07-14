import pytest
import numpy as np

from vcztools import _vcztools

FIXED_FIELD_NAMES = ["chrom", "pos", "id", "qual", "ref", "alt", "filter"]


def example_fixed_data(num_variants, num_samples=0):
    chrom = np.array(["X"] * num_variants, dtype="S")
    pos = np.arange(num_variants, dtype=np.int32)
    id = np.array(["."] * num_variants, dtype="S").reshape((num_variants, 1))
    ref = np.array(["A"] * num_variants, dtype="S")
    alt = np.array(["T"] * num_variants, dtype="S").reshape((num_variants, 1))
    qual = np.arange(num_variants, dtype=np.float32)
    filter_ = np.ones(num_variants, dtype=bool).reshape((num_variants, 1))
    filter_id = np.array(["PASS"], dtype="S")
    return {
        "chrom": chrom,
        "pos": pos,
        "id": id,
        "qual": qual,
        "ref": ref,
        "alt": alt,
        "filter": filter_,
        "filter_ids": filter_id,
    }


class TestTypeChecking:
    @pytest.mark.parametrize("name", FIXED_FIELD_NAMES)
    def test_field_incorrect_length(self, name):
        num_variants = 5
        data = example_fixed_data(num_variants)
        data[name] = data[name][1:]
        with pytest.raises(ValueError, match=f"Array {name.upper()} must have "):
            _vcztools.VcfEncoder(num_variants, 0, **data)

    @pytest.mark.parametrize("name", FIXED_FIELD_NAMES)
    def test_field_incorrect_dtype(self, name):
        num_variants = 5
        data = example_fixed_data(num_variants)
        data[name] = np.zeros(data[name].shape, dtype=np.int64)
        with pytest.raises(ValueError, match=f"Wrong dtype for {name.upper()}"):
            _vcztools.VcfEncoder(num_variants, 0, **data)

    @pytest.mark.parametrize("name", FIXED_FIELD_NAMES)
    def test_field_incorrect_type(self, name):
        num_variants = 5
        data = example_fixed_data(num_variants)
        data[name] = "A Python string"
        with pytest.raises(TypeError, match=f"must be numpy.ndarray"):
            _vcztools.VcfEncoder(num_variants, 0, **data)
