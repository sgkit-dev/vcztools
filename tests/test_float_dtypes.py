"""Characterisation tests for non-float32 floating-point input fields.

vcztools assumes float32 for every floating-point field, but the
assumption is enforced inconsistently across the API surfaces. These
tests pin down what *currently* happens when an input VCZ carries a
half-precision (``<f2``) or double-precision (``<f8``) float field,
across ``view``, ``query`` and ``VczReader`` filtering.

They document the status quo as a regression baseline for the
eventual half-precision support work; the assertions here describe
present behaviour and are expected to be revised deliberately when
that support lands.
"""

import io

import numpy as np
import pytest

from vcztools import bcftools_filter
from vcztools.query import QueryFormatter
from vcztools.retrieval import VczReader
from vcztools.vcf_writer import write_vcf

from .vcz_builder import make_vcz

NON_FLOAT32_DTYPES = [np.float16, np.float64]


def _base_kwargs():
    return dict(
        variant_contig=[0, 0, 0],
        variant_position=[10, 20, 30],
        alleles=[["A", "C"], ["G", "T"], ["A", "T"]],
        num_samples=2,
        call_genotype=np.zeros((3, 2, 2), dtype=np.int8),
    )


def _info_vcz(dtype):
    kwargs = _base_kwargs()
    kwargs["info_fields"] = {"DP": np.array([1.5, 2.5, 3.5], dtype=dtype)}
    return make_vcz(**kwargs)


def _format_vcz(dtype):
    kwargs = _base_kwargs()
    values = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]], dtype=dtype)
    kwargs["call_fields"] = {"AB": values}
    return make_vcz(**kwargs)


def _qual_vcz(dtype):
    kwargs = _base_kwargs()
    kwargs["variant_quality"] = np.array([1.5, 2.5, 3.5], dtype=dtype)
    return make_vcz(**kwargs)


class TestViewNonFloat32:
    """``view`` (write_vcf) currently rejects non-float32 floats.

    INFO/FORMAT fields are caught in the Python encoder type mapping
    (vcf_writer._array_to_vcf_type), QUAL in the C dtype check.
    """

    @pytest.mark.parametrize("dtype", NON_FLOAT32_DTYPES)
    def test_info_field_rejected(self, dtype):
        reader = VczReader(_info_vcz(dtype))
        with pytest.raises(ValueError, match="Unsupported dtype"):
            write_vcf(reader, io.StringIO(), no_version=True)

    @pytest.mark.parametrize("dtype", NON_FLOAT32_DTYPES)
    def test_format_field_rejected(self, dtype):
        reader = VczReader(_format_vcz(dtype))
        with pytest.raises(ValueError, match="Unsupported dtype"):
            write_vcf(reader, io.StringIO(), no_version=True)

    @pytest.mark.parametrize("dtype", NON_FLOAT32_DTYPES)
    def test_qual_rejected(self, dtype):
        reader = VczReader(_qual_vcz(dtype))
        with pytest.raises(ValueError, match="Wrong dtype for qual"):
            write_vcf(reader, io.StringIO(), no_version=True)


class TestQueryNonFloat32:
    """``query`` formats floats in Python.

    Scalar-per-variant access (INFO, QUAL) routes through
    ``utils.missing``, whose ``view(np.int32)`` fails on a 0-d
    non-float32 value. The FORMAT sample-loop path does not hit that
    check and currently produces output.
    """

    @pytest.mark.parametrize("dtype", NON_FLOAT32_DTYPES)
    def test_info_field_query(self, dtype):
        reader = VczReader(_info_vcz(dtype))
        formatter = QueryFormatter("%INFO/DP\n", reader)
        with pytest.raises(ValueError, match="Changing the dtype of a 0d array"):
            "".join(formatter.format_variant(v) for v in reader.variants())

    @pytest.mark.parametrize("dtype", NON_FLOAT32_DTYPES)
    def test_format_field_query(self, dtype):
        reader = VczReader(_format_vcz(dtype))
        formatter = QueryFormatter("[%AB ]\n", reader)
        result = "".join(formatter.format_variant(v) for v in reader.variants())
        assert result == "1.5 2.5 \n3.5 4.5 \n5.5 6.5 \n"

    @pytest.mark.parametrize("dtype", NON_FLOAT32_DTYPES)
    def test_qual_query(self, dtype):
        reader = VczReader(_qual_vcz(dtype))
        formatter = QueryFormatter("%QUAL\n", reader)
        with pytest.raises(ValueError, match="Changing the dtype of a 0d array"):
            "".join(formatter.format_variant(v) for v in reader.variants())


class TestFilterNonFloat32:
    """``VczReader`` filtering is the tolerant path.

    ``bcftools_filter._missing_mask`` uses ``np.isnan`` and comparisons
    use native numpy, so filtering on a non-float32 float field
    currently works correctly for both dtypes.
    """

    @pytest.mark.parametrize("dtype", NON_FLOAT32_DTYPES)
    def test_info_filter_comparison(self, dtype):
        data = {"variant_DP": np.array([0.2, 1.5, 3.5], dtype=dtype)}
        fee = bcftools_filter.BcftoolsFilter(field_names=set(data), include="DP>0.5")
        result = fee.evaluate(data)
        np.testing.assert_array_equal(result, [False, True, True])

    @pytest.mark.parametrize("dtype", NON_FLOAT32_DTYPES)
    def test_missing_detection(self, dtype):
        # A NaN element registers as missing via np.isnan and is
        # excluded from an inclusion filter regardless of float width.
        data = {"variant_DP": np.array([0.2, 1.5, np.nan], dtype=dtype)}
        fee = bcftools_filter.BcftoolsFilter(field_names=set(data), include="DP>0.0")
        result = fee.evaluate(data)
        np.testing.assert_array_equal(result, [True, True, False])

    @pytest.mark.parametrize("dtype", NON_FLOAT32_DTYPES)
    def test_filter_end_to_end(self, dtype):
        reader = VczReader(_info_vcz(dtype))
        field_names = reader.field_names | reader.virtual_field_names
        fee = bcftools_filter.BcftoolsFilter(field_names=field_names, include="DP>2.0")
        reader.set_variant_filter(fee)
        reader.materialise_variant_filter()
        positions = []
        for chunk in reader.variant_chunks(fields=["variant_position"]):
            positions.extend(chunk["variant_position"].tolist())
        assert positions == [20, 30]
