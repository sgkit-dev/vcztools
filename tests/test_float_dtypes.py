"""Round-trip tests for half- and double-precision float input fields.

vcztools converts every floating-point field to canonical float32 at the
read boundary (``utils.to_vcf_float32``), relabelling the width-generalised
missing / end-of-vector sentinels to the float32 sentinels. These tests show
that float16 and float64 input round-trips correctly through ``view``,
``query`` and ``VczReader`` filtering, including ragged columns padded with
the fill sentinel, and that the output is identical to an equivalent float32
store.
"""

import io

import numpy as np
import pytest

from vcztools import bcftools_filter, constants
from vcztools.query import QueryFormatter
from vcztools.retrieval import VczReader
from vcztools.vcf_writer import write_vcf

from .vcz_builder import make_vcz

NON_FLOAT32_DTYPES = [np.float16, np.float64]

# (missing, fill) sentinel pair for each supported float width.
SENTINELS = {
    np.float16: (constants.FLOAT16_MISSING, constants.FLOAT16_FILL),
    np.float32: (constants.FLOAT32_MISSING, constants.FLOAT32_FILL),
    np.float64: (constants.FLOAT64_MISSING, constants.FLOAT64_FILL),
}


def _store(dtype, *, info=None, fmt=None, qual=None):
    """Build an in-memory VCZ with a float field of the given dtype."""
    kwargs = dict(
        variant_contig=[0, 0, 0],
        variant_position=[10, 20, 30],
        alleles=[["A", "C"], ["G", "T"], ["A", "T"]],
        num_samples=2,
        call_genotype=np.zeros((3, 2, 2), dtype=np.int8),
    )
    if info is not None:
        kwargs["info_fields"] = {"DP": np.asarray(info, dtype=dtype)}
    if fmt is not None:
        kwargs["call_fields"] = {"AB": np.asarray(fmt, dtype=dtype)}
    if qual is not None:
        kwargs["variant_quality"] = np.asarray(qual, dtype=dtype)
    return make_vcz(**kwargs)


def _view(group):
    out = io.StringIO()
    write_vcf(VczReader(group), out, no_version=True)
    return out.getvalue()


def _rows(group):
    return [line for line in _view(group).splitlines() if not line.startswith("#")]


def _query(group, query_format):
    reader = VczReader(group)
    # write_query enables the float32 cast; mirror that here.
    reader.set_cast_float32()
    formatter = QueryFormatter(query_format, reader)
    return "".join(formatter.format_variant(v) for v in reader.variants())


class TestViewNonFloat32:
    """``view`` now encodes non-float32 float fields correctly."""

    @pytest.mark.parametrize("dtype", NON_FLOAT32_DTYPES)
    def test_info_field(self, dtype):
        group = _store(dtype, info=[1.5, 2.5, 3.5])
        info_values = [row.split("\t")[7] for row in _rows(group)]
        assert info_values == ["DP=1.5", "DP=2.5", "DP=3.5"]

    @pytest.mark.parametrize("dtype", NON_FLOAT32_DTYPES)
    def test_format_field(self, dtype):
        group = _store(dtype, fmt=[[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]])
        # The sample column is "GT:AB"; take the AB sub-field.
        first_sample = [row.split("\t")[9].split(":")[1] for row in _rows(group)]
        assert first_sample == ["1.5", "3.5", "5.5"]

    @pytest.mark.parametrize("dtype", NON_FLOAT32_DTYPES)
    def test_qual_field(self, dtype):
        group = _store(dtype, qual=[10.0, 20.0, 30.0])
        qual_values = [row.split("\t")[5] for row in _rows(group)]
        assert qual_values == ["10", "20", "30"]


class TestQueryNonFloat32:
    """``query`` now formats non-float32 float fields without error."""

    @pytest.mark.parametrize("dtype", NON_FLOAT32_DTYPES)
    def test_info_field_query(self, dtype):
        group = _store(dtype, info=[1.5, 2.5, 3.5])
        assert _query(group, "%INFO/DP\n") == "1.5\n2.5\n3.5\n"

    @pytest.mark.parametrize("dtype", NON_FLOAT32_DTYPES)
    def test_format_field_query(self, dtype):
        group = _store(dtype, fmt=[[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]])
        assert _query(group, "[%AB ]\n") == "1.5 2.5 \n3.5 4.5 \n5.5 6.5 \n"

    @pytest.mark.parametrize("dtype", NON_FLOAT32_DTYPES)
    def test_qual_query(self, dtype):
        group = _store(dtype, qual=[10.0, 20.0, 30.0])
        assert _query(group, "%QUAL\n") == "10\n20\n30\n"


class TestFilterNonFloat32:
    """Filtering operates on the converted float32 data."""

    @pytest.mark.parametrize("dtype", NON_FLOAT32_DTYPES)
    def test_info_filter_comparison(self, dtype):
        data = {"variant_DP": np.array([0.5, 1.5, 3.5], dtype=dtype)}
        fee = bcftools_filter.BcftoolsFilter(field_names=set(data), include="DP>1.0")
        result = fee.evaluate(data)
        np.testing.assert_array_equal(result, [False, True, True])

    @pytest.mark.parametrize("dtype", NON_FLOAT32_DTYPES)
    def test_missing_and_fill_excluded(self, dtype):
        # Both the missing and the fill sentinel are NaN, so an inclusion
        # filter excludes them regardless of float width.
        missing, fill = SENTINELS[dtype]
        data = {"variant_DP": np.array([1.5, missing, fill], dtype=dtype)}
        fee = bcftools_filter.BcftoolsFilter(field_names=set(data), include="DP>0.0")
        result = fee.evaluate(data)
        np.testing.assert_array_equal(result, [True, False, False])

    @pytest.mark.parametrize("dtype", NON_FLOAT32_DTYPES)
    def test_filter_end_to_end(self, dtype):
        group = _store(dtype, info=[1.5, 2.5, 3.5])
        reader = VczReader(group)
        field_names = reader.field_names | reader.virtual_field_names
        fee = bcftools_filter.BcftoolsFilter(field_names=field_names, include="DP>2.0")
        reader.set_variant_filter(fee)
        reader.materialise_variant_filter()
        positions = []
        for chunk in reader.variant_chunks(fields=["variant_position"]):
            positions.extend(chunk["variant_position"].tolist())
        assert positions == [20, 30]


class TestRaggedColumn:
    """A multi-valued float field padded with the fill sentinel.

    Row 0 is full ``[1.5, 2.5]``; row 1 has a trailing fill (one value);
    row 2 is missing then fill (no values).
    """

    def _ragged_store(self, dtype):
        missing, fill = SENTINELS[dtype]
        af = np.array(
            [[1.5, 2.5], [3.5, fill], [missing, fill]],
            dtype=dtype,
        )
        return make_vcz(
            variant_contig=[0, 0, 0],
            variant_position=[10, 20, 30],
            alleles=[["A", "C"], ["G", "T"], ["A", "T"]],
            num_samples=1,
            call_genotype=np.zeros((3, 1, 2), dtype=np.int8),
            info_fields={"AF": af},
        )

    @pytest.mark.parametrize("dtype", NON_FLOAT32_DTYPES)
    def test_view_drops_fill_and_renders_missing(self, dtype):
        group = self._ragged_store(dtype)
        info_values = [row.split("\t")[7] for row in _rows(group)]
        # Trailing fill truncates the vector; an all-missing row renders ".".
        assert info_values == ["AF=1.5,2.5", "AF=3.5", "."]


class TestFloat32CastOption:
    """The float32 cast is opt-in: by default the reader preserves the
    stored float width; ``set_cast_float32`` casts to canonical float32.

    0.1 is not exactly representable in float32, so truncation is observable.
    """

    @staticmethod
    def _read_dp(group, *, cast):
        reader = VczReader(group)
        if cast:
            reader.set_cast_float32()
        chunk = next(reader.variant_chunks(fields=["variant_DP"]))
        return chunk["variant_DP"]

    def test_float64_preserved_by_default(self):
        group = _store(np.float64, info=[0.1, 0.2, 0.3])
        dp = self._read_dp(group, cast=False)
        assert dp.dtype == np.float64
        np.testing.assert_array_equal(dp, np.array([0.1, 0.2, 0.3], dtype=np.float64))

    def test_float16_preserved_by_default(self):
        group = _store(np.float16, info=[0.1, 0.2, 0.3])
        dp = self._read_dp(group, cast=False)
        assert dp.dtype == np.float16
        np.testing.assert_array_equal(dp, np.array([0.1, 0.2, 0.3], dtype=np.float16))

    def test_cast_option_truncates_to_float32(self):
        group = _store(np.float64, info=[0.1, 0.2, 0.3])
        dp = self._read_dp(group, cast=True)
        assert dp.dtype == np.float32
        np.testing.assert_array_equal(dp, np.array([0.1, 0.2, 0.3], dtype=np.float32))

    def test_cast_relabels_sentinels(self):
        missing, fill = SENTINELS[np.float64]
        group = _store(np.float64, info=[1.5, missing, fill])
        dp = self._read_dp(group, cast=True)
        assert dp.dtype == np.float32
        as_int32 = dp.view(np.int32)
        assert as_int32[1] == constants.FLOAT32_MISSING_AS_INT32
        assert as_int32[2] == constants.FLOAT32_FILL_AS_INT32


class TestFloat32Parity:
    """float16 / float64 input produces output identical to float32.

    Values are exactly representable in float16, so all three widths are
    bit-identical after conversion and the rendered text matches exactly.
    """

    @pytest.mark.parametrize("dtype", NON_FLOAT32_DTYPES)
    def test_view_parity(self, dtype):
        kwargs = dict(
            info=[0.5, 1.5, 2.5],
            fmt=[[0.5, 1.5], [2.5, 0.5], [1.5, 2.5]],
            qual=[0.5, 1.5, 2.5],
        )
        assert _view(_store(dtype, **kwargs)) == _view(_store(np.float32, **kwargs))

    @pytest.mark.parametrize("dtype", NON_FLOAT32_DTYPES)
    def test_query_parity(self, dtype):
        kwargs = dict(info=[0.5, 1.5, 2.5], qual=[0.5, 1.5, 2.5])
        query_format = "%POS %QUAL %INFO/DP\n"
        expected = _query(_store(np.float32, **kwargs), query_format)
        assert _query(_store(dtype, **kwargs), query_format) == expected

    @pytest.mark.parametrize("dtype", NON_FLOAT32_DTYPES)
    def test_ragged_view_parity(self, dtype):
        def store(dt):
            missing, fill = SENTINELS[dt]
            af = np.array([[0.5, 1.5], [2.5, fill], [missing, fill]], dtype=dt)
            return make_vcz(
                variant_contig=[0, 0, 0],
                variant_position=[10, 20, 30],
                alleles=[["A", "C"], ["G", "T"], ["A", "T"]],
                num_samples=1,
                call_genotype=np.zeros((3, 1, 2), dtype=np.int8),
                info_fields={"AF": af},
            )

        assert _view(store(dtype)) == _view(store(np.float32))
