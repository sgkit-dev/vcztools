"""Systematic coverage of every supported dtype and field shape.

These tests drive a single in-memory store (``vcz_builder.make_dtype_matrix``)
that carries each supported integer width (int8/16/32), float width
(float16/32/64), String, Character and Flag, in three shapes — per-variant
scalar, fixed inner dimension, and ragged (fill-padded) — across both the
INFO (``variant_*``) and FORMAT (``call_*``) axes. Values follow an ``arange``
scheme with missing / end-of-vector sentinels woven in at known positions.

The central technique is a width-collapsed *reference* store (every integer
field as int32, every float field as float32, same values and sentinel
positions): the narrow widths are correct exactly when their rendered output
matches the reference. Read-back, header, view, query and filter are each
checked this way, with a handful of explicit assertions pinning the
missing-rendering and ragged-trimming behaviour the parity check assumes.
"""

import io

import numpy as np
import pytest

from vcztools import bcftools_filter
from vcztools.query import QueryFormatter
from vcztools.retrieval import VczReader
from vcztools.vcf_writer import _generate_header, write_vcf

from . import vcz_builder
from .utils import make_reader

_FIELDS = vcz_builder.dtype_matrix_fields()
_NUMERIC_INFO_SCALAR_FIELDS = [
    field
    for field in _FIELDS
    if field.category == "INFO"
    and field.shape_kind == "scalar"
    and field.vcf_type in ("Integer", "Float")
]
# Fields whose bare tag is renderable by both view and query (numeric and
# StringDType); the bytes / character / flag singletons are checked only via
# read-back and the header.
_PARITY_INFO_FIELDS = [
    field
    for field in _FIELDS
    if field.category == "INFO"
    and field.vcf_type in ("Integer", "Float", "String")
    and field.vcf_id.startswith(("IA", "IB", "IC", "ID", "FA", "FB", "FC", "SA"))
]
# Every fixed-width field (everything that is not a VLEN StringDType, kind "T").
_FIXED_WIDTH_FIELDS = [field for field in _FIELDS if field.dtype.kind != "T"]
# Fixed-width numeric fields wider than one byte: their bytes codec records an
# explicit byte order, so a big-endian build is observable in the metadata.
_MULTIBYTE_NUMERIC_FIELDS = [
    field
    for field in _FIELDS
    if field.dtype.kind in ("i", "f") and field.dtype.itemsize > 1
]
# Variable-length string fields use the byte-order-neutral VLenUTF8 codec.
_STRING_FIELDS = [field for field in _FIELDS if field.dtype.kind == "T"]


def _view(group):
    out = io.StringIO()
    write_vcf(VczReader(group), out, no_version=True)
    return out.getvalue()


def _rows(group):
    return [line for line in _view(group).splitlines() if not line.startswith("#")]


def _query(group, query_format):
    reader = VczReader(group)
    formatter = QueryFormatter(query_format, reader)
    return "".join(formatter.format_variant(v) for v in reader.variants())


def _info_dict(row):
    """Parse the INFO column of a VCF data row into a dict; valueless keys
    (flags) map to ``True``."""
    info_column = row.split("\t")[7]
    entries = {}
    for entry in info_column.split(";"):
        if "=" in entry:
            key, value = entry.split("=", 1)
            entries[key] = value
        else:
            entries[entry] = True
    return entries


def _format_samples(row):
    """Return ``(format_keys, [per-sample dict, ...])`` for a VCF data row."""
    columns = row.split("\t")
    format_keys = columns[8].split(":")
    samples = []
    for sample_column in columns[9:]:
        values = sample_column.split(":")
        samples.append(dict(zip(format_keys, values)))
    return format_keys, samples


class TestReadBack:
    """The reader preserves the stored dtype and the exact bytes (sentinels
    included) of every field on every axis."""

    @pytest.mark.parametrize("field", _FIELDS, ids=lambda f: f"{f.category}/{f.vcf_id}")
    def test_dtype_and_bytes_preserved(self, fx_dtype_matrix, field):
        data = vcz_builder.dtype_matrix_arrays()
        if field.category == "INFO":
            expected = data.info_fields[field.vcf_id]
        else:
            expected = data.call_fields[field.vcf_id]

        reader = VczReader(fx_dtype_matrix)
        chunks = list(reader.variant_chunks(fields=[field.array_name]))
        read = np.concatenate([chunk[field.array_name] for chunk in chunks], axis=0)

        assert read.dtype == field.dtype
        expected = np.ascontiguousarray(expected)
        assert read.tobytes() == expected.tobytes()


class TestHeader:
    """Each field announces the expected VCF Number and Type."""

    @pytest.mark.parametrize("field", _FIELDS, ids=lambda f: f"{f.category}/{f.vcf_id}")
    def test_number_and_type(self, fx_dtype_matrix, field):
        header = _generate_header(VczReader(fx_dtype_matrix), no_version=True)
        prefix = f"##{field.category}=<ID={field.vcf_id},"
        matching = [line for line in header.splitlines() if line.startswith(prefix)]
        assert len(matching) == 1
        line = matching[0]
        assert f"Number={field.vcf_number}," in line
        assert f"Type={field.vcf_type}," in line


class TestViewParity:
    """The narrow widths render identically to the int32 / float32 reference,
    and the trimming / missing behaviour the parity relies on is pinned by
    explicit assertions."""

    def test_full_view_parity(self, fx_dtype_matrix, fx_dtype_matrix_reference):
        assert _view(fx_dtype_matrix) == _view(fx_dtype_matrix_reference)

    @pytest.mark.parametrize("code", ["IA", "IB", "IC", "FA", "FB", "FC"])
    def test_info_scalar_missing_omitted(self, fx_dtype_matrix, code):
        # The last variant's scalar value is the missing sentinel, so the INFO
        # key is absent from that row but present on the first.
        rows = _rows(fx_dtype_matrix)
        assert f"{code}S" in _info_dict(rows[0])
        assert f"{code}S" not in _info_dict(rows[-1])

    def test_info_ragged_trims_trailing_fill(self, fx_dtype_matrix):
        # Ragged real-count cycles 3, 2, 1, 0 over the first four variants.
        rows = _rows(fx_dtype_matrix)
        assert _info_dict(rows[0])["IAR"] == "0,1,2"
        assert _info_dict(rows[1])["IAR"] == "3,4"
        assert _info_dict(rows[2])["IAR"] == "6"
        assert "IAR" not in _info_dict(rows[3])

    def test_info_ragged_float_trims(self, fx_dtype_matrix):
        rows = _rows(fx_dtype_matrix)
        assert _info_dict(rows[0])["FAR"] == "0.5,1.5,2.5"
        assert _info_dict(rows[1])["FAR"] == "3.5,4.5"

    def test_format_scalar_missing_rendered(self, fx_dtype_matrix):
        # The last sample of the last variant is the missing sentinel.
        rows = _rows(fx_dtype_matrix)
        _, samples = _format_samples(rows[-1])
        assert samples[-1]["IAS"] == "."

    def test_format_ragged_all_missing_sample(self, fx_dtype_matrix):
        # FORMAT ragged real-count is keyed on (variant + sample); variant 7,
        # sample 0 has count 0 and renders as a single ".".
        rows = _rows(fx_dtype_matrix)
        _, samples = _format_samples(rows[7])
        assert samples[0]["IAR"] == "."


class TestQueryParity:
    """``query`` formats every renderable field identically to the reference
    store, including subfield indexing and missing rendering."""

    def test_full_query_parity(self, fx_dtype_matrix, fx_dtype_matrix_reference):
        info_tags = [f"%{field.vcf_id}" for field in _PARITY_INFO_FIELDS]
        format_tags = [f"[%{field.vcf_id} ]" for field in _PARITY_INFO_FIELDS]
        query_format = "\t".join(["%POS", *info_tags, *format_tags]) + "\n"
        assert _query(fx_dtype_matrix, query_format) == _query(
            fx_dtype_matrix_reference, query_format
        )

    @pytest.mark.parametrize("code", ["IA", "FA", "SA"])
    def test_ragged_subfield_indexing(self, fx_dtype_matrix, code):
        # Subfield 0 of the ragged field; the all-missing row (variant 3)
        # renders ".".
        result = _query(fx_dtype_matrix, f"%{code}R{{0}}\n")
        lines = result.splitlines()
        assert lines[3] == "."
        assert lines[0] != "."

    def test_scalar_missing_renders_dot(self, fx_dtype_matrix):
        result = _query(fx_dtype_matrix, "%IAS\n")
        assert result.splitlines()[-1] == "."


class TestFilter:
    """Filtering happens at the stored width: an inclusion expression selects
    the same variants for every integer and float width, and the missing
    sentinel is excluded."""

    @pytest.mark.parametrize(
        "field", _NUMERIC_INFO_SCALAR_FIELDS, ids=lambda f: f.vcf_id
    )
    def test_scalar_filter_parity(
        self, fx_dtype_matrix, fx_dtype_matrix_reference, field
    ):
        # The bare tag is defined on both axes; disambiguate to the INFO field.
        expression = f"INFO/{field.vcf_id}>3"
        native = self._filtered_positions(fx_dtype_matrix, expression)
        reference = self._filtered_positions(fx_dtype_matrix_reference, expression)
        assert native == reference

    def test_scalar_filter_expected(self, fx_dtype_matrix):
        # IAS scalar values are 0..6 then missing; >3 keeps positions 50,60,70.
        positions = self._filtered_positions(fx_dtype_matrix, "INFO/IAS>3")
        assert positions == [50, 60, 70]

    @staticmethod
    def _filtered_positions(group, expression):
        reader = VczReader(group)
        fee = bcftools_filter.BcftoolsFilter(reader, include=expression)
        reader.set_variant_filter(fee)
        reader.materialise_variant_filter()
        positions = []
        for chunk in reader.variant_chunks(fields=["variant_position"]):
            positions.extend(chunk["variant_position"].tolist())
        return positions


class TestInt64:
    """int64 fields read, filter and query at native width; the VCF text
    encoder casts them to int32 and raises a clear error when a value does
    not fit. The dtype matrix already covers in-range int64 via the ID
    fields; these cases pin the large-value and overflow behaviour the matrix
    cannot (its values all fit in int32)."""

    _BIG = 5_000_000_000  # > 2**31, requires int64

    @staticmethod
    def _info_store(values, *, position_dtype=None, positions=None):
        values = np.asarray(values, dtype=np.int64)
        num_variants = len(values)
        if positions is None:
            positions = [(i + 1) * 10 for i in range(num_variants)]
        return vcz_builder.make_vcz(
            variant_contig=[0] * num_variants,
            variant_position=positions,
            alleles=[["A", "C"]] * num_variants,
            num_samples=1,
            call_genotype=np.zeros((num_variants, 1, 2), dtype=np.int8),
            info_fields={"BIG": values},
            position_dtype=position_dtype,
        )

    def test_view_info_overflow_raises(self):
        group = self._info_store([np.iinfo(np.int32).max + 1])
        with pytest.raises(ValueError, match="INFO/BIG"):
            _view(group)

    def test_view_info_in_range_casts(self):
        # In-range int64 values render exactly like int32.
        group = self._info_store([5, 6, 7])
        values = [_info_dict(row)["BIG"] for row in _rows(group)]
        assert values == ["5", "6", "7"]

    def test_query_large_value(self):
        # query never reaches the int32 encoder, so the full int64 value shows.
        group = self._info_store([self._BIG, 1, 2])
        reader = VczReader(group)
        formatter = QueryFormatter("%BIG\n", reader)
        result = "".join(formatter.format_variant(v) for v in reader.variants())
        assert result == f"{self._BIG}\n1\n2\n"

    def test_filter_large_value(self):
        # Filtering compares at int64 width; only the big-value variant passes.
        group = self._info_store([self._BIG, 1, 2])
        reader = VczReader(group)
        fee = bcftools_filter.BcftoolsFilter(
            reader, include=f"INFO/BIG>{self._BIG - 1}"
        )
        reader.set_variant_filter(fee)
        reader.materialise_variant_filter()
        positions = []
        for chunk in reader.variant_chunks(fields=["variant_position"]):
            positions.extend(chunk["variant_position"].tolist())
        assert positions == [10]

    def test_int64_position_preserved_and_queried(self):
        positions = [self._BIG, self._BIG + 100]
        group = self._info_store([1, 2], position_dtype=np.int64, positions=positions)
        reader = VczReader(group)
        chunk = next(reader.variant_chunks(fields=["variant_position"]))
        assert chunk["variant_position"].dtype == np.int64

        formatter = QueryFormatter("%POS\n", reader)
        result = "".join(formatter.format_variant(v) for v in reader.variants())
        assert result == f"{self._BIG}\n{self._BIG + 100}\n"

    def test_int64_position_region_filter(self):
        positions = [self._BIG, self._BIG + 100, self._BIG + 200]
        group = self._info_store(
            [1, 2, 3], position_dtype=np.int64, positions=positions
        )
        region = f"chr1:{self._BIG + 50}-{self._BIG + 150}"
        reader = make_reader(group, regions=region)
        selected = []
        for chunk in reader.variant_chunks(fields=["variant_position"]):
            selected.extend(chunk["variant_position"].tolist())
        assert selected == [self._BIG + 100]

    def test_view_int64_position_overflow_raises(self):
        group = self._info_store(
            [1], position_dtype=np.int64, positions=[np.iinfo(np.int32).max + 1]
        )
        with pytest.raises(ValueError, match="POS"):
            _view(group)


class TestEndianParity:
    """A store whose fixed-width arrays are serialized big-endian reads back
    identically to the native (default-endian) store across every supported
    dtype. Byte order is a property of the zarr bytes codec, not the dtype, and
    the codec normalizes to native byte order on decode, so the reader and the
    encoder always see native-order arrays regardless of the on-disk order."""

    @staticmethod
    def _codecs(array):
        return array.metadata.to_dict()["codecs"]

    @staticmethod
    def _readback(group, field):
        reader = VczReader(group)
        chunks = list(reader.variant_chunks(fields=[field.array_name]))
        return np.concatenate([chunk[field.array_name] for chunk in chunks], axis=0)

    @pytest.mark.parametrize(
        "field", _MULTIBYTE_NUMERIC_FIELDS, ids=lambda f: f"{f.category}/{f.vcf_id}"
    )
    def test_numeric_fields_stored_big_endian(self, fx_dtype_matrix_big_endian, field):
        # Guard against a vacuous parity test: the store really is big-endian.
        codecs = self._codecs(fx_dtype_matrix_big_endian[field.array_name])
        assert {"name": "bytes", "configuration": {"endian": "big"}} in codecs

    @pytest.mark.parametrize(
        "field", _STRING_FIELDS, ids=lambda f: f"{f.category}/{f.vcf_id}"
    )
    def test_string_fields_keep_vlen_codec(self, fx_dtype_matrix_big_endian, field):
        # VLEN strings have no byte order, so they stay on the VLenUTF8 codec.
        codecs = self._codecs(fx_dtype_matrix_big_endian[field.array_name])
        assert all(codec["name"] != "bytes" for codec in codecs)

    def test_full_view_parity(self, fx_dtype_matrix, fx_dtype_matrix_big_endian):
        assert _view(fx_dtype_matrix_big_endian) == _view(fx_dtype_matrix)

    def test_full_query_parity(self, fx_dtype_matrix, fx_dtype_matrix_big_endian):
        info_tags = [f"%{field.vcf_id}" for field in _PARITY_INFO_FIELDS]
        format_tags = [f"[%{field.vcf_id} ]" for field in _PARITY_INFO_FIELDS]
        query_format = "\t".join(["%POS", *info_tags, *format_tags]) + "\n"
        assert _query(fx_dtype_matrix_big_endian, query_format) == _query(
            fx_dtype_matrix, query_format
        )

    @pytest.mark.parametrize(
        "field", _FIXED_WIDTH_FIELDS, ids=lambda f: f"{f.category}/{f.vcf_id}"
    )
    def test_readback_native_byteorder(
        self, fx_dtype_matrix, fx_dtype_matrix_big_endian, field
    ):
        native = self._readback(fx_dtype_matrix, field)
        big = self._readback(fx_dtype_matrix_big_endian, field)
        # The reader yields native byte order, and the decoded bytes match the
        # default-endian store despite the big-endian on-disk serialization.
        assert big.dtype.isnative
        assert big.tobytes() == native.tobytes()
