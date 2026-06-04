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
    and field.vcf_id.startswith(("IA", "IB", "IC", "FA", "FB", "FC", "SA"))
]


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
        field_names = reader.field_names | reader.virtual_field_names
        fee = bcftools_filter.BcftoolsFilter(
            field_names=field_names, include=expression
        )
        reader.set_variant_filter(fee)
        reader.materialise_variant_filter()
        positions = []
        for chunk in reader.variant_chunks(fields=["variant_position"]):
            positions.extend(chunk["variant_position"].tolist())
        return positions
