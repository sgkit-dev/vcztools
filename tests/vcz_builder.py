"""
Build tiny synthetic VCZ groups in memory for fast unit tests.

The builder produces a `zarr.Group` rooted in a `MemoryStore`, populated
with the minimum schema vcztools reads. Arrays use no compression codecs
so builds are cheap — the point is to exercise chunk-boundary and
retrieval/filter code paths without round-tripping a real VCF through
bio2zarr.

Use this only in tests where VCF parity against bcftools/cyvcf2 is not
the subject of the test. Parity tests still need real fixtures under
``tests/data/vcf``.
"""

import dataclasses
import warnings

import numpy as np
import zarr
import zarr.codecs
import zarr.storage

from vcztools import constants

_NO_COMPRESSION = [zarr.codecs.BytesCodec()]


# Default per-field chunk-size multipliers over ``variants_chunk_size``,
# applied unless ``field_chunk_overrides`` supplies an explicit size.
# Production VCZ stores (e.g. ``performance/long_bench.vcz``) use
# proportional chunking on variant-only fields: ``call_*`` defines the
# minimum chunk size and variant-only fields get bigger chunks tuned to
# their per-row footprint. The multiplier set below mirrors that shape
# at test-fixture scale and deliberately mixes pairwise-non-multiple
# values (e.g. ``2 × v_chunk`` and ``3 × v_chunk``) so the GCD-vs-min
# stream-chunk-size regime is exercised by every test that builds a
# fixture without explicit chunk-size overrides.
DEFAULT_VARIANT_CHUNK_MULTIPLIERS: dict[str, int] = {
    "variant_position": 3,
    "variant_allele": 2,
    "variant_contig": 6,
    "variant_length": 4,
    "variant_filter": 2,
    "variant_id": 2,
    "variant_quality": 2,
}


def _create_array(group, name, data, *, chunks, dimension_names):
    arr = group.create_array(
        name=name,
        shape=data.shape,
        chunks=chunks,
        dtype=data.dtype,
        dimension_names=dimension_names,
        compressors=None,
        filters=None,
    )
    arr[...] = data
    # Mirror dimension names to the v2-style attr that vcztools.vcf_writer.dims
    # also reads, so the same builder works regardless of which branch is hit.
    arr.attrs["_ARRAY_DIMENSIONS"] = list(dimension_names)
    return arr


def _compute_region_index(
    variant_contig, variant_position, variant_length, min_chunk, dtype=np.int32
):
    """Compute the region index, with one row per logical (``min_chunk``-
    sized) chunk. ``min_chunk`` is the minimum variants-axis chunk
    size (driven by ``call_*`` fields); variant-only fields may have
    larger chunks but the region index always indexes at logical-
    chunk granularity.

    ``dtype`` is the integer dtype of the resulting array; pass ``np.int64``
    when positions exceed the int32 range so the position columns are not
    truncated.
    """
    num_variants = variant_contig.shape[0]
    num_chunks = (num_variants + min_chunk - 1) // min_chunk
    rows = []
    for chunk_idx in range(num_chunks):
        start = chunk_idx * min_chunk
        stop = min(start + min_chunk, num_variants)
        chunk_contigs = variant_contig[start:stop]
        chunk_positions = variant_position[start:stop]
        chunk_ends = chunk_positions + variant_length[start:stop]
        # One row per (chunk, contig) span. The simple case — one contig
        # per chunk — is enough for everything the test suite exercises
        # today; split rows if a chunk straddles contigs.
        span_start = 0
        while span_start < len(chunk_contigs):
            contig_id = int(chunk_contigs[span_start])
            span_stop = span_start + 1
            while (
                span_stop < len(chunk_contigs)
                and int(chunk_contigs[span_stop]) == contig_id
            ):
                span_stop += 1
            span_positions = chunk_positions[span_start:span_stop]
            span_ends = chunk_ends[span_start:span_stop]
            rows.append(
                [
                    chunk_idx,
                    contig_id,
                    int(span_positions[0]),
                    int(span_positions[-1]),
                    int(span_ends.max()),
                    int(span_stop - span_start),
                ]
            )
            span_start = span_stop
    return np.array(rows, dtype=dtype)


def make_vcz(
    *,
    variant_contig,
    variant_position,
    alleles,
    num_samples=0,
    sample_id=None,
    contigs=("chr1",),
    filters=("PASS",),
    filter_descriptions=None,
    variants_chunk_size=None,
    samples_chunk_size=None,
    field_chunk_overrides=None,
    proportional_chunks=True,
    call_genotype=None,
    call_fields=None,
    info_fields=None,
    variant_id=None,
    variant_quality=None,
    variant_filter=None,
    variant_length=None,
    region_index=None,
    ploidy=2,
    position_dtype=None,
):
    """
    Build an in-memory VCZ group and return it.

    All parameters are keyword-only. ``variant_contig``, ``variant_position``
    and ``alleles`` are required; everything else has a sensible default.

    ``field_chunk_overrides`` is an optional ``dict[str, int]`` mapping
    a variant-axis array name (e.g. ``"variant_allele"``,
    ``"variant_id"``) to a chunk size that overrides the default for
    that field. Each override must be a positive integer multiple of
    ``variants_chunk_size`` (the minimum chunk size on the variants
    axis); ``call_*`` fields cannot be overridden.

    Variant-only fields named in :data:`DEFAULT_VARIANT_CHUNK_MULTIPLIERS`
    default to ``multiplier * variants_chunk_size`` so a fixture built
    without explicit overrides matches the proportional chunking shape
    that production stores use. Other variant-only fields (anonymous
    INFO arrays passed via ``info_fields``) default to
    ``variants_chunk_size``. The recipe is applied only for keys not
    present in ``field_chunk_overrides``; pass an entry per field to
    pin specific chunk sizes.

    Set ``proportional_chunks=False`` to disable the recipe entirely
    so every variant-only field defaults to ``variants_chunk_size`` —
    intended for unit tests whose assertions depend on uniform chunking
    (plan counts, multipliers, stream-chunk-size derivations).

    ``variant_quality`` preserves a supplied floating-point dtype (so
    fixtures can build a non-float32 QUAL); non-float inputs default to
    float32.

    ``position_dtype`` sets the integer dtype of ``variant_position`` and the
    auto-computed ``region_index`` (default int32); pass ``np.int64`` to build a
    store with 64-bit positions.
    """
    pos_dtype = position_dtype if position_dtype is not None else np.int32
    variant_contig = np.asarray(variant_contig, dtype=np.int32)
    variant_position = np.asarray(variant_position, dtype=pos_dtype)
    num_variants = variant_contig.shape[0]
    if variant_position.shape[0] != num_variants:
        raise ValueError("variant_contig and variant_position length mismatch")

    max_alleles = max(len(a) for a in alleles) if len(alleles) > 0 else 1
    allele_array = np.full((num_variants, max_alleles), "", dtype="<U16")
    for i, row in enumerate(alleles):
        for j, a in enumerate(row):
            allele_array[i, j] = a

    if variant_length is None:
        variant_length = np.ones(num_variants, dtype=np.int32)
    else:
        variant_length = np.asarray(variant_length, dtype=np.int32)

    if variant_filter is None:
        variant_filter = np.ones((num_variants, len(filters)), dtype=bool)
    else:
        variant_filter = np.asarray(variant_filter, dtype=bool)

    v_chunk = variants_chunk_size if variants_chunk_size is not None else num_variants
    v_chunk = max(v_chunk, 1)
    s_chunk = (
        samples_chunk_size if samples_chunk_size is not None else max(num_samples, 1)
    )

    explicit_overrides = (
        dict(field_chunk_overrides) if field_chunk_overrides is not None else {}
    )
    for name, size in explicit_overrides.items():
        if name.startswith("call_"):
            raise ValueError(
                f"field_chunk_overrides[{name!r}]: call_* fields define the "
                f"minimum variants chunk size and cannot be overridden"
            )
        if size <= 0 or size % v_chunk != 0:
            raise ValueError(
                f"field_chunk_overrides[{name!r}]={size} must be a positive "
                f"multiple of variants_chunk_size={v_chunk}"
            )
    if proportional_chunks:
        overrides = {
            name: factor * v_chunk
            for name, factor in DEFAULT_VARIANT_CHUNK_MULTIPLIERS.items()
        }
    else:
        overrides = {}
    overrides.update(explicit_overrides)

    def _vc(name: str) -> int:
        return overrides.get(name, v_chunk)

    if region_index is None:
        region_index = _compute_region_index(
            variant_contig, variant_position, variant_length, v_chunk, dtype=pos_dtype
        )
    else:
        region_index = np.asarray(region_index, dtype=pos_dtype)

    store = zarr.storage.MemoryStore()
    # zarr emits warnings about unstable V3 specification for fixed-width
    # string dtypes. These are test fixtures, not on-disk artefacts, so
    # silence them locally.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The data type",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore", category=zarr.core.dtype.common.UnstableSpecificationWarning
        )
        root = zarr.group(store=store, zarr_format=3)

        if sample_id is None:
            sample_id_arr = np.array(
                [f"sample_{i}" for i in range(num_samples)], dtype="<U16"
            )
        else:
            sample_id_arr = np.asarray(sample_id, dtype="<U64")
            if sample_id_arr.shape != (num_samples,):
                raise ValueError("sample_id length must match num_samples")
        _create_array(
            root,
            "sample_id",
            sample_id_arr,
            chunks=(s_chunk,),
            dimension_names=("samples",),
        )
        _create_array(
            root,
            "contig_id",
            np.array(list(contigs), dtype="<U32"),
            chunks=(len(contigs),),
            dimension_names=("contigs",),
        )
        _create_array(
            root,
            "filter_id",
            np.array(list(filters), dtype="<U32"),
            chunks=(len(filters),),
            dimension_names=("filters",),
        )
        if filter_descriptions is not None:
            if len(filter_descriptions) != len(filters):
                raise ValueError("filter_descriptions length must match filters length")
            _create_array(
                root,
                "filter_description",
                np.array(list(filter_descriptions), dtype="<U64"),
                chunks=(len(filters),),
                dimension_names=("filters",),
            )
        _create_array(
            root,
            "variant_contig",
            variant_contig,
            chunks=(_vc("variant_contig"),),
            dimension_names=("variants",),
        )
        _create_array(
            root,
            "variant_position",
            variant_position,
            chunks=(_vc("variant_position"),),
            dimension_names=("variants",),
        )
        _create_array(
            root,
            "variant_length",
            variant_length,
            chunks=(_vc("variant_length"),),
            dimension_names=("variants",),
        )
        _create_array(
            root,
            "variant_allele",
            allele_array,
            chunks=(_vc("variant_allele"), max_alleles),
            dimension_names=("variants", "alleles"),
        )
        _create_array(
            root,
            "variant_filter",
            variant_filter,
            chunks=(_vc("variant_filter"), len(filters)),
            dimension_names=("variants", "filters"),
        )
        _create_array(
            root,
            "region_index",
            region_index,
            chunks=region_index.shape,
            dimension_names=("regions", "fields"),
        )

        if variant_id is not None:
            # dtype inferred from input so tests can exercise rsids
            # longer than 16 characters; the production VCZ schema
            # uses whatever width bio2zarr emits for the source data.
            _create_array(
                root,
                "variant_id",
                np.asarray(variant_id),
                chunks=(_vc("variant_id"),),
                dimension_names=("variants",),
            )
        if variant_quality is not None:
            variant_quality = np.asarray(variant_quality)
            if variant_quality.dtype.kind != "f":
                variant_quality = variant_quality.astype(np.float32)
            _create_array(
                root,
                "variant_quality",
                variant_quality,
                chunks=(_vc("variant_quality"),),
                dimension_names=("variants",),
            )

        if num_samples > 0 and call_genotype is not None:
            call_genotype = np.asarray(call_genotype, dtype=np.int8)
            _create_array(
                root,
                "call_genotype",
                call_genotype,
                chunks=(v_chunk, s_chunk, ploidy),
                dimension_names=("variants", "samples", "ploidy"),
            )

        if call_fields is not None:
            for name, data in call_fields.items():
                data = np.asarray(data)
                if data.ndim == 2:
                    dims = ("variants", "samples")
                    chunks = (v_chunk, s_chunk)
                elif data.ndim == 3:
                    dims = ("variants", "samples", f"FORMAT_{name}_dim")
                    chunks = (v_chunk, s_chunk, data.shape[2])
                else:
                    raise ValueError(
                        f"call_fields[{name!r}] must be 2- or 3-dimensional, "
                        f"got ndim={data.ndim}"
                    )
                _create_array(
                    root,
                    f"call_{name}",
                    data,
                    chunks=chunks,
                    dimension_names=dims,
                )

        if info_fields is not None:
            for name, data in info_fields.items():
                data = np.asarray(data)
                dims = (
                    ("variants",)
                    if data.ndim == 1
                    else ("variants", f"INFO_{name}_dim")
                )
                field_v = _vc(f"variant_{name}")
                chunks = (field_v,) if data.ndim == 1 else (field_v, data.shape[1])
                _create_array(
                    root,
                    f"variant_{name}",
                    data,
                    chunks=chunks,
                    dimension_names=dims,
                )

    return root


# Arrays that copy_vcz handles via dedicated make_vcz parameters.
_COPY_BUILTIN_ARRAYS = frozenset(
    {
        "sample_id",
        "contig_id",
        "filter_id",
        "filter_description",
        "region_index",
        "variant_contig",
        "variant_position",
        "variant_length",
        "variant_allele",
        "variant_id",
        "variant_quality",
        "variant_filter",
        "call_genotype",
    }
)


def copy_vcz(
    source,
    *,
    variants_chunk_size=None,
    samples_chunk_size=None,
    field_chunk_overrides=None,
):
    """
    Copy a zarr root group into a fresh in-memory VCZ group by
    delegating to :func:`make_vcz`, optionally rewriting chunk sizes.

    Produces a *full* copy: every array in ``source`` is present in
    the result with the same values. ``variant_*`` arrays not in the
    builtin set are passed through as ``info_fields``; ``call_*``
    arrays (other than ``call_genotype``) are passed through as
    ``call_fields``. ``region_index`` is recomputed from the copied
    inputs so chunk-size overrides produce a correct index.

    When ``field_chunk_overrides`` is omitted, per-field chunk sizes
    on the variants axis are preserved from the source — so a source
    with ``variant_allele.chunks[0] = K * call_genotype.chunks[0]``
    round-trips exactly. Pass an explicit dict to rewrite them.

    Raises :class:`ValueError` if ``source`` contains an array
    :func:`make_vcz` does not know how to construct.
    """

    def _read(name):
        return source[name][...]

    sample_id = _read("sample_id")
    contig_ids = tuple(str(c) for c in _read("contig_id").tolist())
    filter_ids = tuple(str(f) for f in _read("filter_id").tolist())

    filter_descriptions = None
    if "filter_description" in source:
        filter_descriptions = tuple(
            str(d) for d in _read("filter_description").tolist()
        )

    variant_contig = _read("variant_contig")
    variant_position = _read("variant_position")
    variant_length = _read("variant_length") if "variant_length" in source else None
    variant_id = _read("variant_id") if "variant_id" in source else None
    variant_quality = _read("variant_quality") if "variant_quality" in source else None
    variant_filter = _read("variant_filter") if "variant_filter" in source else None

    # variant_allele back to list-of-lists, stripping the trailing ""
    # padding that make_vcz adds for ragged allele rows.
    allele_arr = _read("variant_allele")
    alleles = [[str(a) for a in row if str(a) != ""] for row in allele_arr]

    info_fields = {}
    call_fields = {}
    call_genotype = None
    ploidy = None
    for name in sorted(source.array_keys()):
        if name in _COPY_BUILTIN_ARRAYS:
            continue
        if name.startswith("variant_"):
            info_fields[name[len("variant_") :]] = _read(name)
        elif name.startswith("call_"):
            call_fields[name[len("call_") :]] = _read(name)
        else:
            raise ValueError(f"copy_vcz: source array {name!r} is not supported")

    if "call_genotype" in source:
        call_genotype = _read("call_genotype")
        ploidy = call_genotype.shape[2]

    # When the caller does not override a chunk size, default to the
    # source's call_* chunk size (the minimum). Only when chunks are
    # overridden do we let make_vcz pick fresh defaults (and recompute
    # region_index accordingly).
    if "call_genotype" in source:
        source_min_chunk = int(source["call_genotype"].chunks[0])
    else:
        source_min_chunk = int(source["variant_position"].chunks[0])
    if variants_chunk_size is None:
        variants_chunk_size = source_min_chunk
    if samples_chunk_size is None and "sample_id" in source:
        samples_chunk_size = int(source["sample_id"].chunks[0])

    # When the caller didn't pass field_chunk_overrides and isn't
    # rechunking, default to preserving every variant-axis field's
    # source chunk size. Recording all of them (not just ones that
    # differ from v_chunk) keeps make_vcz's proportional-chunking
    # recipe from re-applying to a field the copy is supposed to
    # preserve.
    caller_provided_overrides = field_chunk_overrides is not None
    if not caller_provided_overrides and variants_chunk_size == source_min_chunk:
        derived_overrides: dict[str, int] = {}
        for name in source.array_keys():
            if name.startswith("call_"):
                continue
            arr = source[name]
            dims = (
                arr.attrs.get("_ARRAY_DIMENSIONS", None) or arr.metadata.dimension_names
            )
            if dims is None or len(dims) == 0 or dims[0] != "variants":
                continue
            derived_overrides[name] = int(arr.chunks[0])
        field_chunk_overrides = derived_overrides or None

    # Preserve the source region_index when chunk sizes match the source
    # and the caller didn't override them; recompute (via make_vcz) when
    # they are overridden, since the chunk_idx column would otherwise be
    # wrong. The recomputed index can also differ in its end-position
    # convention from whatever produced the source (e.g. bio2zarr), so
    # only recompute when the chunk-rewrite truly invalidates it.
    keep_region_index = (
        variants_chunk_size == source_min_chunk
        and not caller_provided_overrides
        and "region_index" in source
    )

    kwargs = dict(
        variant_contig=variant_contig,
        variant_position=variant_position,
        alleles=alleles,
        num_samples=int(sample_id.shape[0]),
        sample_id=sample_id,
        contigs=contig_ids,
        filters=filter_ids,
        filter_descriptions=filter_descriptions,
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
        field_chunk_overrides=field_chunk_overrides,
        call_genotype=call_genotype,
        call_fields=call_fields or None,
        info_fields=info_fields or None,
        variant_id=variant_id,
        variant_quality=variant_quality,
        variant_filter=variant_filter,
        variant_length=variant_length,
        region_index=_read("region_index") if keep_region_index else None,
    )
    if ploidy is not None:
        kwargs["ploidy"] = ploidy
    return make_vcz(**kwargs)


# ---------------------------------------------------------------------------
# Systematic dtype / shape matrix fixture.
#
# A single in-memory store carrying every dtype the encoder branches on
# (each supported integer and float width, String, Character and Flag) in
# every field shape (per-variant scalar, fixed inner dimension, and ragged
# fill-padded), across both the INFO (``variant_*``) and FORMAT (``call_*``)
# axes. Values are derived from ``arange`` so every cell is predictable, and
# missing / end-of-vector sentinels are woven in at known positions so the
# trimming and missing-rendering paths are exercised for every dtype.
#
# Numeric values are chosen to be exactly representable at every width
# (integers and half-integers): a ``reference=True`` store collapses each
# integer field to int32 and each float field to float32 but keeps the same
# values, names and sentinel positions, so its output is byte-for-byte
# identical to the native-width store. Tests assert that equality to prove
# the narrow widths behave like the canonical ones.
# ---------------------------------------------------------------------------

_MATRIX_NUM_VARIANTS = 8
_MATRIX_NUM_SAMPLES = 3
_MATRIX_FIXED_WIDTH = 3
_MATRIX_RAGGED_WIDTH = 3

_MATRIX_INT_TOKENS = {"i1": np.int8, "i2": np.int16, "i4": np.int32, "i8": np.int64}
_MATRIX_FLOAT_TOKENS = {"f2": np.float16, "f4": np.float32, "f8": np.float64}
_MATRIX_STRING_DTYPE = np.dtypes.StringDType()

# Field IDs are restricted to ``[A-Z]+`` because the query grammar only
# recognises uppercase tag names. Each (dtype, shape) gets a three-letter ID:
# a two-letter dtype code plus a shape suffix (S=scalar, F=fixed, R=ragged).
# The same ID names both the INFO (``variant_*``) and FORMAT (``call_*``)
# array, mirroring how a real field such as DP appears on both axes; query
# resolves the bare tag to FORMAT inside ``[]`` and INFO otherwise.
_MATRIX_TOKEN_CODE = {
    "i1": "IA",
    "i2": "IB",
    "i4": "IC",
    "i8": "ID",
    "f2": "FA",
    "f4": "FB",
    "f8": "FC",
}
_MATRIX_STRING_CODE = "SA"

# (suffix, shape_kind, vcf_number) for the three field shapes. A ragged
# field's VCF Number is its stored width, since the writer derives Number
# from shape[-1]; raggedness shows only as trimmed trailing fill on output.
_MATRIX_SHAPE_SPECS = (
    ("S", "scalar", 1),
    ("F", "fixed", _MATRIX_FIXED_WIDTH),
    ("R", "ragged", _MATRIX_RAGGED_WIDTH),
)

_MATRIX_FLOAT_SENTINELS = {
    np.dtype(np.float16): (constants.FLOAT16_MISSING, constants.FLOAT16_FILL),
    np.dtype(np.float32): (constants.FLOAT32_MISSING, constants.FLOAT32_FILL),
    np.dtype(np.float64): (constants.FLOAT64_MISSING, constants.FLOAT64_FILL),
}


@dataclasses.dataclass(frozen=True)
class DtypeMatrixField:
    """Description of one field in the dtype matrix (no values)."""

    vcf_id: str
    category: str  # "INFO" or "FORMAT"
    array_name: str  # variant_<id> or call_<id>
    dtype: np.dtype  # native (non-reference) stored dtype
    shape_kind: str  # "scalar", "fixed" or "ragged"
    vcf_type: str  # Integer / Float / String / Character / Flag
    vcf_number: object  # int (0 for Flag)


@dataclasses.dataclass(frozen=True)
class DtypeMatrixData:
    """The arrays backing a dtype-matrix store."""

    num_variants: int
    num_samples: int
    variant_contig: np.ndarray
    variant_position: np.ndarray
    alleles: list
    call_genotype: np.ndarray
    info_fields: dict
    call_fields: dict


def _matrix_numeric_tokens():
    return list(_MATRIX_INT_TOKENS) + list(_MATRIX_FLOAT_TOKENS)


def _matrix_dtype(token, reference):
    if token in _MATRIX_INT_TOKENS:
        return np.int32 if reference else _MATRIX_INT_TOKENS[token]
    return np.float32 if reference else _MATRIX_FLOAT_TOKENS[token]


def _matrix_sentinels(dtype):
    dtype = np.dtype(dtype)
    if dtype.kind == "i":
        return constants.INT_MISSING, constants.INT_FILL
    return _MATRIX_FLOAT_SENTINELS[dtype]


def _matrix_real(dtype, n):
    # Floats get a half-integer offset so values stay fractional and are
    # exactly representable at every width (float16 included).
    if np.dtype(dtype).kind == "f":
        return n + 0.5
    return n


def _matrix_ragged_count(index):
    # Cycles RAGGED_WIDTH, RAGGED_WIDTH - 1, ..., 0 so every ragged field
    # has full rows, partially filled rows and at least one zero-length
    # (all-missing) row.
    period = _MATRIX_RAGGED_WIDTH + 1
    return _MATRIX_RAGGED_WIDTH - (index % period)


def _matrix_info_scalar(dtype):
    missing, _ = _matrix_sentinels(dtype)
    values = np.empty(_MATRIX_NUM_VARIANTS, dtype=dtype)
    for i in range(_MATRIX_NUM_VARIANTS):
        values[i] = _matrix_real(dtype, i)
    values[_MATRIX_NUM_VARIANTS - 1] = missing
    return values


def _matrix_info_fixed(dtype):
    width = _MATRIX_FIXED_WIDTH
    values = np.empty((_MATRIX_NUM_VARIANTS, width), dtype=dtype)
    for i in range(_MATRIX_NUM_VARIANTS):
        for j in range(width):
            values[i, j] = _matrix_real(dtype, i * width + j)
    return values


def _matrix_info_ragged(dtype):
    missing, fill = _matrix_sentinels(dtype)
    width = _MATRIX_RAGGED_WIDTH
    values = np.full((_MATRIX_NUM_VARIANTS, width), fill, dtype=dtype)
    for i in range(_MATRIX_NUM_VARIANTS):
        count = _matrix_ragged_count(i)
        if count == 0:
            values[i, 0] = missing
        else:
            for j in range(count):
                values[i, j] = _matrix_real(dtype, i * width + j)
    return values


def _matrix_format_scalar(dtype):
    missing, _ = _matrix_sentinels(dtype)
    num_samples = _MATRIX_NUM_SAMPLES
    values = np.empty((_MATRIX_NUM_VARIANTS, num_samples), dtype=dtype)
    for i in range(_MATRIX_NUM_VARIANTS):
        for s in range(num_samples):
            values[i, s] = _matrix_real(dtype, i * num_samples + s)
    values[_MATRIX_NUM_VARIANTS - 1, num_samples - 1] = missing
    return values


def _matrix_format_fixed(dtype):
    num_samples = _MATRIX_NUM_SAMPLES
    width = _MATRIX_FIXED_WIDTH
    values = np.empty((_MATRIX_NUM_VARIANTS, num_samples, width), dtype=dtype)
    for i in range(_MATRIX_NUM_VARIANTS):
        for s in range(num_samples):
            for j in range(width):
                values[i, s, j] = _matrix_real(dtype, (i * num_samples + s) * width + j)
    return values


def _matrix_format_ragged(dtype):
    missing, fill = _matrix_sentinels(dtype)
    num_samples = _MATRIX_NUM_SAMPLES
    width = _MATRIX_RAGGED_WIDTH
    values = np.full((_MATRIX_NUM_VARIANTS, num_samples, width), fill, dtype=dtype)
    for i in range(_MATRIX_NUM_VARIANTS):
        for s in range(num_samples):
            count = _matrix_ragged_count(i + s)
            if count == 0:
                values[i, s, 0] = missing
            else:
                for j in range(count):
                    base = (i * num_samples + s) * width + j
                    values[i, s, j] = _matrix_real(dtype, base)
    return values


def _matrix_info_str_scalar():
    values = np.empty(_MATRIX_NUM_VARIANTS, dtype=_MATRIX_STRING_DTYPE)
    for i in range(_MATRIX_NUM_VARIANTS):
        values[i] = f"v{i}"
    values[_MATRIX_NUM_VARIANTS - 1] = constants.STR_MISSING
    return values


def _matrix_info_str_fixed():
    width = _MATRIX_FIXED_WIDTH
    values = np.empty((_MATRIX_NUM_VARIANTS, width), dtype=_MATRIX_STRING_DTYPE)
    for i in range(_MATRIX_NUM_VARIANTS):
        for j in range(width):
            values[i, j] = f"v{i}c{j}"
    return values


def _matrix_info_str_ragged():
    width = _MATRIX_RAGGED_WIDTH
    values = np.full(
        (_MATRIX_NUM_VARIANTS, width), constants.STR_FILL, dtype=_MATRIX_STRING_DTYPE
    )
    for i in range(_MATRIX_NUM_VARIANTS):
        count = _matrix_ragged_count(i)
        if count == 0:
            values[i, 0] = constants.STR_MISSING
        else:
            for j in range(count):
                values[i, j] = f"v{i}c{j}"
    return values


def _matrix_format_str_scalar():
    num_samples = _MATRIX_NUM_SAMPLES
    values = np.empty((_MATRIX_NUM_VARIANTS, num_samples), dtype=_MATRIX_STRING_DTYPE)
    for i in range(_MATRIX_NUM_VARIANTS):
        for s in range(num_samples):
            values[i, s] = f"s{i}_{s}"
    values[_MATRIX_NUM_VARIANTS - 1, num_samples - 1] = constants.STR_MISSING
    return values


def _matrix_format_str_fixed():
    num_samples = _MATRIX_NUM_SAMPLES
    width = _MATRIX_FIXED_WIDTH
    values = np.empty(
        (_MATRIX_NUM_VARIANTS, num_samples, width), dtype=_MATRIX_STRING_DTYPE
    )
    for i in range(_MATRIX_NUM_VARIANTS):
        for s in range(num_samples):
            for j in range(width):
                values[i, s, j] = f"s{i}_{s}c{j}"
    return values


def _matrix_format_str_ragged():
    num_samples = _MATRIX_NUM_SAMPLES
    width = _MATRIX_RAGGED_WIDTH
    values = np.full(
        (_MATRIX_NUM_VARIANTS, num_samples, width),
        constants.STR_FILL,
        dtype=_MATRIX_STRING_DTYPE,
    )
    for i in range(_MATRIX_NUM_VARIANTS):
        for s in range(num_samples):
            count = _matrix_ragged_count(i + s)
            if count == 0:
                values[i, s, 0] = constants.STR_MISSING
            else:
                for j in range(count):
                    values[i, s, j] = f"s{i}_{s}c{j}"
    return values


def _matrix_info_bytes_scalar():
    values = np.empty(_MATRIX_NUM_VARIANTS, dtype="S4")
    for i in range(_MATRIX_NUM_VARIANTS):
        values[i] = f"b{i}"
    return values


def _matrix_info_char():
    letters = "ACGTNACG"
    values = np.empty(_MATRIX_NUM_VARIANTS, dtype="<U1")
    for i in range(_MATRIX_NUM_VARIANTS):
        values[i] = letters[i]
    return values


def _matrix_info_flag():
    values = np.zeros(_MATRIX_NUM_VARIANTS, dtype=bool)
    for i in range(_MATRIX_NUM_VARIANTS):
        values[i] = i % 2 == 0
    return values


def dtype_matrix_arrays(reference=False):
    """Return the arrays backing the dtype-matrix store as a
    :class:`DtypeMatrixData`.

    With ``reference=True`` every integer field is built as int32 and every
    float field as float32; all other arrays (string, character, flag and the
    fixed schema) are unchanged. Tests use the reference store as the
    width-collapsed oracle the native-width store must match.
    """
    info_fields = {}
    call_fields = {}
    for token in _matrix_numeric_tokens():
        dtype = _matrix_dtype(token, reference)
        code = _MATRIX_TOKEN_CODE[token]
        info_fields[f"{code}S"] = _matrix_info_scalar(dtype)
        info_fields[f"{code}F"] = _matrix_info_fixed(dtype)
        info_fields[f"{code}R"] = _matrix_info_ragged(dtype)
        call_fields[f"{code}S"] = _matrix_format_scalar(dtype)
        call_fields[f"{code}F"] = _matrix_format_fixed(dtype)
        call_fields[f"{code}R"] = _matrix_format_ragged(dtype)

    code = _MATRIX_STRING_CODE
    info_fields[f"{code}S"] = _matrix_info_str_scalar()
    info_fields[f"{code}F"] = _matrix_info_str_fixed()
    info_fields[f"{code}R"] = _matrix_info_str_ragged()
    call_fields[f"{code}S"] = _matrix_format_str_scalar()
    call_fields[f"{code}F"] = _matrix_format_str_fixed()
    call_fields[f"{code}R"] = _matrix_format_str_ragged()

    info_fields["BYT"] = _matrix_info_bytes_scalar()
    info_fields["CHR"] = _matrix_info_char()
    info_fields["FLG"] = _matrix_info_flag()

    num_variants = _MATRIX_NUM_VARIANTS
    num_samples = _MATRIX_NUM_SAMPLES
    variant_contig = np.zeros(num_variants, dtype=np.int32)
    variant_position = (np.arange(num_variants, dtype=np.int32) + 1) * 10
    alleles = [["A", "C"] for _ in range(num_variants)]
    call_genotype = np.zeros((num_variants, num_samples, 2), dtype=np.int8)
    return DtypeMatrixData(
        num_variants=num_variants,
        num_samples=num_samples,
        variant_contig=variant_contig,
        variant_position=variant_position,
        alleles=alleles,
        call_genotype=call_genotype,
        info_fields=info_fields,
        call_fields=call_fields,
    )


def _matrix_axis_pair(code, suffix, dtype, shape_kind, vcf_type, number):
    vcf_id = f"{code}{suffix}"
    info = DtypeMatrixField(
        vcf_id, "INFO", f"variant_{vcf_id}", dtype, shape_kind, vcf_type, number
    )
    call = DtypeMatrixField(
        vcf_id, "FORMAT", f"call_{vcf_id}", dtype, shape_kind, vcf_type, number
    )
    return info, call


def dtype_matrix_fields():
    """Return the :class:`DtypeMatrixField` manifest in store order.

    Numeric and StringDType fields appear twice — once as INFO and once as
    FORMAT under the same ``vcf_id`` — while the bytes, character and flag
    fields are INFO only.
    """
    fields = []
    for token in _matrix_numeric_tokens():
        dtype = np.dtype(_matrix_dtype(token, reference=False))
        vcf_type = "Integer" if dtype.kind == "i" else "Float"
        code = _MATRIX_TOKEN_CODE[token]
        for suffix, shape_kind, number in _MATRIX_SHAPE_SPECS:
            info, call = _matrix_axis_pair(
                code, suffix, dtype, shape_kind, vcf_type, number
            )
            fields.append(info)
            fields.append(call)

    str_dtype = np.dtype(_MATRIX_STRING_DTYPE)
    for suffix, shape_kind, number in _MATRIX_SHAPE_SPECS:
        info, call = _matrix_axis_pair(
            _MATRIX_STRING_CODE, suffix, str_dtype, shape_kind, "String", number
        )
        fields.append(info)
        fields.append(call)

    fields.append(
        DtypeMatrixField(
            "BYT", "INFO", "variant_BYT", np.dtype("S4"), "scalar", "String", 1
        )
    )
    fields.append(
        DtypeMatrixField(
            "CHR", "INFO", "variant_CHR", np.dtype("<U1"), "scalar", "Character", 1
        )
    )
    fields.append(
        DtypeMatrixField(
            "FLG", "INFO", "variant_FLG", np.dtype(bool), "scalar", "Flag", 0
        )
    )
    return fields


def make_dtype_matrix(
    *, reference=False, variants_chunk_size=3, samples_chunk_size=2, **kwargs
):
    """Build the systematic dtype / shape matrix store in memory.

    See :func:`dtype_matrix_arrays` for the value scheme and the meaning of
    ``reference``. The default chunk sizes split both axes into multiple
    chunks so chunk-boundary handling is exercised; pass ``**kwargs`` through
    to :func:`make_vcz` to override chunking or other options.
    """
    data = dtype_matrix_arrays(reference=reference)
    return make_vcz(
        variant_contig=data.variant_contig,
        variant_position=data.variant_position,
        alleles=data.alleles,
        num_samples=data.num_samples,
        call_genotype=data.call_genotype,
        info_fields=data.info_fields,
        call_fields=data.call_fields,
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
        **kwargs,
    )
