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

import warnings

import numpy as np
import zarr
import zarr.codecs
import zarr.storage

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


def _compute_region_index(variant_contig, variant_position, variant_length, min_chunk):
    """Compute the region index, with one row per logical (``min_chunk``-
    sized) chunk. ``min_chunk`` is the minimum variants-axis chunk
    size (driven by ``call_*`` fields); variant-only fields may have
    larger chunks but the region index always indexes at logical-
    chunk granularity.
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
    return np.array(rows, dtype=np.int32)


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
    """
    variant_contig = np.asarray(variant_contig, dtype=np.int32)
    variant_position = np.asarray(variant_position, dtype=np.int32)
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
            variant_contig, variant_position, variant_length, v_chunk
        )
    else:
        region_index = np.asarray(region_index, dtype=np.int32)

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
            _create_array(
                root,
                "variant_quality",
                np.asarray(variant_quality, dtype=np.float32),
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
