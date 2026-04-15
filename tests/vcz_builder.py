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


def _compute_region_index(variant_contig, variant_position, variant_length, v_chunk):
    num_variants = variant_contig.shape[0]
    num_chunks = (num_variants + v_chunk - 1) // v_chunk
    rows = []
    for chunk_idx in range(num_chunks):
        start = chunk_idx * v_chunk
        stop = min(start + v_chunk, num_variants)
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
    contigs=("chr1",),
    filters=("PASS",),
    variants_chunk_size=None,
    samples_chunk_size=None,
    call_genotype=None,
    info_fields=None,
    variant_id=None,
    variant_quality=None,
    variant_filter=None,
    variant_length=None,
    ploidy=2,
):
    """
    Build an in-memory VCZ group and return it.

    All parameters are keyword-only. ``variant_contig``, ``variant_position``
    and ``alleles`` are required; everything else has a sensible default.
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

    region_index = _compute_region_index(
        variant_contig, variant_position, variant_length, v_chunk
    )

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

        _create_array(
            root,
            "sample_id",
            np.array([f"sample_{i}" for i in range(num_samples)], dtype="<U16"),
            chunks=(max(num_samples, 1),),
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
        _create_array(
            root,
            "variant_contig",
            variant_contig,
            chunks=(v_chunk,),
            dimension_names=("variants",),
        )
        _create_array(
            root,
            "variant_position",
            variant_position,
            chunks=(v_chunk,),
            dimension_names=("variants",),
        )
        _create_array(
            root,
            "variant_length",
            variant_length,
            chunks=(v_chunk,),
            dimension_names=("variants",),
        )
        _create_array(
            root,
            "variant_allele",
            allele_array,
            chunks=(v_chunk, max_alleles),
            dimension_names=("variants", "alleles"),
        )
        _create_array(
            root,
            "variant_filter",
            variant_filter,
            chunks=(v_chunk, len(filters)),
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
            _create_array(
                root,
                "variant_id",
                np.asarray(variant_id, dtype="<U16"),
                chunks=(v_chunk,),
                dimension_names=("variants",),
            )
        if variant_quality is not None:
            _create_array(
                root,
                "variant_quality",
                np.asarray(variant_quality, dtype=np.float32),
                chunks=(v_chunk,),
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

        if info_fields is not None:
            for name, data in info_fields.items():
                data = np.asarray(data)
                dims = ("variants",) if data.ndim == 1 else ("variants", f"{name}_dim1")
                chunks = (v_chunk,) if data.ndim == 1 else (v_chunk, data.shape[1])
                _create_array(
                    root,
                    f"variant_{name}",
                    data,
                    chunks=chunks,
                    dimension_names=dims,
                )

    return root
