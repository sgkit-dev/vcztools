import collections.abc

import numpy as np
import zarr

from vcztools import filter as filter_mod
from vcztools.regions import (
    parse_regions,
    parse_targets,
    regions_to_chunk_indexes,
    regions_to_selection,
)
from vcztools.samples import parse_samples
from vcztools.utils import _as_fixed_length_unicode, open_zarr


# NOTE:  this class is just a skeleton for now. The idea is that this
# will provide readahead, caching etc, and will be the central location
# for fetching bulk Zarr data.
class VariantChunkReader(collections.abc.Sequence):
    """
    Retrieve data from a Zarr store and return chunk-by-chunk in the
    variants dimension.
    """

    def __init__(self, root, *, fields=None):
        self.root = root
        if fields is None:
            fields = [
                key
                for key in root.keys()
                if key.startswith("variant_") or key.startswith("call_")
            ]
        self.arrays = {key: self.root[key] for key in fields}
        # TODO validate the arrays have the correct shapes setc
        self.num_chunks = next(iter(self.arrays.values())).cdata_shape[0]

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, chunk):
        return {key: array.blocks[chunk] for key, array in self.arrays.items()}

    def get_chunk_data(self, chunk, mask, samples_selection=None):
        num_samples = len(samples_selection) if samples_selection is not None else 0
        return {
            key: get_vchunk_array(
                array,
                chunk,
                mask,
                samples_selection
                if (key.startswith("call_") and num_samples > 0)
                else None,
            )
            for key, array in self.arrays.items()
        }


def get_vchunk_array(zarray, v_chunk, mask, samples_selection=None):
    v_chunksize = zarray.chunks[0]
    start = v_chunksize * v_chunk
    end = v_chunksize * (v_chunk + 1)
    if samples_selection is None:
        result = zarray[start:end]
    else:
        result = zarray.oindex[start:end, samples_selection]
    if mask is not None:
        result = result[mask]
    return result


def variant_chunk_index_iter(root, regions=None, targets=None):
    """Iterate over variant chunk indexes that overlap the given regions or targets.

    Returns tuples of variant chunk indexes and (optional) variant masks.

    A variant mask of None indicates that all the variants in the chunk are included.
    """

    pos = root["variant_position"]

    if regions is None and targets is None:
        num_chunks = pos.cdata_shape[0]
        # no regions or targets selected
        for v_chunk in range(num_chunks):
            v_mask_chunk = None
            yield v_chunk, v_mask_chunk

    else:
        contigs_u = _as_fixed_length_unicode(root["contig_id"][:]).tolist()
        regions_pyranges = parse_regions(regions, contigs_u)
        targets_pyranges, complement = parse_targets(targets, contigs_u)

        # Use the region index to find the chunks that overlap specfied regions or
        # targets
        region_index = root["region_index"][:]
        chunk_indexes = regions_to_chunk_indexes(
            regions_pyranges,
            targets_pyranges,
            complement,
            region_index,
        )

        # Then use only load required variant_contig/position chunks
        if len(chunk_indexes) == 0:
            # no chunks - no variants to write
            return
        elif len(chunk_indexes) == 1:
            # single chunk
            block_sel = chunk_indexes[0]
        else:
            # TODO: zarr.blocks doesn't support int array indexing
            # use that when it does
            block_sel = slice(chunk_indexes[0], chunk_indexes[-1] + 1)

        region_variant_contig = root["variant_contig"].blocks[block_sel][:]
        region_variant_position = root["variant_position"].blocks[block_sel][:]
        region_variant_length = root["variant_length"].blocks[block_sel][:]

        # Find the final variant selection
        variant_selection = regions_to_selection(
            regions_pyranges,
            targets_pyranges,
            complement,
            region_variant_contig,
            region_variant_position,
            region_variant_length,
        )
        variant_mask = np.zeros(region_variant_position.shape[0], dtype=bool)
        variant_mask[variant_selection] = 1
        # Use zarr arrays to get mask chunks aligned with the main data
        # for convenience.
        z_variant_mask = zarr.array(variant_mask, chunks=pos.chunks[0])

        # TODO: this should be the actual chunk indexes (not a range)
        # when zarr.blocks supports int array indexing (above)
        for i, v_chunk in enumerate(range(chunk_indexes[0], chunk_indexes[-1] + 1)):
            v_mask_chunk = z_variant_mask.blocks[i]
            yield v_chunk, v_mask_chunk


def variant_chunk_index_iter_with_filtering(
    root,
    *,
    regions=None,
    targets=None,
    include: str | None = None,
    exclude: str | None = None,
):
    """Iterate over variant chunk indexes that overlap the given regions or targets
    and which match the include/exclude filter expression.

    Returns tuples of variant chunk indexes and (optional) variant masks.

    A variant mask of None indicates that all the variants in the chunk are included.
    """

    filter_expr = filter_mod.FilterExpression(
        field_names=set(root), include=include, exclude=exclude
    )
    if filter_expr.parse_result is None:
        filter_expr = None
    else:
        filter_fields = list(filter_expr.referenced_fields)
        filter_fields_reader = VariantChunkReader(root, fields=filter_fields)

    for v_chunk, v_mask_chunk in variant_chunk_index_iter(root, regions, targets):
        if filter_expr is not None:
            chunk_data = filter_fields_reader[v_chunk]
            v_mask_chunk_filter = filter_expr.evaluate(chunk_data)
            if v_mask_chunk is None:
                v_mask_chunk = v_mask_chunk_filter
            else:
                if v_mask_chunk_filter.ndim == 2:
                    v_mask_chunk = np.expand_dims(v_mask_chunk, axis=1)
                v_mask_chunk = np.logical_and(v_mask_chunk, v_mask_chunk_filter)
        if v_mask_chunk is None or np.any(v_mask_chunk):
            yield v_chunk, v_mask_chunk


def variant_chunk_iter(
    root,
    *,
    fields: list[str] | None = None,
    regions=None,
    targets=None,
    include: str | None = None,
    exclude: str | None = None,
    samples_selection=None,
):
    if fields is not None and len(fields) == 0:
        return  # empty iterator
    query_fields_reader = VariantChunkReader(root, fields=fields)
    for v_chunk, v_mask_chunk in variant_chunk_index_iter_with_filtering(
        root,
        regions=regions,
        targets=targets,
        include=include,
        exclude=exclude,
    ):
        # The variants_selection is used to subset variant chunks along
        # the variants dimension.
        # The call_mask is returned to the client to indicate which samples
        # matched (for each variant) in the case of per-sample filtering.
        if v_mask_chunk is None or v_mask_chunk.ndim == 1:
            variants_selection = v_mask_chunk
            call_mask = None
        else:
            variants_selection = np.any(v_mask_chunk, axis=1)
            call_mask = v_mask_chunk[variants_selection]
            if samples_selection is not None:
                call_mask = call_mask[:, samples_selection]
        chunk_data = query_fields_reader.get_chunk_data(
            v_chunk, variants_selection, samples_selection=samples_selection
        )
        if call_mask is not None:
            chunk_data["call_mask"] = call_mask
        yield chunk_data


def variant_iter(
    vcz,
    *,
    fields: list[str] | None = None,
    regions: str | None = None,
    targets: str | None = None,
    include: str | None = None,
    exclude: str | None = None,
    samples: list[str] | str | None = None,
    zarr_backend_storage: str | None = None,
):
    """Iterate over variants that overlap the given regions or targets
    and which match the include/exclude filter expression.

    Only values for the samples specified are returned.

    Returns dicts containing the specified fields keyed by VCF Zarr name.

    By default all fields for all variants and samples are returned.
    """
    root = open_zarr(vcz, mode="r", zarr_backend_storage=zarr_backend_storage)
    all_samples = root["sample_id"][:]
    _, samples_selection = parse_samples(samples, all_samples)

    for chunk_data in variant_chunk_iter(
        root,
        fields=fields,
        regions=regions,
        targets=targets,
        include=include,
        exclude=exclude,
        samples_selection=samples_selection,
    ):
        # get first field in chunk_data to find number of variants
        field = next(iter(chunk_data.values()))
        num_variants = len(field)
        for i in range(num_variants):
            yield {name: chunk_data[name][i] for name in chunk_data.keys()}
