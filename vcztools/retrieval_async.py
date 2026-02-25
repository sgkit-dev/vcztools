import asyncio

import numpy as np

from vcztools import filter as filter_mod
from vcztools.regions import (
    parse_regions,
    parse_targets,
    regions_to_chunk_indexes,
    regions_to_selection,
)
from vcztools.utils import (
    _as_fixed_length_unicode,
    async_get_zarr_array,
    get_block_selection,
)


# NOTE:  this class is just a skeleton for now. The idea is that this
# will provide readahead, caching etc, and will be the central location
# for fetching bulk Zarr data.
class AsyncVariantChunkReader:
    def __init__(self, root, arrays, num_chunks):
        self.root = root
        self.arrays = arrays
        self.num_chunks = num_chunks

    @classmethod
    async def create(cls, root, *, fields=None):
        if fields is None:
            fields = [
                key
                async for key in root.keys()
                if key.startswith("variant_") or key.startswith("call_")
            ]

        arrays = [root.getitem(key) for key in fields]
        arrays = dict(zip(fields, await asyncio.gather(*arrays)))

        num_chunks = next(iter(arrays.values())).cdata_shape[0]

        return cls(root, arrays, num_chunks)

    async def getitem(self, chunk):
        chunk_data = [
            get_block_selection(array, chunk) for array in self.arrays.values()
        ]
        chunk_data = await asyncio.gather(*chunk_data)
        return dict(zip(self.arrays.keys(), chunk_data))

    async def get_chunk_data(self, chunk, mask=None, samples_selection=None):
        def get_vchunk_array(zarray, v_chunk, samples_selection=None):
            v_chunksize = zarray.chunks[0]
            start = v_chunksize * v_chunk
            end = v_chunksize * (v_chunk + 1)
            if samples_selection is None:
                result = zarray.getitem(slice(start, end))
            else:
                result = zarray.get_orthogonal_selection(
                    (slice(start, end), samples_selection)
                )
            return result

        num_samples = len(samples_selection) if samples_selection is not None else 0
        chunk_data = [
            get_vchunk_array(
                array,
                chunk,
                samples_selection=samples_selection
                if (key.startswith("call_") and num_samples > 0)
                else None,
            )
            for key, array in self.arrays.items()
        ]
        chunk_data = await asyncio.gather(*chunk_data)
        if mask is not None:
            chunk_data = [arr[mask] for arr in chunk_data]
        return dict(zip(self.arrays.keys(), chunk_data))


async def async_variant_chunk_index_iter(root, regions=None, targets=None):
    pos = await root.getitem("variant_position")

    if regions is None and targets is None:
        num_chunks = pos.cdata_shape[0]
        # no regions or targets selected
        for v_chunk in range(num_chunks):
            v_mask_chunk = None
            yield v_chunk, v_mask_chunk
    else:
        contigs = await async_get_zarr_array(root, "contig_id")
        contigs_u = _as_fixed_length_unicode(contigs).tolist()
        regions_pyranges = parse_regions(regions, contigs_u)
        targets_pyranges, complement = parse_targets(targets, contigs_u)

        # Use the region index to find the chunks that overlap specfied regions or
        # targets
        region_index = await async_get_zarr_array(root, "region_index")
        chunk_indexes = regions_to_chunk_indexes(
            regions_pyranges,
            targets_pyranges,
            complement,
            region_index,
        )

        if len(chunk_indexes) == 0:
            # no chunks - no variants to write
            return

        # Then only load required variant_contig/position chunks
        region_variant_contig_arr = await root.getitem("variant_contig")
        region_variant_position_arr = await root.getitem("variant_position")
        region_variant_length_arr = await root.getitem("variant_length")
        for chunk_index in chunk_indexes:
            # TODO: get all three concurrently
            region_variant_contig = await get_block_selection(
                region_variant_contig_arr, chunk_index
            )
            region_variant_position = await get_block_selection(
                region_variant_position_arr, chunk_index
            )
            region_variant_length = await get_block_selection(
                region_variant_length_arr, chunk_index
            )

            # Find the variant selection for the chunk
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

            yield chunk_index, variant_mask


async def async_variant_chunk_index_iter_with_filtering(
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

    field_names = set([key async for key in root.keys()])
    filter_expr = filter_mod.FilterExpression(
        field_names=field_names, include=include, exclude=exclude
    )
    if filter_expr.parse_result is None:
        filter_expr = None
    else:
        filter_fields = list(filter_expr.referenced_fields)
        filter_fields_reader = await AsyncVariantChunkReader.create(
            root, fields=filter_fields
        )

    async for v_chunk, v_mask_chunk in async_variant_chunk_index_iter(
        root, regions, targets
    ):
        if filter_expr is not None:
            chunk_data = await filter_fields_reader.getitem(v_chunk)
            v_mask_chunk_filter = filter_expr.evaluate(chunk_data)
            if v_mask_chunk is None:
                v_mask_chunk = v_mask_chunk_filter
            else:
                if v_mask_chunk_filter.ndim == 2:
                    v_mask_chunk = np.expand_dims(v_mask_chunk, axis=1)
                v_mask_chunk = np.logical_and(v_mask_chunk, v_mask_chunk_filter)
        if v_mask_chunk is None or np.any(v_mask_chunk):
            yield v_chunk, v_mask_chunk


async def async_variant_chunk_iter(
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
    query_fields_reader = await AsyncVariantChunkReader.create(root, fields=fields)

    async for v_chunk, v_mask_chunk in async_variant_chunk_index_iter_with_filtering(
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
        chunk_data = await query_fields_reader.get_chunk_data(
            v_chunk, variants_selection, samples_selection=samples_selection
        )
        if call_mask is not None:
            chunk_data["call_mask"] = call_mask
        yield chunk_data
