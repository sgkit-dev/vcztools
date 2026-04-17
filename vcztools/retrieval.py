import collections.abc
import functools

import numpy as np
import pandas as pd

from vcztools import filter as filter_mod
from vcztools import regions as regions_mod
from vcztools import utils
from vcztools.samples import parse_samples
from vcztools.utils import (
    _as_fixed_length_string,
    _as_fixed_length_unicode,
    open_zarr,
)


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
        all_arrays = {key: self.root[key] for key in fields}
        # Partition into variant-chunked arrays and static (non-variant-axis)
        # arrays. Static arrays like filter_id are read once in full and
        # returned unchanged for every chunk.
        self.arrays = {}
        self.static_data = {}
        for key, arr in all_arrays.items():
            dim_names = utils.array_dims(arr)
            if dim_names is not None and dim_names[0] == "variants":
                self.arrays[key] = arr
            else:
                self.static_data[key] = arr[:]
        self.num_chunks = next(iter(self.arrays.values())).cdata_shape[0]

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, chunk):
        data = {key: array.blocks[chunk] for key, array in self.arrays.items()}
        data.update(self.static_data)
        return data

    def get_chunk_data(self, chunk, mask, samples_selection=None):
        num_samples = len(samples_selection) if samples_selection is not None else 0
        data = {
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
        data.update(self.static_data)
        return data


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

    ``regions`` and ``targets`` are ``GenomicRanges`` instances (already parsed
    by :class:`VczReader`) or ``None``.

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
        if regions is None:
            num_chunks = pos.cdata_shape[0]
            chunk_indexes = range(num_chunks)
        else:
            # Use the region index to find the chunks that overlap specified regions
            region_index = root["region_index"][:]
            chunk_indexes = regions_mod.regions_to_chunk_indexes(regions, region_index)

        if len(chunk_indexes) == 0:
            # no chunks - no variants to write
            return

        # Then only load required variant_contig/position chunks
        for chunk_index in chunk_indexes:
            region_variant_contig = root["variant_contig"].blocks[chunk_index][:]
            region_variant_position = root["variant_position"].blocks[chunk_index][:]
            region_variant_length = root["variant_length"].blocks[chunk_index][:]

            # Find the variant selection for the chunk
            variant_selection = regions_mod.regions_to_selection(
                regions,
                targets,
                region_variant_contig,
                region_variant_position,
                region_variant_length,
            )
            variant_mask = np.zeros(region_variant_position.shape[0], dtype=bool)
            variant_mask[variant_selection] = 1

            yield chunk_index, variant_mask


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


def get_filter_ids(root):
    """
    Returns the filter IDs from the specified Zarr store. If the array
    does not exist, return a single filter "PASS" by default.
    """
    if "filter_id" in root:
        filters = _as_fixed_length_string(root["filter_id"][:])
    else:
        filters = np.array(["PASS"], dtype="S")
    return filters


_REGION_DF_COLUMNS = ("contig", "start", "end")


def _regions_input_to_df(value, *, arg_name: str) -> pd.DataFrame | None:
    """Normalise a regions/targets input into the DataFrame schema used by
    :mod:`vcztools.regions`.

    Accepts ``None``, a single region string, a list of region strings, or a
    DataFrame with ``contig``, ``start`` and ``end`` columns. A leading ``^``
    on a string input is rejected with a ``ValueError`` pointing users at the
    ``targets_complement`` flag.
    """
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        missing = set(_REGION_DF_COLUMNS) - set(value.columns)
        if missing:
            raise ValueError(
                f"{arg_name} DataFrame is missing required columns: {sorted(missing)}"
            )
        return value
    if isinstance(value, str):
        if value.startswith("^"):
            raise ValueError(
                f"{arg_name} does not accept a '^' prefix in the Python API; "
                f"use targets_complement=True for complement"
            )
        return regions_mod.region_strings_to_dataframe(value)
    if isinstance(value, list):
        return regions_mod.region_strings_to_dataframe(value)
    raise TypeError(
        f"{arg_name} must be str, list[str], pandas.DataFrame, or None; "
        f"got {type(value).__name__}"
    )


class VczReader:
    """Central reader for VCZ (Zarr-based VCF) files.

    Owns the zarr root and provides metadata properties and
    variant iteration at both chunk and row granularity.

    Parameters
    ----------
    regions, targets
        Regions/targets to restrict iteration to. May be a single
        comma-separated region string, a list of region strings, or a
        pandas DataFrame with columns ``contig`` (str), ``start`` and
        ``end`` (nullable ``Int64``, ``pd.NA`` for unbounded). ``None``
        disables the filter. ``regions`` uses overlap semantics; ``targets``
        uses exact-position semantics and additionally accepts
        ``targets_complement``.
    targets_complement
        If ``True``, the targets selection is inverted (matches everything
        *outside* the listed intervals).
    """

    def __init__(
        self,
        vcz,
        *,
        regions=None,
        targets=None,
        targets_complement: bool = False,
        samples=None,
        force_samples: bool = False,
        drop_genotypes: bool = False,
        zarr_backend_storage: str | None = None,
    ):
        regions_df = _regions_input_to_df(regions, arg_name="regions")
        targets_df = _regions_input_to_df(targets, arg_name="targets")

        self.root = open_zarr(vcz, mode="r", zarr_backend_storage=zarr_backend_storage)

        all_samples = self.root["sample_id"][:]

        if drop_genotypes:
            self.sample_ids = []
            self.samples_selection = np.array([])
        else:
            self.sample_ids, self.samples_selection = parse_samples(
                samples,
                all_samples=all_samples,
                force_samples=force_samples,
            )

        contigs_u = _as_fixed_length_unicode(self.root["contig_id"][:]).tolist()
        self.regions = regions_mod.dataframe_to_ranges(regions_df, contigs_u)
        self.targets = regions_mod.dataframe_to_ranges(
            targets_df, contigs_u, complement=targets_complement
        )

    @functools.cached_property
    def contig_ids(self):
        """Contig IDs as raw zarr strings."""
        return self.root["contig_id"][:]

    @functools.cached_property
    def filter_ids(self):
        """Filter IDs as raw zarr strings."""
        return self.root["filter_id"][:]

    @functools.cached_property
    def contigs(self):
        """Contig IDs as fixed-length bytes (for VcfEncoder)."""
        return _as_fixed_length_string(self.root["contig_id"][:])

    @functools.cached_property
    def filters(self):
        """Filter IDs as fixed-length bytes (for VcfEncoder)."""
        return get_filter_ids(self.root)

    def variant_chunks(
        self,
        *,
        fields: list[str] | None = None,
        include: str | None = None,
        exclude: str | None = None,
    ):
        """Yield dict[str, np.ndarray] per chunk."""
        yield from variant_chunk_iter(
            self.root,
            fields=fields,
            regions=self.regions,
            targets=self.targets,
            include=include,
            exclude=exclude,
            samples_selection=self.samples_selection,
        )

    def variants(
        self,
        *,
        fields: list[str] | None = None,
        include: str | None = None,
        exclude: str | None = None,
    ):
        """Yield dict[str, scalar/1d-array] per variant row."""
        for chunk_data in self.variant_chunks(
            fields=fields, include=include, exclude=exclude
        ):
            first_field = next(iter(chunk_data.values()))
            num_variants = len(first_field)
            for i in range(num_variants):
                yield {name: chunk_data[name][i] for name in chunk_data}
