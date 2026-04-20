import collections.abc
import functools

import numpy as np
import pandas as pd

from vcztools import filter as filter_mod
from vcztools import regions as regions_mod
from vcztools import samples as samples_mod
from vcztools import utils
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

    def __init__(self, root, *, fields=None, sample_chunk_plan=None):
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
        self.sample_chunk_plan = sample_chunk_plan

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, chunk):
        data = {key: array.blocks[chunk] for key, array in self.arrays.items()}
        data.update(self.static_data)
        return data

    def get_chunk_data(self, chunk, mask):
        plan = self.sample_chunk_plan
        data = {}
        for key, array in self.arrays.items():
            if key.startswith("call_") and plan is not None:
                parts = [
                    array.blocks[(chunk, sci) + (slice(None),) * (array.ndim - 2)]
                    for sci in plan.chunk_indexes
                ]
                raw = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
                # Advanced indexing along a non-first axis of a 3-D array
                # yields a non-C-contiguous view; ascontiguousarray produces
                # the fresh contiguous buffer that the C encoder requires.
                result = np.ascontiguousarray(raw[:, plan.local_selection])
            else:
                result = array.blocks[chunk]
            if mask is not None:
                result = result[mask]
            data[key] = result
        data.update(self.static_data)
        return data


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
        if "," in value:
            raise ValueError(
                f"{arg_name} string '{value}' contains ','. "
                f"Pass a list[str] for multiple regions."
            )
        return regions_mod.region_strings_to_dataframe([value])
    if isinstance(value, list):
        return regions_mod.region_strings_to_dataframe(value)
    raise TypeError(
        f"{arg_name} must be str, list[str], pandas.DataFrame, or None; "
        f"got {type(value).__name__}"
    )


def _validate_samples_input(value) -> None:
    """Reject samples inputs that are not ``None`` or a ``list[str]``.

    A leading ``^`` on the first list element is rejected with a
    ``ValueError`` pointing at ``samples_complement=True``, mirroring the
    regions/targets rejection.
    """
    if value is None:
        return
    if isinstance(value, list):
        if len(value) > 0 and isinstance(value[0], str) and value[0].startswith("^"):
            raise ValueError(
                "samples does not accept a '^' prefix in the Python API; "
                "use samples_complement=True for complement"
            )
        return
    raise TypeError(f"samples must be list[str] or None; got {type(value).__name__}")


class VczReader:
    """Central reader for VCZ (Zarr-based VCF) files.

    Owns the zarr root and provides metadata properties and
    variant iteration at both chunk and row granularity.

    Parameters
    ----------
    regions, targets
        Regions/targets to restrict iteration to. May be a single region
        string, a list of region strings, or a pandas DataFrame with
        columns ``contig`` (str), ``start`` and ``end`` (nullable ``Int64``,
        ``pd.NA`` for unbounded). ``None`` disables the filter.
        ``regions`` uses overlap semantics; ``targets`` uses exact-position
        semantics and additionally accepts ``targets_complement``.
    targets_complement
        If ``True``, the targets selection is inverted (matches everything
        *outside* the listed intervals).
    samples
        Sample IDs to select, as a ``list[str]``. ``None`` selects every
        sample.
    samples_complement
        If ``True``, selects every sample *except* those in ``samples``.
    """

    def __init__(
        self,
        vcz,
        *,
        regions=None,
        targets=None,
        targets_complement: bool = False,
        samples=None,
        samples_complement: bool = False,
        ignore_missing_samples: bool = False,
        drop_genotypes: bool = False,
        zarr_backend_storage: str | None = None,
    ):
        regions_df = _regions_input_to_df(regions, arg_name="regions")
        targets_df = _regions_input_to_df(targets, arg_name="targets")
        _validate_samples_input(samples)

        self.root = open_zarr(vcz, mode="r", zarr_backend_storage=zarr_backend_storage)

        all_samples = self.root["sample_id"][:]

        if drop_genotypes:
            self.sample_ids = []
            self.samples_selection = np.array([], dtype=np.int64)
            self.subsetting_samples = False
        else:
            self.sample_ids, self.samples_selection = samples_mod.parse_samples(
                samples,
                all_samples=all_samples,
                ignore_missing_samples=ignore_missing_samples,
                complement=samples_complement,
            )
            self.subsetting_samples = samples is not None

        if len(self.samples_selection) > 0:
            # Plan covers the missing-header-samples case too (samples=None
            # with empty-string entries): samples_selection excludes those
            # indices, and the plan reduces accordingly. When the selection
            # covers every sample in order, the plan is a no-op index.
            self.sample_chunk_plan = samples_mod.build_chunk_plan(
                self.samples_selection,
                num_samples=len(all_samples),
                samples_chunk_size=int(self.root["sample_id"].chunks[0]),
            )
        else:
            self.sample_chunk_plan = None

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
        """Yield dict[str, np.ndarray] per variant chunk that passes the
        current regions/targets/samples/include-exclude selection.

        The per-chunk computation runs in stages:

        1. ``_chunk_indexes`` — which chunks to visit (region-index pruning).
        2. ``_chunk_mask`` — build a per-variant (or per-variant-per-sample)
           boolean mask combining regions/targets and filter expression.
        3. ``_reduce_chunk_mask`` — split that mask into a
           ``variants_selection`` and an optional ``call_mask``.
        4. ``VariantChunkReader.get_chunk_data`` — load the query fields,
           applying the variant and sample selections.
        """
        if fields is not None and len(fields) == 0:
            return

        filter_expr = self._make_filter_expression(include, exclude)
        query_reader = VariantChunkReader(
            self.root, fields=fields, sample_chunk_plan=self.sample_chunk_plan
        )
        filter_reader = (
            VariantChunkReader(self.root, fields=list(filter_expr.referenced_fields))
            if filter_expr is not None
            else None
        )

        for chunk_idx in self._chunk_indexes():
            v_mask = self._chunk_mask(chunk_idx, filter_expr, filter_reader)
            if v_mask is not None and not np.any(v_mask):
                continue
            variants_selection, call_mask = self._reduce_chunk_mask(v_mask)
            chunk_data = query_reader.get_chunk_data(chunk_idx, variants_selection)
            if call_mask is not None:
                chunk_data["call_mask"] = call_mask
            yield chunk_data

    def _make_filter_expression(self, include, exclude):
        """Return a :class:`FilterExpression` for ``include``/``exclude`` or
        ``None`` when neither is supplied (or the expression is empty)."""
        expr = filter_mod.FilterExpression(
            field_names=set(self.root), include=include, exclude=exclude
        )
        return expr if expr.parse_result is not None else None

    def _chunk_indexes(self):
        """Variant-chunk indexes that may contain matching rows.

        With ``self.regions`` set we prune via the region index; otherwise
        we visit every chunk.
        """
        if self.regions is None:
            num_chunks = self.root["variant_position"].cdata_shape[0]
            return range(num_chunks)
        region_index = self.root["region_index"][:]
        return regions_mod.regions_to_chunk_indexes(self.regions, region_index)

    def _chunk_region_mask(self, chunk_idx):
        """1-D boolean per-variant mask from regions/targets for one chunk,
        or ``None`` when neither ``self.regions`` nor ``self.targets`` is set.
        """
        if self.regions is None and self.targets is None:
            return None
        contig = self.root["variant_contig"].blocks[chunk_idx][:]
        position = self.root["variant_position"].blocks[chunk_idx][:]
        length = self.root["variant_length"].blocks[chunk_idx][:]
        selection = regions_mod.regions_to_selection(
            self.regions, self.targets, contig, position, length
        )
        mask = np.zeros(position.shape[0], dtype=bool)
        mask[selection] = True
        return mask

    def _chunk_mask(self, chunk_idx, filter_expr, filter_reader):
        """Combined region + filter mask for a chunk.

        Returns ``None`` when no subsetting applies, a 1-D mask for variant-
        scoped filters, or a 2-D mask for sample-scoped filters.
        """
        v_mask = self._chunk_region_mask(chunk_idx)
        if filter_expr is None:
            return v_mask
        filter_mask = filter_expr.evaluate(filter_reader[chunk_idx])
        if v_mask is None:
            return filter_mask
        if filter_mask.ndim == 2:
            v_mask = np.expand_dims(v_mask, axis=1)
        return np.logical_and(v_mask, filter_mask)

    def _reduce_chunk_mask(self, v_mask):
        """Reduce a chunk mask into ``(variants_selection, call_mask)``.

        A ``None`` or 1-D mask is the variant selection; there is no
        call mask. A 2-D mask (from a sample-scoped filter) collapses
        along the sample axis for the variant selection and carries the
        per-sample matches through as ``call_mask``, with the sample-axis
        subset applied when a samples subset was requested.
        """
        if v_mask is None or v_mask.ndim == 1:
            return v_mask, None
        variants_selection = np.any(v_mask, axis=1)
        call_mask = v_mask[variants_selection]
        if self.subsetting_samples:
            call_mask = call_mask[:, self.samples_selection]
        return variants_selection, call_mask

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
