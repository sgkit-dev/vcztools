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


class VariantChunkReader:
    """Zarr I/O with block-level per-chunk caching.

    Sole I/O path for a ``variant_chunks`` walk: each array block needed
    for a chunk visit is fetched at most once. Readahead/caching will
    be added here later; callers should not go around it.

    Two access modes for ``call_*`` fields:

    - :meth:`get_for_filter` returns the full sample dimension (filter
      evaluation must see every sample, even those not in the user's
      selection).
    - :meth:`get_for_output` applies ``sample_chunk_plan`` (reads only
      the sample chunks that contain selected samples and indexes into
      their concatenation).

    For variant-only and static fields the two modes are equivalent —
    there is no samples axis to prune.
    """

    def __init__(self, root, *, sample_chunk_plan=None):
        self.root = root
        self.sample_chunk_plan = sample_chunk_plan
        self._static = {}
        self._variant_blocks = {}
        self._call_blocks = {}
        self._chunk_idx = None

    def set_chunk(self, chunk_idx):
        """Advance to a new variant chunk and evict the previous chunk's
        per-chunk cache entries."""
        if chunk_idx == self._chunk_idx:
            return
        self._chunk_idx = chunk_idx
        self._variant_blocks.clear()
        self._call_blocks.clear()

    def get_for_filter(self, field):
        """Return the field at the full sample dimension."""
        return self._get(field, prune=False)

    def get_for_output(self, field):
        """Return the field with ``sample_chunk_plan`` applied to the
        samples axis of ``call_*`` fields."""
        return self._get(field, prune=True)

    def has_variants_axis(self, field):
        """Whether ``field``'s first dimension is ``variants``. Static
        fields (``filter_id`` etc.) return ``False`` and are not sliced
        by the variants selection."""
        arr = self.root[field]
        dims = utils.array_dims(arr)
        return dims is not None and len(dims) > 0 and dims[0] == "variants"

    def _get(self, field, *, prune):
        if field in self._static:
            return self._static[field]
        arr = self.root[field]
        if not self.has_variants_axis(field):
            self._static[field] = arr[:]
            return self._static[field]
        if not field.startswith("call_"):
            return self._load_variant_block(field, arr)
        plan = self.sample_chunk_plan
        if plan is None or not prune:
            return self._load_call_full(field, arr)
        return self._load_call_pruned(field, arr, plan)

    def _load_variant_block(self, field, arr):
        key = (self._chunk_idx, field)
        block = self._variant_blocks.get(key)
        if block is None:
            block = arr.blocks[self._chunk_idx]
            self._variant_blocks[key] = block
        return block

    def _load_call_block(self, field, arr, sci):
        key = (self._chunk_idx, field, sci)
        block = self._call_blocks.get(key)
        if block is None:
            index = (self._chunk_idx, sci) + (slice(None),) * (arr.ndim - 2)
            block = arr.blocks[index]
            self._call_blocks[key] = block
        return block

    def _load_call_full(self, field, arr):
        num_s_chunks = int(arr.cdata_shape[1])
        parts = [self._load_call_block(field, arr, sci) for sci in range(num_s_chunks)]
        return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

    def _load_call_pruned(self, field, arr, plan):
        # Invariant: the concatenation must span exactly plan.chunk_indexes
        # (in that order) because plan.local_selection indexes into that
        # specific layout — not into the superset the cache might hold.
        parts = [self._load_call_block(field, arr, sci) for sci in plan.chunk_indexes]
        raw = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        # Advanced indexing along a non-first axis of a 3-D array yields a
        # non-C-contiguous view; ascontiguousarray produces the fresh
        # contiguous buffer that the C encoder requires.
        return np.ascontiguousarray(raw[:, plan.local_selection])


def _combine_masks(v_mask, filter_mask):
    """Combine a region mask with a filter mask. Broadcasts a 1-D region
    mask against a 2-D filter mask when the filter is sample-scoped."""
    if v_mask is None:
        return filter_mask
    if filter_mask.ndim == 2:
        v_mask = np.expand_dims(v_mask, axis=1)
    return np.logical_and(v_mask, filter_mask)


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
        2. ``_chunk_region_mask`` / filter evaluation — build a per-variant
           (or per-variant-per-sample) boolean mask.
        3. ``_reduce_chunk_mask`` — split that mask into a
           ``variants_selection`` and an optional ``call_mask``.
        4. ``_collect_output`` — load the query fields, applying the
           variant and sample selections.

        A single :class:`VariantChunkReader` is the only Zarr I/O path;
        each block needed for a chunk visit is fetched at most once,
        even if a field is referenced by both the filter and the query.
        """
        if fields is not None and len(fields) == 0:
            return

        filter_expr = self._make_filter_expression(include, exclude)
        reader = VariantChunkReader(self.root, sample_chunk_plan=self.sample_chunk_plan)
        query_fields = self._resolve_query_fields(fields)

        for chunk_idx in self._chunk_indexes():
            reader.set_chunk(chunk_idx)
            v_mask = self._chunk_region_mask(reader)
            if filter_expr is not None:
                filter_data = {
                    f: reader.get_for_filter(f) for f in filter_expr.referenced_fields
                }
                filter_mask = filter_expr.evaluate(filter_data)
                v_mask = _combine_masks(v_mask, filter_mask)
            if v_mask is not None and not np.any(v_mask):
                continue
            variants_selection, call_mask = self._reduce_chunk_mask(v_mask)
            chunk_data = self._collect_output(reader, query_fields, variants_selection)
            if call_mask is not None:
                chunk_data["call_mask"] = call_mask
            yield chunk_data

    def _resolve_query_fields(self, fields):
        if fields is not None:
            return list(fields)
        return [
            key
            for key in self.root.keys()
            if key.startswith("variant_") or key.startswith("call_")
        ]

    def _collect_output(self, reader, fields, variants_selection):
        data = {}
        for field in fields:
            value = reader.get_for_output(field)
            if variants_selection is not None and reader.has_variants_axis(field):
                value = value[variants_selection]
            data[field] = value
        return data

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

    def _chunk_region_mask(self, reader):
        """1-D boolean per-variant mask from regions/targets for the
        reader's current chunk, or ``None`` when neither ``self.regions``
        nor ``self.targets`` is set."""
        if self.regions is None and self.targets is None:
            return None
        contig = reader.get_for_output("variant_contig")
        position = reader.get_for_output("variant_position")
        length = reader.get_for_output("variant_length")
        selection = regions_mod.regions_to_selection(
            self.regions, self.targets, contig, position, length
        )
        mask = np.zeros(position.shape[0], dtype=bool)
        mask[selection] = True
        return mask

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
