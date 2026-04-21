import functools

import numpy as np

from vcztools import regions as regions_mod
from vcztools import samples as samples_mod
from vcztools import utils
from vcztools import variant_filter as variant_filter_mod
from vcztools.utils import (
    _as_fixed_length_string,
)


class VariantChunkReader:
    """Zarr I/O with block-level per-chunk caching.

    Sole I/O path for a ``variant_chunks`` walk: each array block needed
    for a chunk visit is fetched at most once. Readahead/caching will
    be added here later; callers should not go around it.

    Two public accessors differ only for ``call_*`` fields when a
    ``sample_chunk_plan`` is set:

    - :meth:`get` is the default — applies the plan, so only the sample
      chunks containing selected samples are read, then indexed and
      returned C-contiguous for the C encoder.
    - :meth:`get_with_all_samples` opts out of sample pruning and
      returns the full sample dimension. Needed when a filter
      expression must see every sample to decide variant membership
      (pre-subset semantics).

    For variant-only and static fields the two are equivalent — there
    is no samples axis to prune.

    :meth:`set_chunk` takes a :class:`~vcztools.utils.ChunkRead` and
    its ``selection`` (which may be ``None`` meaning "full chunk") is
    applied to every subsequent variant-axis or ``call_*`` field read.
    """

    def __init__(self, root, *, sample_chunk_plan=None):
        self.root = root
        self.sample_chunk_plan = sample_chunk_plan
        self._static = {}
        self._variant_blocks = {}
        self._call_blocks = {}
        self._chunk_idx = None
        self._current_variant_selection = None

    def set_chunk(self, chunk_read):
        """Advance to the chunk described by ``chunk_read`` and evict
        the previous chunk's per-chunk cache entries. The chunk's
        ``selection`` (``None`` means "full chunk") is applied to every
        subsequent field read."""
        if chunk_read.index == self._chunk_idx:
            return
        self._chunk_idx = chunk_read.index
        self._variant_blocks.clear()
        self._call_blocks.clear()
        self._current_variant_selection = chunk_read.selection

    def get(self, field):
        """Return ``field`` from the current chunk, with
        ``sample_chunk_plan`` applied to the samples axis of
        ``call_*`` fields. For variant-only and static fields the plan
        is irrelevant and the field is returned as-is."""
        return self._get(field, prune=True)

    def get_with_all_samples(self, field):
        """Like :meth:`get` but for ``call_*`` fields returns the full
        sample dimension (no pruning). Use when a filter expression
        must see every sample in order to decide variant membership.
        Equivalent to :meth:`get` for variant-only and static fields."""
        return self._get(field, prune=False)

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
            return self._slice_variants(self._load_variant_block(field, arr))
        plan = self.sample_chunk_plan
        if plan is None or not prune:
            return self._slice_variants(self._load_call_full(field, arr))
        return self._slice_variants(self._load_call_pruned(field, arr, plan))

    def _slice_variants(self, block):
        sel = self._current_variant_selection
        if sel is None:
            return block
        return block[sel]

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
        # Subset per chunk before concatenating: for a handful of samples
        # across many chunks the concat intermediate is O(num_selected)
        # rather than O(sum of sample-chunk sizes).
        parts = []
        for cr in plan.chunk_reads:
            block = self._load_call_block(field, arr, cr.index)
            if cr.selection is not None:
                block = block[:, cr.selection]
            parts.append(block)
        raw = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        if plan.permutation is not None:
            raw = raw[:, plan.permutation]
        # Advanced indexing along a non-first axis of a 3-D array yields a
        # non-C-contiguous view; ascontiguousarray produces the fresh
        # contiguous buffer that the C encoder requires.
        return np.ascontiguousarray(raw)


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


def _validate_samples_input(value) -> None:
    """Reject samples inputs that are not ``None`` or an integer
    sequence.

    String IDs are rejected with a pointer at
    :func:`vcztools.samples.resolve_sample_selection`, which translates
    bcftools-style name arguments into the integer indexes this
    constructor expects.
    """
    if value is None:
        return
    if isinstance(value, np.ndarray):
        if not np.issubdtype(value.dtype, np.integer):
            raise TypeError(
                "samples must be a sequence of integer indexes or None; "
                "use vcztools.samples.resolve_sample_selection to translate "
                "sample names into indexes"
            )
        return
    if isinstance(value, list):
        if len(value) > 0 and not isinstance(value[0], int | np.integer):
            raise TypeError(
                "samples must be a sequence of integer indexes or None; "
                "use vcztools.samples.resolve_sample_selection to translate "
                "sample names into indexes"
            )
        return
    raise TypeError(
        f"samples must be a sequence of integer indexes or None; "
        f"got {type(value).__name__}"
    )


class VczReader:
    """Central reader for VCZ (Zarr-based VCF) files.

    Owns the zarr root and provides metadata properties and
    variant iteration at both chunk and row granularity.

    Parameters
    ----------
    root
        An already-opened :class:`zarr.Group` pointing at the VCZ
        dataset. Use :func:`vcztools.utils.open_zarr` to open a path
        (local, remote, or zip) with the desired backend before
        constructing the reader.
    variants
        Variant selection. Accepts either a ``list[ChunkRead]`` (use
        :func:`vcztools.regions.build_chunk_plan` to build one from
        region/target strings and a root), or a sorted 1-D array of
        global variant indexes (which will be bucketed into a plan
        internally). ``None`` iterates every variant.
    samples
        Integer indexes into the VCZ ``sample_id`` array, in the
        order the caller wants. ``None`` selects every real sample
        (skipping any empty-string masked entries). Out-of-range
        indexes raise ``ValueError``; duplicates are permitted.
        See :func:`vcztools.samples.resolve_sample_selection` for
        the bcftools-style name-to-index translation the CLI uses.
    variant_filter
        An object implementing the
        :class:`~vcztools.variant_filter.VariantFilter` protocol, or
        ``None`` for no filter. The CLI constructs a
        :class:`~vcztools.bcftools_filter.BcftoolsFilter` from ``-i``/
        ``-e`` flags; external callers can plug in any alternative
        implementation.
    filter_on_subset_samples
        Which sample axis a sample-scope filter sees. ``False``
        (default) evaluates on the full sample axis, matching
        ``bcftools view`` semantics. ``True`` evaluates on the subset
        sample axis, matching ``bcftools query``'s FMT-scope
        post-subset semantics and avoiding reads of unselected sample
        chunks. No-op for variant-scope filters.
    """

    def __init__(
        self,
        root,
        *,
        variants: np.ndarray | list[utils.ChunkRead] | None = None,
        samples=None,
        drop_genotypes: bool = False,
        variant_filter: variant_filter_mod.VariantFilter | None = None,
        filter_on_subset_samples: bool = False,
    ):
        _validate_samples_input(samples)

        self.root = root

        all_samples = self.root["sample_id"][:]

        if drop_genotypes:
            self.sample_ids = []
            self.samples_selection = np.array([], dtype=np.int64)
            self.subsetting_samples = False
        elif samples is None:
            self.samples_selection = np.flatnonzero(all_samples != "")
            self.sample_ids = all_samples[self.samples_selection]
            self.subsetting_samples = False
        else:
            self.samples_selection = np.asarray(samples, dtype=np.int64)
            if self.samples_selection.size > 0:
                lo = self.samples_selection.min()
                hi = self.samples_selection.max()
                if lo < 0 or hi >= all_samples.size:
                    raise ValueError(
                        f"sample index out of range: must be in [0, {all_samples.size})"
                    )
            self.sample_ids = all_samples[self.samples_selection]
            self.subsetting_samples = True

        if len(self.samples_selection) > 0:
            # Plan covers the missing-header-samples case too (samples=None
            # with empty-string entries): samples_selection excludes those
            # indices, and the plan reduces accordingly. When the selection
            # covers every sample in order, the plan is a no-op index.
            self.sample_chunk_plan = samples_mod.build_chunk_plan(
                self.samples_selection,
                samples_chunk_size=int(self.root["sample_id"].chunks[0]),
            )
        else:
            self.sample_chunk_plan = None

        if variants is None:
            num_chunks = int(self.root["variant_position"].cdata_shape[0])
            self.variant_chunk_plan = [
                utils.ChunkRead(index=i) for i in range(num_chunks)
            ]
        elif isinstance(variants, list):
            self.variant_chunk_plan = variants
        else:
            self.variant_chunk_plan = regions_mod.chunk_plan_from_indexes(
                np.asarray(variants),
                variants_chunk_size=int(self.root["variant_position"].chunks[0]),
            )

        if (
            variant_filter is not None
            and drop_genotypes
            and variant_filter.scope == "sample"
        ):
            raise ValueError(
                "sample-scope variant_filter is incompatible with drop_genotypes=True"
            )
        self.variant_filter = variant_filter
        self.filter_on_subset_samples = filter_on_subset_samples

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
    ):
        """Yield dict[str, np.ndarray] per variant chunk that passes the
        current variants/samples/variant-filter selection.

        The per-chunk computation runs in stages:

        1. Iterate the ``variant_chunk_plan``. The reader pre-slices
           each chunk's variant axis to the plan's per-chunk selection
           (or returns the full chunk when ``selection`` is ``None``),
           so by the time we see the data the region/target filter has
           already been applied.
        2. Filter evaluation builds a per-variant (or per-variant-per-
           sample) boolean mask on top of the pre-sliced rows.
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

        reader = VariantChunkReader(
            self.root,
            sample_chunk_plan=self.sample_chunk_plan,
        )
        query_fields = self._resolve_query_fields(fields)
        filter_getter = (
            reader.get if self.filter_on_subset_samples else reader.get_with_all_samples
        )

        for chunk_read in self.variant_chunk_plan:
            reader.set_chunk(chunk_read)
            v_mask = None
            if self.variant_filter is not None:
                filter_data = {
                    f: filter_getter(f) for f in self.variant_filter.referenced_fields
                }
                v_mask = self.variant_filter.evaluate(filter_data)
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
            value = reader.get(field)
            if variants_selection is not None and reader.has_variants_axis(field):
                value = value[variants_selection]
            data[field] = value
        return data

    def _reduce_chunk_mask(self, v_mask):
        """Reduce a chunk mask into ``(variants_selection, call_mask)``.

        A ``None`` or 1-D mask is the variant selection; there is no
        call mask. A 2-D mask (from a sample-scoped filter) collapses
        along the sample axis for the variant selection and carries the
        per-sample matches through as ``call_mask``. The sample-axis
        subset is applied here only when the filter was evaluated on
        the *full* sample axis; under ``filter_on_subset_samples`` the
        mask is already in subset-sample space.
        """
        if v_mask is None or v_mask.ndim == 1:
            return v_mask, None
        variants_selection = np.any(v_mask, axis=1)
        call_mask = v_mask[variants_selection]
        if self.subsetting_samples and not self.filter_on_subset_samples:
            call_mask = call_mask[:, self.samples_selection]
        return variants_selection, call_mask

    def variants(
        self,
        *,
        fields: list[str] | None = None,
    ):
        """Yield dict[str, scalar/1d-array] per variant row."""
        for chunk_data in self.variant_chunks(fields=fields):
            first_field = next(iter(chunk_data.values()))
            num_variants = len(first_field)
            for i in range(num_variants):
                yield {name: chunk_data[name][i] for name in chunk_data}
