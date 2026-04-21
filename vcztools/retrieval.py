import dataclasses
import functools

import numpy as np

from vcztools import regions as regions_mod
from vcztools import samples as samples_mod
from vcztools import utils
from vcztools import variant_filter as variant_filter_mod
from vcztools.utils import (
    _as_fixed_length_string,
)


@dataclasses.dataclass(frozen=True)
class FieldInfo:
    """Schema snapshot for a single store field.

    Materialized once per field via :meth:`VczReader.get_field_info`
    and cached on the reader. External callers (VCF header
    generation, etc.) should never reach into the Zarr store for
    metadata themselves — go through this dataclass instead.
    """

    name: str
    dtype: np.dtype
    shape: tuple[int, ...]
    dims: tuple[str, ...]
    attrs: dict


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


def _get_filter_ids(root):
    """Return the filter IDs as fixed-length bytes, defaulting to a
    single ``b"PASS"`` entry when the store has no ``filter_id``
    field."""
    if "filter_id" in root:
        return _as_fixed_length_string(root["filter_id"][:])
    return np.array(["PASS"], dtype="S")


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

    The reader starts with no configured selection or filter — plan
    state is lazy and is filled in to sensible defaults (every real
    sample; every variant chunk) on first iteration. Callers
    customize via the one-shot setters **before** iterating:

    - :meth:`set_samples` — sample selection (integer indexes).
    - :meth:`set_variants` — variant selection (``list[ChunkRead]``
      or a sorted 1-D index array).
    - :meth:`set_variant_filter` — per-variant filter predicate.

    Each setter can be called at most once; a second call raises
    ``RuntimeError``. The state field itself is the "configured"
    signal: e.g. ``samples_selection is None`` means "not set", so
    ``set_samples`` refuses to run when ``samples_selection`` is
    already populated (either by a prior ``set_samples`` or by
    default resolution during iteration).

    Parameters
    ----------
    root
        An already-opened :class:`zarr.Group` pointing at the VCZ
        dataset. Use :func:`vcztools.utils.open_zarr` to open a path
        (local, remote, or zip) with the desired backend before
        constructing the reader.
    """

    def __init__(self, root):
        self.root = root
        self._sample_chunk_plan = None
        self._variant_chunk_plan = None
        self._samples_selection = None
        self._sample_ids = None
        self.variant_filter = None
        self.filter_on_subset_samples = False
        # True iff set_samples was called. Distinct from
        # "samples_selection is not None" once defaults are resolved:
        # the default resolve populates _samples_selection but
        # leaves this False, so _reduce_chunk_mask can tell whether
        # to subset call_mask to the user's requested samples.
        self._subsetting_samples = False

    def _resolve_samples_if_needed(self) -> None:
        if self._samples_selection is not None:
            return
        all_samples = self.all_sample_ids
        self._samples_selection = np.flatnonzero(all_samples != "")
        self._sample_ids = all_samples[self._samples_selection]
        self._sample_chunk_plan = samples_mod.build_chunk_plan(
            self._samples_selection,
            samples_chunk_size=int(self.root["sample_id"].chunks[0]),
        )

    @property
    def samples_selection(self):
        self._resolve_samples_if_needed()
        return self._samples_selection

    @property
    def sample_ids(self):
        self._resolve_samples_if_needed()
        return self._sample_ids

    @property
    def sample_chunk_plan(self):
        self._resolve_samples_if_needed()
        return self._sample_chunk_plan

    @property
    def variant_chunk_plan(self):
        if self._variant_chunk_plan is None:
            num_chunks = int(self.root["variant_position"].cdata_shape[0])
            self._variant_chunk_plan = [
                utils.ChunkRead(index=i) for i in range(num_chunks)
            ]
        return self._variant_chunk_plan

    def set_samples(self, samples) -> None:
        """Configure the sample selection.

        Accepts a list or ndarray of integer indexes into the VCZ
        ``sample_id`` array, in the order the caller wants. An empty
        sequence is valid and means "no samples in output" (used by
        e.g. ``bcftools view --drop-genotypes``).

        Out-of-range indexes raise ``ValueError``; duplicates are
        permitted. Raises ``RuntimeError`` if already configured
        (including if the default was already resolved by reading
        ``samples_selection`` / ``sample_ids`` / ``sample_chunk_plan``).
        See :func:`vcztools.samples.resolve_sample_selection` for
        the bcftools-style name-to-index translation the CLI uses.
        """
        if self._samples_selection is not None:
            raise RuntimeError("samples already configured")
        _validate_samples_input(samples)

        all_samples = self.all_sample_ids
        samples_selection = np.asarray(samples, dtype=np.int64)
        if samples_selection.size > 0:
            lo = samples_selection.min()
            hi = samples_selection.max()
            if lo < 0 or hi >= all_samples.size:
                raise ValueError(
                    f"sample index out of range: must be in [0, {all_samples.size})"
                )
        self._samples_selection = samples_selection
        self._sample_ids = all_samples[samples_selection]
        if len(samples_selection) > 0:
            self._sample_chunk_plan = samples_mod.build_chunk_plan(
                samples_selection,
                samples_chunk_size=int(self.root["sample_id"].chunks[0]),
            )
        else:
            # Empty selection: leaving plan as None sends call_*
            # reads through the _load_call_full path, which reads
            # every sample chunk. That's what lets a sample-scope
            # filter or AC/AN recompute still see the full genotype
            # matrix even though no samples will be emitted.
            self._sample_chunk_plan = None
        self._subsetting_samples = True

    def set_variants(self, variants) -> None:
        """Configure the variant selection.

        Accepts a list of :class:`~vcztools.utils.ChunkRead` (use
        :func:`vcztools.regions.build_chunk_plan` to build one from
        region/target strings and a root) or a sorted 1-D array of
        global variant indexes (bucketed into a plan internally).
        Raises ``RuntimeError`` if already configured (including if
        the default was already resolved by reading
        ``variant_chunk_plan``).
        """
        if self._variant_chunk_plan is not None:
            raise RuntimeError("variants already configured")
        if isinstance(variants, list):
            self._variant_chunk_plan = variants
        else:
            self._variant_chunk_plan = regions_mod.chunk_plan_from_indexes(
                np.asarray(variants),
                variants_chunk_size=self.variants_chunk_size,
            )

    def set_variant_filter(
        self,
        variant_filter: variant_filter_mod.VariantFilter,
        *,
        filter_on_subset_samples: bool = False,
    ) -> None:
        """Configure the variant filter.

        ``variant_filter`` is any object implementing the
        :class:`~vcztools.variant_filter.VariantFilter` protocol.
        ``filter_on_subset_samples`` controls which sample axis a
        sample-scope filter sees: ``False`` (default) matches
        ``bcftools view`` semantics (full sample axis);
        ``True`` matches ``bcftools query`` FMT-scope post-subset
        semantics. No-op for variant-scope filters.

        Raises ``RuntimeError`` if already configured.
        """
        if self.variant_filter is not None:
            raise RuntimeError("variant_filter already configured")
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
        """Filter IDs as fixed-length bytes (for VcfEncoder).
        Returns a single ``b"PASS"`` entry when the store has no
        ``filter_id`` array."""
        return _get_filter_ids(self.root)

    @functools.cached_property
    def num_variants(self) -> int:
        """Total variants in the store (before any plan/filter)."""
        return int(self.root["variant_position"].shape[0])

    @functools.cached_property
    def variants_chunk_size(self) -> int:
        """Chunk size along the variants axis."""
        return int(self.root["variant_position"].chunks[0])

    @functools.cached_property
    def num_samples(self) -> int:
        """Total samples in the store (including any masked entries)."""
        return int(self.root["sample_id"].shape[0])

    @functools.cached_property
    def all_sample_ids(self) -> np.ndarray:
        """Full ``sample_id`` array from the store, including any
        masked (empty-string) entries. For the post-subset order used
        when encoding rows, see :attr:`sample_ids`."""
        return self.root["sample_id"][:]

    @functools.cached_property
    def contig_lengths(self) -> np.ndarray | None:
        """``contig_length`` array, or ``None`` if absent."""
        if "contig_length" not in self.root:
            return None
        return self.root["contig_length"][:]

    @functools.cached_property
    def region_index(self) -> np.ndarray:
        """``region_index`` table. Raises ``ValueError`` if absent —
        callers that need it (e.g. :func:`vcztools.stats.stats`)
        should surface that to the user."""
        if "region_index" not in self.root:
            raise ValueError(
                "Could not load 'region_index' variable. "
                "Use 'vcz2zarr' to create an index."
            )
        return self.root["region_index"][:]

    @functools.cached_property
    def filter_descriptions(self) -> np.ndarray | None:
        """Per-filter descriptions, or ``None`` if absent."""
        if "filter_description" not in self.root:
            return None
        return self.root["filter_description"][:]

    @functools.cached_property
    def source(self) -> str | None:
        """Store-level ``source`` attribute, or ``None`` if not set."""
        return self.root.attrs.get("source", None)

    @functools.cached_property
    def vcf_meta_information(self) -> list | None:
        """Store-level ``vcf_meta_information`` attribute, or
        ``None`` if not set."""
        return self.root.attrs.get("vcf_meta_information", None)

    @functools.cached_property
    def field_names(self) -> frozenset[str]:
        """Set of field names present in the store."""
        return frozenset(self.root)

    @functools.cached_property
    def _field_info_cache(self) -> dict[str, FieldInfo]:
        return {}

    def get_field_info(self, name: str) -> FieldInfo:
        """Return a :class:`FieldInfo` snapshot for the named field.
        Reads Zarr metadata on first access, then memoizes per-field.
        Raises ``KeyError`` if the field is absent."""
        cache = self._field_info_cache
        if name not in cache:
            arr = self.root[name]
            cache[name] = FieldInfo(
                name=name,
                dtype=arr.dtype,
                shape=tuple(arr.shape),
                dims=tuple(utils.array_dims(arr)),
                attrs=dict(arr.attrs),
            )
        return cache[name]

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
        if self._subsetting_samples and not self.filter_on_subset_samples:
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
