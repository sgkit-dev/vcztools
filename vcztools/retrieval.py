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


def _has_variants_axis(arr) -> bool:
    """Whether ``arr``'s first dimension is the variants axis."""
    dims = utils.array_dims(arr)
    return dims is not None and len(dims) > 0 and dims[0] == "variants"


@dataclasses.dataclass(frozen=True)
class FieldRead:
    """Per-field fetch spec for one variant chunk visit.

    Used by :meth:`CachedChunk.prefetch` to describe which Zarr blocks
    to warm into the cache. Not required at the caller interface; the
    ``filter_view`` / ``output_view`` methods know which blocks to
    fetch on demand. This data structure is the hook where future
    async I/O and readahead will fan out reads concurrently.

    - ``sample_chunk_reads is None`` → the field has no samples axis
      (``variant_*`` or static).
    - ``sample_chunk_reads`` is a list of :class:`~vcztools.utils.ChunkRead`
      → a ``call_*`` field; each entry names a sample chunk and the
      local selection to apply on read.
    - ``sample_permutation`` (optional) reorders the concatenated sample
      axis after per-chunk subsetting.
    """

    field: str
    sample_chunk_reads: list[utils.ChunkRead] | None = None
    sample_permutation: np.ndarray | None = None


@dataclasses.dataclass(frozen=True)
class VariantChunkReadPlan:
    """Fetch spec for one variant chunk visit.

    Only consumed by :meth:`CachedChunk.prefetch`; the main read
    interface is :meth:`CachedChunk.filter_view` / :meth:`output_view`.
    """

    variant_chunk: utils.ChunkRead
    field_reads: list[FieldRead]


class CachedChunk:
    """I/O + cache + sample-axis views for one variant chunk visit.

    Encapsulates the per-chunk read state that used to live across
    :class:`VariantChunkReader` + :meth:`VczReader._reduce_chunk_mask`:

    - :meth:`filter_view` — returns ``field`` as the filter should see
      it. In subset-mode (``filter_on_subset_samples=True``) that is
      the subset sample axis. In view-mode (``False``) that is the
      real-samples axis (masked-slot-free).
    - :meth:`output_view` — returns ``field`` on the subset sample
      axis. Output is always post-subset.
    - :meth:`reduce_mask` — converts a 2-D filter-axis mask into
      ``(variants_selection, subset-axis call_mask)``; applies the
      real→subset axis remap internally for view-mode.

    Non-``call_*`` fields (static, ``variant_*``) don't have a
    differentiated sample axis; ``filter_view`` and ``output_view``
    return the same array. Subset-mode call_\\* fields behave the
    same way. Only view-mode ``call_*`` fields have two distinct
    views, and even those share underlying raw Zarr blocks via the
    cache.
    """

    def __init__(
        self,
        root,
        variant_chunk: utils.ChunkRead,
        *,
        subset_plan: samples_mod.SampleChunkPlan | None,
        real_plan: samples_mod.SampleChunkPlan | None,
        real_to_subset: np.ndarray | None,
        filter_on_subset_samples: bool,
    ):
        self.root = root
        self.variant_chunk = variant_chunk
        self._subset_plan = subset_plan
        self._real_plan = real_plan
        self._real_to_subset = real_to_subset
        self._filter_on_subset = filter_on_subset_samples
        self._filter_plan = subset_plan if filter_on_subset_samples else real_plan
        # Raw Zarr data keyed by:
        #   (field,)         → static (full array) or variant-only block
        #   (field, sci)     → call_* raw sample-chunk block
        self._raw: dict[tuple, np.ndarray] = {}
        # Assembled per-view arrays keyed by (field, view_kind) where
        # view_kind is "shared" (non-call or subset-mode), "filter"
        # (view-mode call_* on real axis), or "output" (view-mode
        # call_* on subset axis).
        self._views: dict[tuple[str, str], np.ndarray] = {}

    def filter_view(self, field: str) -> np.ndarray:
        """Field data in the filter's sample axis.

        Subset-mode: subset axis. View-mode: real-samples axis.
        """
        return self._view(field, "filter")

    def output_view(self, field: str) -> np.ndarray:
        """Field data in the output's sample axis (always subset)."""
        return self._view(field, "output")

    def reduce_mask(self, v_mask):
        """Convert a filter-scope mask into ``(variants_selection, call_mask)``.

        ``v_mask is None`` or 1-D: the mask is the variant selection;
        ``call_mask`` is ``None``.

        2-D mask: ``np.any`` along the sample axis produces the variant
        selection; the surviving rows become ``call_mask``. In view-mode,
        ``call_mask`` is in the real-samples axis and is remapped to the
        subset axis so consumers can use it against ``output_view``.
        """
        if v_mask is None or v_mask.ndim == 1:
            return v_mask, None
        variants_selection = np.any(v_mask, axis=1)
        call_mask = v_mask[variants_selection]
        if not self._filter_on_subset:
            call_mask = call_mask[:, self._real_to_subset]
        return variants_selection, call_mask

    def prefetch(self, plan: VariantChunkReadPlan) -> None:
        """Warm the raw-block cache from an explicit plan.

        Today this is a synchronous eager load; future iterations will
        fan out as concurrent async reads so that by the time
        ``filter_view`` / ``output_view`` are called, all blocks are
        already in memory. Callers that don't prefetch still get
        correct behaviour — views fetch on demand.
        """
        for fr in plan.field_reads:
            arr = self.root[fr.field]
            if fr.sample_chunk_reads is None:
                self._load_non_call_raw(fr.field, arr)
            else:
                for cr in fr.sample_chunk_reads:
                    self._load_call_raw(fr.field, arr, cr.index)

    def _view(self, field: str, view_kind: str) -> np.ndarray:
        actual_kind = self._effective_kind(field, view_kind)
        key = (field, actual_kind)
        cached = self._views.get(key)
        if cached is not None:
            return cached
        value = self._materialize(field, actual_kind)
        self._views[key] = value
        return value

    def _effective_kind(self, field: str, view_kind: str) -> str:
        # Only view-mode call_* fields have distinct filter/output views.
        if not field.startswith("call_") or self._filter_on_subset:
            return "shared"
        return view_kind

    def _materialize(self, field: str, kind: str) -> np.ndarray:
        arr = self.root[field]
        if not _has_variants_axis(arr):
            return self._load_non_call_raw(field, arr)
        if not field.startswith("call_"):
            block = self._load_non_call_raw(field, arr)
            return self._slice_variants(block)
        plan = self._filter_plan if kind == "filter" else self._subset_plan
        return self._assemble_call(field, arr, plan)

    def _slice_variants(self, block):
        sel = self.variant_chunk.selection
        if sel is None:
            return block
        return block[sel]

    def _load_non_call_raw(self, field: str, arr) -> np.ndarray:
        key = (field,)
        cached = self._raw.get(key)
        if cached is not None:
            return cached
        if _has_variants_axis(arr):
            value = arr.blocks[self.variant_chunk.index]
        else:
            value = arr[:]
        self._raw[key] = value
        return value

    def _load_call_raw(self, field: str, arr, sci: int) -> np.ndarray:
        key = (field, sci)
        cached = self._raw.get(key)
        if cached is not None:
            return cached
        index = (self.variant_chunk.index, sci) + (slice(None),) * (arr.ndim - 2)
        value = arr.blocks[index]
        self._raw[key] = value
        return value

    def _assemble_call(
        self, field: str, arr, plan: samples_mod.SampleChunkPlan | None
    ) -> np.ndarray:
        if plan is None:
            # Empty subset (set_samples([])) legacy path: read the full
            # on-disk sample axis so sample-scope filters and legacy
            # call_* output can still see every sample.
            num_s_chunks = int(arr.cdata_shape[1])
            parts = [
                self._load_call_raw(field, arr, sci) for sci in range(num_s_chunks)
            ]
        else:
            parts = []
            for cr in plan.chunk_reads:
                raw = self._load_call_raw(field, arr, cr.index)
                if cr.selection is not None:
                    raw = raw[:, cr.selection]
                parts.append(raw)
        data = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        if plan is not None and plan.permutation is not None:
            data = data[:, plan.permutation]
        data = self._slice_variants(data)
        # Advanced indexing along a non-first axis of a 3-D array yields a
        # non-C-contiguous view; ascontiguousarray gives the C encoder the
        # contiguous buffer it requires.
        return np.ascontiguousarray(data)


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
        self.filter_on_subset_samples = True

    def _resolve_samples_if_needed(self) -> None:
        if self._samples_selection is not None:
            return
        all_samples = self.all_sample_ids
        self._samples_selection = np.flatnonzero(all_samples != "")
        self._sample_ids = all_samples[self._samples_selection]
        self._sample_chunk_plan = samples_mod.build_chunk_plan(
            self._samples_selection,
            samples_chunk_size=self.samples_chunk_size,
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
                samples_chunk_size=self.samples_chunk_size,
            )
        else:
            # Empty selection: leaving plan as None sends call_*
            # reads through the _load_call_full path, which reads
            # every sample chunk. That's what lets a sample-scope
            # filter or AC/AN recompute still see the full genotype
            # matrix even though no samples will be emitted.
            self._sample_chunk_plan = None

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
        filter_on_subset_samples: bool = True,
    ) -> None:
        """Configure the variant filter.

        ``variant_filter`` is any object implementing the
        :class:`~vcztools.variant_filter.VariantFilter` protocol.
        ``filter_on_subset_samples`` controls which sample axis a
        sample-scope filter sees: ``True`` (default) matches
        ``bcftools query`` FMT-scope post-subset semantics (filter
        sees only the selected samples); ``False`` matches
        ``bcftools view`` pre-subset semantics (filter evaluated over
        every real sample, regardless of the user subset). No-op for
        variant-scope filters.

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
    def samples_chunk_size(self) -> int:
        """Chunk size along the samples axis."""
        return int(self.root["sample_id"].chunks[0])

    @functools.cached_property
    def all_sample_ids(self) -> np.ndarray:
        """Full ``sample_id`` array from the store, including any
        masked (empty-string) entries. For the post-subset order used
        when encoding rows, see :attr:`sample_ids`."""
        return self.root["sample_id"][:]

    @functools.cached_property
    def real_sample_indices(self) -> np.ndarray:
        """Global indices of real (non-masked) samples in
        ``sample_id``. Sorted; empty if every entry is masked."""
        return np.flatnonzero(self.all_sample_ids != "")

    @functools.cached_property
    def real_sample_chunk_plan(self) -> samples_mod.SampleChunkPlan | None:
        """:class:`~vcztools.samples.SampleChunkPlan` for every real
        sample. Used by :class:`CachedChunk` to build view-mode filter
        views. ``None`` when no real samples exist."""
        if self.real_sample_indices.size == 0:
            return None
        return samples_mod.build_chunk_plan(
            self.real_sample_indices,
            samples_chunk_size=self.samples_chunk_size,
        )

    @functools.cached_property
    def _real_to_subset_indices(self) -> np.ndarray:
        """For each entry in ``samples_selection``, its position in
        :attr:`real_sample_indices`. Used to remap a view-mode 2-D
        call-mask back onto the subset axis. Entries that don't match
        (e.g. a masked slot explicitly included in the subset via
        ``set_samples``) are an edge case; the resulting ``call_mask``
        position is still indexed but the guarantee is undefined."""
        return np.searchsorted(self.real_sample_indices, self.samples_selection)

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

        The per-chunk flow:

        1. Iterate ``variant_chunk_plan``; each entry's ``selection``
           pre-slices the chunk's variant axis.
        2. Construct a :class:`CachedChunk` scoped to this variant
           chunk. It owns the raw-block cache and the
           subset-vs-real-axis decision.
        3. Evaluate the filter against ``CachedChunk.filter_view`` for
           each referenced field.
        4. ``CachedChunk.reduce_mask`` turns the filter's mask into a
           variant selection and an (optional) subset-axis ``call_mask``.
        5. Assemble output from ``CachedChunk.output_view`` for each
           query field; apply the variant selection to variants-axis
           fields.
        """
        if fields is not None and len(fields) == 0:
            return

        query_fields = self._resolve_query_fields(fields)
        filter_fields = frozenset(
            self.variant_filter.referenced_fields
            if self.variant_filter is not None
            else ()
        )

        for chunk_read in self.variant_chunk_plan:
            chunk = CachedChunk(
                self.root,
                chunk_read,
                subset_plan=self.sample_chunk_plan,
                real_plan=self.real_sample_chunk_plan,
                real_to_subset=self._real_to_subset_indices,
                filter_on_subset_samples=self.filter_on_subset_samples,
            )
            v_mask = None
            if self.variant_filter is not None:
                filter_data = {f: chunk.filter_view(f) for f in filter_fields}
                v_mask = self.variant_filter.evaluate(filter_data)
            if v_mask is not None and not np.any(v_mask):
                continue
            variants_selection, call_mask = chunk.reduce_mask(v_mask)
            chunk_data = {}
            for field in query_fields:
                value = chunk.output_view(field)
                if variants_selection is not None and _has_variants_axis(
                    self.root[field]
                ):
                    value = value[variants_selection]
                chunk_data[field] = value
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
