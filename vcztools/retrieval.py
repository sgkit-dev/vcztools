import concurrent.futures as cf
import dataclasses
import functools
import os
import sys

import numpy as np

from vcztools import regions as regions_mod
from vcztools import samples as samples_mod
from vcztools import utils
from vcztools import variant_filter as variant_filter_mod
from vcztools.utils import (
    _as_fixed_length_string,
)

DEFAULT_READAHEAD_BYTES = 256 * 1024 * 1024
DEFAULT_READAHEAD_WORKERS = min(32, (os.cpu_count() or 4))


def _read_block(arr, block_index: tuple) -> np.ndarray:
    """Fetch one Zarr block by block-index tuple, or the whole array
    when ``block_index`` is empty (used for static, non-chunked fields).
    """
    if len(block_index) == 0:
        return arr[:]
    return arr.blocks[block_index]


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


def _array_memory_bytes(arr: np.ndarray) -> int:
    """Best-effort in-memory footprint of ``arr``, in bytes.

    For fixed-size dtypes ``arr.nbytes`` is exact. For variable-length
    string dtypes it only covers the per-element metadata cells; the
    string content itself lives in Python heap (``object`` dtype) or
    numpy's ``StringDType`` arena, both of which we have to walk.

    - kind ``"O"`` (numpy ``object``): per-element ``sys.getsizeof``
      captures each Python str's header + content.
    - kind ``"T"`` (numpy ``StringDType``): ``arr.nbytes`` for the
      metadata cells, plus the UTF-8 byte length of every element.
    - everything else: ``arr.nbytes`` is exact.
    """
    if arr.dtype.kind == "O":
        return sum(sys.getsizeof(x) for x in arr.flat)
    if arr.dtype.kind == "T":
        content = sum(len(s.encode("utf-8")) for s in arr.flat)
        return int(arr.nbytes) + content
    return int(arr.nbytes)


class _ReadaheadPipeline:
    """Cross-chunk readahead controller for ``VczReader.variant_chunks``.

    Iterates ``variant_chunk_plan``, constructing a :class:`CachedVariantChunk`
    per entry and submitting each of its block reads to a shared
    thread pool so subsequent chunks' reads overlap with the current
    chunk's processing in the consumer.

    The window is sized by a byte budget rather than a chunk count:
    one variant-chunk prefetch can vary from a few MB (single
    sample-chunk read for a partial subset) to >1 GB (every
    sample chunk for a wide call_* field), so a count-based depth
    would either starve fan-out or blow RSS.

    Per-chunk byte cost is *measured*, not predicted: the first chunk
    is scheduled solo, and once its prefetched blocks land we sum
    their :func:`_array_memory_bytes` and use that as the window-sizing
    estimate for every later chunk. The estimate is approximate —

    - The bootstrap chunk runs even when its prefetch alone exceeds
      ``budget_bytes`` (the alternative is to never make progress).
    - Chunks can drift in content size across the iteration, especially
      when variable-length string fields are in the prefetch set, so
      later chunks may over- or under-shoot the budget.

    ``budget_bytes=0`` pins pipeline depth at 1: the consumer's
    current chunk plus exactly one prefetched ahead. The pipeline
    never goes below depth 1 (the consumer would have to wait for
    every chunk's I/O on the request thread), so this is the
    smallest readahead the caller can ask for.
    """

    def __init__(
        self,
        root,
        plan: list[utils.ChunkRead],
        read_plan: "samples_mod.SampleChunkPlan",
        output_columns: np.ndarray | None,
        prefetch_fields: list[str],
        *,
        budget_bytes: int,
        workers: int,
    ):
        self.root = root
        self._plan_iter = iter(plan)
        self._read_plan = read_plan
        self._output_columns = output_columns
        self._prefetch_fields = prefetch_fields
        self._budget_bytes = budget_bytes
        # Set on the first chunk's completion in __iter__.
        self._per_chunk_bytes: int | None = None
        self._executor = cf.ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="vcztools-readahead",
        )
        # in_flight: list of (CachedVariantChunk, [(cache_key, Future), ...]).
        # The futures list is empty when the chunk needed no prefetch.
        self._in_flight: list = []

    def _schedule_one(self) -> bool:
        """Construct the next CachedVariantChunk and submit its block reads
        to the thread pool. Returns False once the plan is exhausted."""
        try:
            chunk_read = next(self._plan_iter)
        except StopIteration:
            return False
        chunk = CachedVariantChunk(
            self.root,
            chunk_read,
            read_plan=self._read_plan,
            output_columns=self._output_columns,
        )
        reads = chunk._build_read_plan(self._prefetch_fields)
        futures = [
            (key, self._executor.submit(_read_block, arr, block_index))
            for key, arr, block_index in reads
        ]
        self._in_flight.append((chunk, futures))
        return True

    def _refill(self) -> None:
        # Until the first chunk has been measured we can't size the
        # window — schedule exactly one chunk and wait for its reads
        # to land. Subsequent refills fall through to the budget loop.
        if self._per_chunk_bytes is None:
            if len(self._in_flight) == 0:
                self._schedule_one()
            return
        # Always keep at least one chunk in flight; otherwise honour the
        # byte budget (use an effective per-chunk cost of at least 1 to
        # avoid an infinite loop when prefetch_fields is empty).
        per_chunk = max(1, self._per_chunk_bytes)
        while len(self._in_flight) == 0 or (
            len(self._in_flight) * per_chunk < self._budget_bytes
        ):
            if not self._schedule_one():
                return

    def __iter__(self):
        try:
            self._refill()
            while len(self._in_flight) > 0:
                chunk, futures = self._in_flight.pop(0)
                # as_completed surfaces an exception from any read at the
                # moment it occurs, instead of after every preceding future
                # has finished. Total wall time per chunk is unchanged —
                # we still need every read before yielding.
                future_to_key = {fut: key for key, fut in futures}
                for fut in cf.as_completed(future_to_key):
                    chunk._raw[future_to_key[fut]] = fut.result()
                if self._per_chunk_bytes is None:
                    self._per_chunk_bytes = sum(
                        _array_memory_bytes(v) for v in chunk._raw.values()
                    )
                yield chunk
                # After the consumer drops the previous chunk reference,
                # top the pipeline back up.
                self._refill()
        finally:
            self._executor.shutdown(wait=False, cancel_futures=True)


class CachedVariantChunk:
    """I/O + cache + sample-axis views for one variant chunk visit.

    Two reads' worth of logic collapses to one read plan plus an
    optional column slice:

    - ``read_plan`` — the sample chunks to fetch for every ``call_*``
      field. In subset-mode this is the subset plan; in view-mode it
      is the non-null-samples plan. An empty plan (no ``chunk_reads``)
      is valid and produces zero-sample-column arrays without any
      I/O. Non-``call_*`` fields ignore it.
    - ``output_columns`` — indices into the read-plan axis that
      produce the subset axis. ``None`` when the read plan is already
      the subset axis (subset-mode) — ``output_view`` returns the
      assembled read untouched.

    Methods:

    - :meth:`filter_view` — raw assembled read at the read-plan axis.
    - :meth:`output_view` — subset-axis data. For ``call_*`` fields
      with ``output_columns`` set, a column slice of the read.

    The raw-block cache is keyed by the underlying Zarr tuple; shared
    fields between filter and output reuse blocks by identity.
    """

    def __init__(
        self,
        root,
        variant_chunk: utils.ChunkRead,
        *,
        read_plan: samples_mod.SampleChunkPlan,
        output_columns: np.ndarray | None,
    ):
        self.root = root
        self.variant_chunk = variant_chunk
        self._read_plan = read_plan
        self._output_columns = output_columns
        # Raw Zarr data keyed by:
        #   (field,)         → static (full array) or variant-only block
        #   (field, sci)     → call_* raw sample-chunk block
        self._raw: dict[tuple, np.ndarray] = {}
        # Assembled read-axis arrays keyed by field.
        self._views: dict[str, np.ndarray] = {}

    def filter_view(self, field: str) -> np.ndarray:
        """Field data in the filter's sample axis (the read-plan axis)."""
        cached = self._views.get(field)
        if cached is not None:
            return cached
        value = self._materialize(field)
        self._views[field] = value
        return value

    def output_view(self, field: str) -> np.ndarray:
        """Field data in the output (subset) sample axis.

        For every field except view-mode ``call_*``, this is the same
        object as :meth:`filter_view`. For view-mode ``call_*``, the
        read-axis array is sliced down to the subset axis.

        The returned array may be a strided view over another buffer
        (e.g. when the view-mode slice uses fancy indexing). Callers
        that need a C-contiguous buffer — primarily the VCF C encoder
        — must convert at their boundary; everything else tolerates
        strided views.
        """
        data = self.filter_view(field)
        if self._output_columns is None or not field.startswith("call_"):
            return data
        return data[:, self._output_columns]

    def _build_read_plan(self, fields: list[str]) -> list[tuple]:
        """Return ``[(cache_key, arr, block_index), ...]`` for the
        block reads needed to satisfy ``fields`` on this variant chunk,
        skipping any blocks already cached.
        """
        reads = []
        for field in fields:
            arr = self.root[field]
            if not _has_variants_axis(arr):
                key = (field,)
                if key in self._raw:
                    continue
                reads.append((key, arr, ()))
            elif not field.startswith("call_"):
                key = (field,)
                if key in self._raw:
                    continue
                index = (self.variant_chunk.index,) + (slice(None),) * (arr.ndim - 1)
                reads.append((key, arr, index))
            else:
                for cr in self._read_plan.chunk_reads:
                    key = (field, cr.index)
                    if key in self._raw:
                        continue
                    index = (self.variant_chunk.index, cr.index) + (slice(None),) * (
                        arr.ndim - 2
                    )
                    reads.append((key, arr, index))
        return reads

    def _materialize(self, field: str) -> np.ndarray:
        arr = self.root[field]
        if not _has_variants_axis(arr):
            return self._load_non_call_raw(field, arr)
        if not field.startswith("call_"):
            block = self._load_non_call_raw(field, arr)
            return self._slice_variants(block)
        return self._assemble_call(field, arr)

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

    def _assemble_call(self, field: str, arr) -> np.ndarray:
        plan = self._read_plan
        parts = []
        for cr in plan.chunk_reads:
            raw = self._load_call_raw(field, arr, cr.index)
            if cr.selection is not None:
                raw = raw[:, cr.selection]
            parts.append(raw)
        if len(parts) == 0:
            return self._empty_call_array(arr)
        data = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        if plan.permutation is not None:
            data = data[:, plan.permutation]
        # The returned array can be a strided view — e.g. when the sample
        # selection was a prefix slice, or the permutation step gathered
        # columns on axis 1. Callers that need C-contiguous buffers (the
        # VCF C encoder) convert at their boundary; every other consumer
        # tolerates strided views, so we avoid a full-chunk memcpy here.
        return self._slice_variants(data)

    def _empty_call_array(self, arr) -> np.ndarray:
        """Zero-sample-column array for a call_* field, without I/O.
        Used when the read plan is empty (e.g. set_samples([]))."""
        sel = self.variant_chunk.selection
        if sel is not None:
            n_variants = int(sel.size)
        else:
            chunk_size_v = int(arr.chunks[0])
            chunk_start = self.variant_chunk.index * chunk_size_v
            n_variants = min(chunk_size_v, int(arr.shape[0]) - chunk_start)
        return np.empty((n_variants, 0) + tuple(arr.shape[2:]), dtype=arr.dtype)


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
    readahead_threads
        Worker count for the per-call readahead thread pool. ``None``
        (default) uses :data:`DEFAULT_READAHEAD_WORKERS`
        (``min(32, cpu_count)``).
    readahead_bytes
        Cap, in bytes, on the cross-chunk readahead window. ``None``
        (default) uses :data:`DEFAULT_READAHEAD_BYTES` (256 MiB).
        ``0`` pins pipeline depth at 1 (one chunk prefetched ahead of
        the consumer); the pipeline cannot go lower.
    """

    def __init__(
        self,
        root,
        *,
        readahead_threads: int | None = None,
        readahead_bytes: int | None = None,
    ):
        self.root = root
        self.readahead_threads = readahead_threads
        self.readahead_bytes = readahead_bytes
        self._sample_chunk_plan = None
        self._variant_chunk_plan = None
        self._samples_selection = None
        self._sample_ids = None
        self._filter_samples = None
        self.variant_filter = None

    def _resolve_samples_if_needed(self) -> None:
        if self._samples_selection is not None:
            return
        raw_sample_ids = self.raw_sample_ids
        self._samples_selection = np.flatnonzero(raw_sample_ids != "")
        self._sample_ids = raw_sample_ids[self._samples_selection]
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
        samples_selection = self._normalize_sample_indexes(samples, label="sample")
        self._samples_selection = samples_selection
        self._sample_ids = self.raw_sample_ids[samples_selection]
        self._sample_chunk_plan = samples_mod.build_chunk_plan(
            samples_selection,
            samples_chunk_size=self.samples_chunk_size,
        )

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
    ) -> None:
        """Configure the variant filter.

        ``variant_filter`` is any object implementing the
        :class:`~vcztools.variant_filter.VariantFilter` protocol. The
        sample axis a sample-scope filter evaluates over is controlled
        separately via :meth:`set_filter_samples`; the default axis is
        the user's sample selection (``bcftools query`` FMT-scope
        post-subset semantics). Call :meth:`set_filter_samples` with
        :attr:`non_null_sample_indices` for ``bcftools view`` pre-subset
        semantics.

        Raises ``RuntimeError`` if already configured.
        """
        if self.variant_filter is not None:
            raise RuntimeError("variant_filter already configured")
        self.variant_filter = variant_filter

    def set_filter_samples(self, filter_samples) -> None:
        """Configure the sample axis that a sample-scope variant filter
        evaluates over.

        Accepts a list or 1-D integer ndarray of sample indexes. Must
        be **sorted ascending, unique, and in range**
        ``[0, num_samples)``. Out-of-range, unsorted, or duplicate
        indexes raise ``ValueError``. Raises ``RuntimeError`` if
        already configured.

        When not called, the filter axis defaults to
        :attr:`samples_selection`. For ``bcftools view`` pre-subset
        semantics, pass :attr:`non_null_sample_indices`.
        """
        if self._filter_samples is not None:
            raise RuntimeError("filter_samples already configured")
        self._filter_samples = self._normalize_sample_indexes(
            filter_samples, label="filter_samples", sorted_unique=True
        )

    def _normalize_sample_indexes(
        self, value, *, label: str, sorted_unique: bool = False
    ) -> np.ndarray:
        """Validate and convert a sample-index input to a 1-D int64 ndarray.

        Checks input type (via :func:`_validate_samples_input`) and that
        every index is in ``[0, num_samples)``. When ``sorted_unique=True``,
        also requires strictly ascending order. ``label`` is interpolated
        into error messages.
        """
        _validate_samples_input(value)
        arr = np.asarray(value, dtype=np.int64)
        if arr.size > 0:
            lo = arr.min()
            hi = arr.max()
            raw_size = self.raw_sample_ids.size
            if lo < 0 or hi >= raw_size:
                raise ValueError(
                    f"{label} index out of range: must be in [0, {raw_size})"
                )
            if sorted_unique and np.any(np.diff(arr) <= 0):
                raise ValueError(f"{label} must be sorted ascending and unique")
        return arr

    @property
    def filter_sample_chunk_plan(self) -> samples_mod.SampleChunkPlan:
        """Chunk plan for reading the filter sample axis. When
        :meth:`set_filter_samples` has not been called this is the
        same as :attr:`sample_chunk_plan`; otherwise it is built
        fresh from the configured filter samples."""
        if self._filter_samples is None:
            return self.sample_chunk_plan
        return samples_mod.build_chunk_plan(
            self._filter_samples,
            samples_chunk_size=self.samples_chunk_size,
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
    def samples_chunk_size(self) -> int:
        """Chunk size along the samples axis."""
        return int(self.root["sample_id"].chunks[0])

    @functools.cached_property
    def raw_sample_ids(self) -> np.ndarray:
        """Full ``sample_id`` array from the store, including any
        null-string entries. For the post-subset order used when
        encoding rows, see :attr:`sample_ids`."""
        return self.root["sample_id"][:]

    @functools.cached_property
    def non_null_sample_indices(self) -> np.ndarray:
        """Global indices of non-null samples in ``sample_id``. Sorted;
        empty if every entry is null."""
        return np.flatnonzero(self.raw_sample_ids != "")

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
        2. Construct a :class:`CachedVariantChunk` scoped to this variant
           chunk. It owns the raw-block cache and the
           subset-vs-real-axis decision.
        3. Evaluate the filter against ``CachedVariantChunk.filter_view`` for
           each referenced field. Collapse a 2-D sample-scope mask
           into a 1-D variant selection (with the surviving rows kept
           as ``sample_filter_pass`` on the subset axis).
        4. Assemble output from ``CachedVariantChunk.output_view`` for each
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
        if self._filter_samples is None:
            # Default: filter axis IS the subset axis. Covers non-empty
            # subsets, the no-subset default, and ``--drop-genotypes``
            # (empty subset collapses to an empty plan — no reads,
            # zero-column call_* output via CachedVariantChunk._empty_call_array).
            read_plan = self.sample_chunk_plan
            output_columns = None
        else:
            # Explicit view mode: filter axis differs from subset. Read
            # on the filter axis; remap columns to produce the subset
            # output. Assumes ``samples_selection`` ⊆ ``filter_samples``;
            # an explicitly-included null slot (edge case) produces an
            # undefined ``sample_filter_pass`` position. The CLI refuses
            # empty view-mode subsets (``view -s ^<all>`` etc.), so the
            # empty-subset AC/AN quirk cannot reach this branch via the
            # CLI; Python-API callers opting into this shape will see
            # zero-column output.
            read_plan = self.filter_sample_chunk_plan
            output_columns = np.searchsorted(
                self._filter_samples, self.samples_selection
            )

        # Prefetch the union of filter + query fields in one pass per
        # variant chunk. We used to split into two stages — filter
        # fields first, short-circuit if none pass, then query fields
        # — but the short-circuit only fires when a whole chunk has
        # zero passing variants. For typical filters (INFO/DP>N,
        # QUAL>N) and chunk sizes in the thousands that is
        # vanishingly rare; it just forfeits cross-chunk readahead on
        # every filtered query. Pay the rare chunk-rejection bandwidth
        # waste and keep the pipeline uniform.
        prefetch_union = list(dict.fromkeys([*filter_fields, *query_fields]))
        budget_bytes = (
            self.readahead_bytes
            if self.readahead_bytes is not None
            else DEFAULT_READAHEAD_BYTES
        )
        workers = (
            self.readahead_threads
            if self.readahead_threads is not None
            else DEFAULT_READAHEAD_WORKERS
        )
        for chunk in _ReadaheadPipeline(
            self.root,
            self.variant_chunk_plan,
            read_plan,
            output_columns,
            prefetch_union,
            budget_bytes=budget_bytes,
            workers=workers,
        ):
            # variants_selection: 1-D bool over the chunk's variant axis, or
            # None meaning "no filter, keep every variant".
            # sample_filter_pass: 2-D bool over (surviving variants, output
            # samples) for sample-scope filters only; published so query.py
            # can emit only matching samples in FORMAT-loop queries.
            variants_selection = None
            sample_filter_pass = None
            if self.variant_filter is not None:
                filter_data = {f: chunk.filter_view(f) for f in filter_fields}
                filter_result = self.variant_filter.evaluate(filter_data)
                if filter_result.ndim == 1:
                    # Variant-scope filter: one bool per variant.
                    variants_selection = filter_result
                else:
                    # Sample-scope filter: a variant survives if at least one
                    # sample matched; the surviving rows are kept so the query
                    # layer can emit only matching samples.
                    variants_selection = filter_result.any(axis=1)
                    sample_filter_pass = filter_result[variants_selection]
                    if output_columns is not None:
                        # Filter ran on the real-sample axis but output is the
                        # user's subset axis; reindex columns to match.
                        sample_filter_pass = sample_filter_pass[:, output_columns]

            if variants_selection is not None and not variants_selection.any():
                continue

            chunk_data = {}
            for field in query_fields:
                value = chunk.output_view(field)
                if variants_selection is not None and _has_variants_axis(
                    self.root[field]
                ):
                    value = value[variants_selection]
                chunk_data[field] = value
            if sample_filter_pass is not None:
                chunk_data["sample_filter_pass"] = sample_filter_pass
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
