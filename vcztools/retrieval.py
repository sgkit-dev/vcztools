import concurrent.futures as cf
import dataclasses
import functools
import logging
import time

import numpy as np

from vcztools import regions as regions_mod
from vcztools import samples as samples_mod
from vcztools import utils
from vcztools import variant_filter as variant_filter_mod
from vcztools.utils import (
    _as_fixed_length_string,
)

logger = logging.getLogger(__name__)


def _fmt_bytes(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GiB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MiB"
    if n >= 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n} bytes"


def _one_line_repr(obj) -> str:
    """Collapse ``repr(obj)`` to a single physical line.

    Some Zarr stores (notably icechunk) emit multi-line reprs that
    fragment our DEBUG log output. ``" ".join(repr(...).split())``
    keeps the discriminating fields on one line.
    """
    if obj is None:
        return "None"
    return " ".join(repr(obj).split())


DEFAULT_READAHEAD_BYTES = 256 * 1024 * 1024
# Fixed by design: these threads dispatch I/O to the Zarr backend
# (which already handles its own async/decompression parallelism),
# so usable parallelism is dispatch-bound and GIL-capped rather than
# scaling with cpu_count.
DEFAULT_READAHEAD_WORKERS = 32


def _read_block(arr, block_index: tuple) -> np.ndarray:
    """Fetch one Zarr block by block-index tuple."""
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


@dataclasses.dataclass(frozen=True, slots=True)
class BlockReadTemplate:
    """Variant-chunk-independent read pattern for one block.

    ``block_index_suffix`` is the part of the Zarr ``block_index``
    *after* the variant chunk index slot — empty tuple for a 1-D
    variant-axis field, ``(slice(None),) * (ndim - 1)`` for higher-D
    non-call fields, ``(sci, slice(None) * (ndim - 2))`` for ``call_*``.
    """

    key: tuple
    arr: object
    block_index_suffix: tuple


def create_chunk_read_list(
    root,
    sample_chunk_plan: "samples_mod.SampleChunkPlan",
    fields,
) -> list[BlockReadTemplate]:
    """Resolve ``fields`` to a list of :class:`BlockReadTemplate`
    once per query, before any variant chunk is visited.

    Each template carries the variant-chunk-independent parts of one
    block read — the cache key, the resolved Zarr array, and the
    suffix of ``block_index`` that follows the variant chunk index
    slot. :func:`update_chunk_read_list` substitutes a specific
    variant chunk index to produce executor-ready
    ``(key, arr, block_index)`` tuples.

    For a ``call_*`` field the template list fans out one entry per
    sample chunk in ``sample_chunk_plan.chunk_reads``; for any other
    field exactly one entry is produced.

    Every field must be variant-axis. Static (no-variants-axis) fields
    are handled by the reader's static-field cache, not the pipeline.
    """
    templates = []
    for field in fields:
        arr = root[field]
        assert _has_variants_axis(arr), f"non-variants-axis field in pipeline: {field}"
        if not field.startswith("call_"):
            suffix = (slice(None),) * (arr.ndim - 1)
            templates.append(
                BlockReadTemplate(key=(field,), arr=arr, block_index_suffix=suffix)
            )
        else:
            for cr in sample_chunk_plan.chunk_reads:
                suffix = (cr.index,) + (slice(None),) * (arr.ndim - 2)
                templates.append(
                    BlockReadTemplate(
                        key=(field, cr.index), arr=arr, block_index_suffix=suffix
                    )
                )
    return templates


def update_chunk_read_list(
    templates: list[BlockReadTemplate],
    variant_chunk_index: int,
) -> list[tuple]:
    """Substitute ``variant_chunk_index`` into each template, returning
    the ``[(key, arr, block_index), ...]`` list that
    :class:`ReadaheadPipeline` submits to the thread pool. The
    template list itself is unchanged.
    """
    reads = []
    for t in templates:
        block_index = (variant_chunk_index,) + t.block_index_suffix
        reads.append((t.key, t.arr, block_index))
    return reads


class ReadaheadPipeline:
    """Cross-chunk readahead controller for ``VczReader.variant_chunks``.

    Resolves the per-field read pattern once at init via
    :func:`create_chunk_read_list`, then for each entry in
    ``variant_chunk_plan``: substitute the variant chunk index via
    :func:`update_chunk_read_list`, submit the resulting block reads
    to the reader-owned thread pool, collect results into a
    ``blocks`` dict, then construct a :class:`CachedVariantChunk`
    over those prefetched blocks and yield it. Cross-chunk readahead
    overlaps later chunks' reads with the current chunk's processing
    in the consumer.

    The executor is supplied by the caller (typically
    :class:`VczReader`) and lives across pipelines. Multiple pipelines
    on the same reader — for example the BedEncoder shared-reader
    fanout — submit to a single shared pool. When iteration is
    abandoned mid-stream (consumer breaks early, generator closed,
    exception propagates), the pipeline cancels its own pending
    futures only; the executor itself outlives the pipeline.

    The window is sized by a byte budget rather than a chunk count:
    one variant-chunk prefetch can vary from a few MB (single
    sample-chunk read for a partial subset) to >1 GB (every
    sample chunk for a wide call_* field), so a count-based depth
    would either starve fan-out or blow RSS.

    Per-chunk byte cost is *measured*, not predicted: the first chunk
    is scheduled solo, and once its prefetched blocks land we sum
    their :func:`vcztools.utils.array_memory_bytes` and use that as
    the window-sizing estimate for every later chunk. The estimate is
    approximate —

    - The bootstrap chunk runs even when its prefetch alone exceeds
      ``readahead_bytes`` (the alternative is to never make progress).
    - Chunks can drift in content size across the iteration, especially
      when variable-length string fields are in the prefetch set, so
      later chunks may over- or under-shoot the budget.

    ``readahead_bytes=0`` pins pipeline depth at 1: the consumer's
    current chunk plus exactly one prefetched ahead. The pipeline
    never goes below depth 1 (the consumer would have to wait for
    every chunk's I/O on the request thread), so this is the
    smallest readahead the caller can ask for.
    """

    def __init__(
        self,
        root,
        variant_chunk_plan: list[utils.ChunkRead],
        sample_chunk_plan: "samples_mod.SampleChunkPlan",
        output_columns: np.ndarray | None,
        read_fields,
        *,
        readahead_bytes: int,
        executor: cf.ThreadPoolExecutor,
    ):
        self.root = root
        self._variant_chunk_plan_iter = iter(variant_chunk_plan)
        self._sample_chunk_plan = sample_chunk_plan
        self._output_columns = output_columns
        self._read_templates = create_chunk_read_list(
            root, sample_chunk_plan, read_fields
        )
        self._readahead_bytes = readahead_bytes
        # Set on the first chunk's completion in __iter__.
        self._per_chunk_bytes: int | None = None
        # Wall-clock seconds spent on the most recent chunk's block reads;
        # consumed by VczReader.variant_chunks to attribute per-chunk time
        # into "read" vs. "assemble".
        self.last_chunk_read_seconds: float | None = None
        # Sum of utils.array_memory_bytes() over the most recent chunk's
        # decompressed blocks; consumed by VczReader.variant_chunks to
        # accumulate retrieval-side throughput stats.
        self.last_chunk_bytes: int | None = None
        self._executor = executor
        # in_flight: list of (variant_chunk, [(blocks_key, Future), ...]).
        # The futures list is empty when the chunk needs no prefetch.
        self._in_flight: list = []
        logger.debug(
            f"ReadaheadPipeline init: {len(read_fields)} read_fields, "
            f"{len(self._read_templates)} templates, "
            f"readahead_bytes={_fmt_bytes(readahead_bytes)}"
        )

    def _schedule_one(self) -> bool:
        """Plan the next variant chunk's reads and submit them to the
        thread pool. Returns False once the plan is exhausted."""
        try:
            variant_chunk = next(self._variant_chunk_plan_iter)
        except StopIteration:
            return False
        reads = update_chunk_read_list(self._read_templates, variant_chunk.index)
        futures = [
            (key, self._executor.submit(_read_block, arr, block_index))
            for key, arr, block_index in reads
        ]
        self._in_flight.append((variant_chunk, futures))
        logger.debug(
            f"schedule chunk {variant_chunk.index}: {len(futures)} blocks submitted"
        )
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
        # avoid an infinite loop when read_fields is empty).
        per_chunk = max(1, self._per_chunk_bytes)
        while len(self._in_flight) == 0 or (
            len(self._in_flight) * per_chunk < self._readahead_bytes
        ):
            if not self._schedule_one():
                return

    def __iter__(self):
        try:
            self._refill()
            while len(self._in_flight) > 0:
                variant_chunk, futures = self._in_flight.pop(0)
                future_to_key = {fut: key for key, fut in futures}
                blocks: dict[tuple, np.ndarray] = {}
                read_start = time.perf_counter()
                for fut in cf.as_completed(future_to_key):
                    blocks[future_to_key[fut]] = fut.result()
                read_seconds = time.perf_counter() - read_start
                self.last_chunk_read_seconds = read_seconds
                chunk_bytes = sum(utils.array_memory_bytes(v) for v in blocks.values())
                self.last_chunk_bytes = chunk_bytes
                if self._per_chunk_bytes is None:
                    self._per_chunk_bytes = chunk_bytes
                    if self._readahead_bytes > 0 and chunk_bytes > 0:
                        window_chunks = max(
                            1, self._readahead_bytes // max(1, chunk_bytes)
                        )
                    else:
                        window_chunks = 1
                    logger.info(
                        f"Per-chunk read size: {_fmt_bytes(chunk_bytes)} "
                        f"(chunk {variant_chunk.index}); window will hold "
                        f"~{window_chunks} chunks under budget "
                        f"{_fmt_bytes(self._readahead_bytes)}"
                    )
                logger.debug(
                    f"chunk {variant_chunk.index} read complete in "
                    f"{read_seconds:.2f}s ({len(blocks)} blocks, "
                    f"{_fmt_bytes(chunk_bytes)})"
                )
                yield CachedVariantChunk(
                    self.root,
                    variant_chunk,
                    sample_chunk_plan=self._sample_chunk_plan,
                    output_columns=self._output_columns,
                    blocks=blocks,
                )
                # After the consumer drops the previous chunk reference,
                # top the pipeline back up.
                self._refill()
        finally:
            cancelled = 0
            for _variant_chunk, futures in self._in_flight:
                for _key, fut in futures:
                    if fut.cancel():
                        cancelled += 1
            if cancelled > 0:
                logger.debug(f"cancelled {cancelled} pending futures")


class CachedVariantChunk:
    """View assembler over prefetched blocks for one variant chunk visit.

    Constructed by :class:`ReadaheadPipeline` once its block reads have
    completed; performs no I/O itself. The constructor takes:

    - ``blocks`` — ``{key: ndarray}`` of prefetched Zarr blocks keyed by
      ``(field,)`` for variants-axis non-``call_*`` reads and
      ``(field, sci)`` for one ``call_*`` sample-chunk read. Keys are
      assigned by :func:`create_chunk_read_list`. Static fields (no
      variants axis) are not handled here — they live in the reader's
      static-field cache and are seeded directly into the per-chunk
      output by :meth:`VczReader.variant_chunks`.
    - ``sample_chunk_plan`` — the sample chunks the prefetch covers for
      every ``call_*`` field. In subset-mode this is the subset plan;
      in view-mode it is the non-null-samples plan. An empty plan
      (no ``chunk_reads``) is valid and produces zero-sample-column
      arrays without any prefetched ``call_*`` blocks. Non-``call_*``
      fields ignore it.
    - ``output_columns`` — indices into the read-plan axis that produce
      the subset axis. ``None`` when the read plan is already the
      subset axis (subset-mode) — :meth:`output_view` returns the
      assembled read untouched.

    Methods:

    - :meth:`filter_view` — assembled read at the read-plan axis.
    - :meth:`output_view` — subset-axis data. For ``call_*`` fields
      with ``output_columns`` set, a column slice of the read.

    Looking up a field that isn't in ``blocks`` raises ``KeyError``;
    the consumer must call view methods only for fields that were in
    the prefetch list.
    """

    def __init__(
        self,
        root,
        variant_chunk: utils.ChunkRead,
        *,
        sample_chunk_plan: samples_mod.SampleChunkPlan,
        output_columns: np.ndarray | None,
        blocks: dict[tuple, np.ndarray],
    ):
        self.root = root
        self.variant_chunk = variant_chunk
        self._sample_chunk_plan = sample_chunk_plan
        self._output_columns = output_columns
        self._blocks = blocks
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
        that need a C-contiguous buffer must convert at their boundary.
        """
        data = self.filter_view(field)
        if self._output_columns is None or not field.startswith("call_"):
            return data
        return data[:, self._output_columns]

    def _materialize(self, field: str) -> np.ndarray:
        arr = self.root[field]
        if field.startswith("call_"):
            return self._assemble_call(field, arr)
        return self._slice_variants(self._blocks[(field,)])

    def _slice_variants(self, block):
        sel = self.variant_chunk.selection
        if sel is None:
            return block
        return block[sel]

    def _assemble_call(self, field: str, arr) -> np.ndarray:
        plan = self._sample_chunk_plan
        if len(plan.chunk_reads) == 0:
            return self._empty_call_array(arr)
        parts = []
        for cr in plan.chunk_reads:
            raw = self._blocks[(field, cr.index)]
            if cr.selection is not None:
                raw = raw[:, cr.selection]
            parts.append(raw)

        sel = self.variant_chunk.selection
        # A slice selection is a basic-indexing view that costs nothing to
        # apply post-concat, so the fused path is only useful for the
        # ndarray (fancy-index) case where concat-then-slice would copy the
        # full block twice.
        if isinstance(sel, np.ndarray) and len(parts) > 1:
            n_samples = sum(p.shape[1] for p in parts)
            out_shape = (sel.size, n_samples) + parts[0].shape[2:]
            data = np.empty(out_shape, dtype=parts[0].dtype)
            col = 0
            for p in parts:
                cols = p.shape[1]
                data[:, col : col + cols] = p[sel]
                col += cols
        else:
            data = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
            if sel is not None:
                data = data[sel]

        if plan.permutation is not None:
            data = data[:, plan.permutation]
        return data

    def _empty_call_array(self, arr) -> np.ndarray:
        """Zero-sample-column array for a call_* field, without I/O.
        Used when the sample chunk plan is empty (e.g. set_samples([]))."""
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
    customize via the setters **before** iterating:

    - :meth:`set_samples` — sample selection (integer indexes).
    - :meth:`set_variants` — variant selection (``list[ChunkRead]``
      or a sorted 1-D index array). Re-callable; replaces.
    - :meth:`set_variant_filter` — per-variant filter predicate (or
      ``None`` to clear). Re-callable; replaces.
    - :meth:`set_filter_samples` — sample axis a sample-scope filter
      evaluates over.

    :meth:`set_samples` and :meth:`set_filter_samples` are one-shot —
    a second call raises ``RuntimeError``. ``set_samples`` also
    refuses to run when ``samples_selection`` has already been
    resolved by reading the corresponding property during default
    iteration.

    The reader owns a single :class:`concurrent.futures.ThreadPoolExecutor`
    that every :class:`ReadaheadPipeline` it spawns submits work to.
    Use as a context manager (``with VczReader(root) as reader:``) so
    the pool is torn down deterministically on exit. Multiple
    pipelines (e.g. several :class:`vcztools.plink.BedEncoder`
    instances driven concurrently against the same reader, or
    repeated ``variant_chunks()`` calls) share the pool — submission
    is thread-safe at the executor level.

    Parameters
    ----------
    root
        An already-opened :class:`zarr.Group` pointing at the VCZ
        dataset. Use :func:`vcztools.open_zarr` to open a path
        (local, remote, or zip) with the desired backend before
        constructing the reader.
    readahead_workers
        Worker count for the readahead thread pool. ``None``
        (default) uses :data:`DEFAULT_READAHEAD_WORKERS` (``32``).
        The pool is created at construction; this parameter has no
        post-init knob.
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
        readahead_workers: int | None = None,
        readahead_bytes: int | None = None,
    ):
        self.root = root
        self.readahead_bytes = readahead_bytes
        workers = (
            readahead_workers
            if readahead_workers is not None
            else DEFAULT_READAHEAD_WORKERS
        )
        self._executor = cf.ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="vcztools-readahead",
        )
        self._readahead_workers = workers
        self._sample_chunk_plan = None
        self._variant_chunk_plan = None
        self._samples_selection = None
        self._sample_ids = None
        self._filter_samples = None
        self.variant_filter = None
        # Concurrent ``variant_chunks()`` calls may both miss and both
        # write a static field here — last writer wins; both observers
        # get a valid array (dict assignment is GIL-atomic).
        self._static_field_cache: dict[str, np.ndarray] = {}
        logger.debug(
            f"VczReader init: store={_one_line_repr(getattr(root, 'store', None))}, "
            f"num_variants={self.num_variants}, num_samples={self.num_samples}, "
            f"variants_chunk_size={self.variants_chunk_size}, "
            f"samples_chunk_size={self.samples_chunk_size}, "
            f"readahead_workers={workers}, "
            f"readahead_bytes={readahead_bytes}"
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._executor.shutdown(wait=True)
        return False

    def _load_static_field(self, name: str) -> np.ndarray:
        """Read a static (no variants axis) field once and cache it on
        the reader. Static fields don't change across variant chunk
        visits, so paying for them per chunk in the readahead pipeline
        is wasteful.
        """
        cached = self._static_field_cache.get(name)
        if cached is not None:
            return cached
        value = self.root[name][:]
        self._static_field_cache[name] = value
        logger.debug(
            f"Loaded static field: {name} (shape={value.shape}, "
            f"dtype={value.dtype}, {_fmt_bytes(utils.array_memory_bytes(value))})"
        )
        return value

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
        logger.debug(
            f"Resolved default samples_selection: "
            f"{self._samples_selection.size} non-null samples"
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
            logger.debug(f"Built default variant_chunk_plan: {num_chunks} chunks")
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
        logger.debug(
            f"set_samples: {samples_selection.size} samples, "
            f"{len(self._sample_chunk_plan.chunk_reads)} sample chunks"
        )

    def set_variants(self, variants) -> None:
        """Configure the variant selection.

        Accepts a list of :class:`~vcztools.utils.ChunkRead` (use
        :func:`vcztools.regions.build_chunk_plan` to build one from
        region/target strings and a root) or a sorted 1-D array of
        global variant indexes (bucketed into a plan internally).

        May be called multiple times; each call replaces the prior
        selection. A ``variant_chunks()`` generator already iterating
        is unaffected — it snapshots the plan at start.
        """
        if isinstance(variants, list):
            self._variant_chunk_plan = variants
            logger.debug(
                f"set_variants: {len(self._variant_chunk_plan)} variant chunks "
                f"(from ChunkRead list)"
            )
        else:
            indexes = np.asarray(variants)
            self._variant_chunk_plan = regions_mod.chunk_plan_from_indexes(
                indexes,
                variants_chunk_size=self.variants_chunk_size,
            )
            logger.debug(
                f"set_variants: {indexes.size} variant indexes -> "
                f"{len(self._variant_chunk_plan)} variant chunks"
            )

    def set_variant_filter(
        self,
        variant_filter: variant_filter_mod.VariantFilter | None,
    ) -> None:
        """Configure (or clear) the variant filter.

        ``variant_filter`` is any object implementing the
        :class:`~vcztools.variant_filter.VariantFilter` protocol, or
        ``None`` to clear a previously-set filter. The sample axis a
        sample-scope filter evaluates over is controlled separately
        via :meth:`set_filter_samples`; the default axis is the user's
        sample selection (``bcftools query`` FMT-scope post-subset
        semantics). Call :meth:`set_filter_samples` with
        :attr:`non_null_sample_indices` for ``bcftools view`` pre-subset
        semantics.

        May be called multiple times; each call replaces the prior
        filter. A ``variant_chunks()`` generator already iterating is
        unaffected — it snapshots the filter at start.
        """
        self.variant_filter = variant_filter
        if variant_filter is None:
            logger.debug("set_variant_filter: cleared")
        else:
            logger.debug(
                f"set_variant_filter: referenced_fields="
                f"{sorted(variant_filter.referenced_fields)}"
            )

    def materialise_variant_filter(self) -> None:
        """Resolve the configured variant filter into a fixed selection.

        Walks ``variant_chunk_plan``, evaluates the filter, and replaces
        ``(variant_filter, variant_chunk_plan)`` with a chunk plan over
        the surviving global variant indexes. No-op if no filter is
        configured.

        Only variant-scope filters are supported. Sample-scope filters
        require reading on the filter sample axis and would duplicate
        much of :meth:`variant_chunks`; resolve them by iterating
        :meth:`variant_chunks` instead. Raises ``ValueError`` on a
        sample-scope filter.
        """
        variant_filter = self.variant_filter
        if variant_filter is None:
            return
        if variant_filter.scope != "variant":
            raise ValueError(
                "Sample-scope variant filters are not supported by "
                "materialise_variant_filter; iterate variant_chunks() "
                "to resolve them, or supply a variant-scope filter."
            )
        indexes = self._surviving_variant_indexes(variant_filter)
        self.set_variant_filter(None)
        self.set_variants(indexes)
        logger.debug(f"materialise_variant_filter: {indexes.size} surviving variants")

    def _surviving_variant_indexes(self, variant_filter) -> np.ndarray:
        """Walk ``variant_chunk_plan`` and evaluate ``variant_filter``,
        returning a sorted 1-D ``int64`` array of surviving global
        variant indexes. Caller is responsible for ensuring the filter
        is variant-scope."""
        chunk_size = self.variants_chunk_size
        static_fields = {}
        dynamic_fields = []
        for name in variant_filter.referenced_fields:
            arr = self.root[name]
            if _has_variants_axis(arr):
                dynamic_fields.append(name)
            else:
                static_fields[name] = arr[:]

        surviving = []
        for entry in self.variant_chunk_plan:
            base = entry.index * chunk_size
            chunk_data = dict(static_fields)
            for name in dynamic_fields:
                arr = self.root[name]
                length = min(chunk_size, arr.shape[0] - base)
                block = arr[base : base + length]
                if entry.selection is not None:
                    block = block[entry.selection]
                chunk_data[name] = block
            result = variant_filter.evaluate(chunk_data)
            local = np.flatnonzero(result)
            if entry.selection is None:
                global_idx = base + local
            elif isinstance(entry.selection, slice):
                offsets = np.arange(*entry.selection.indices(chunk_size))
                global_idx = base + offsets[local]
            else:
                global_idx = base + np.asarray(entry.selection)[local]
            surviving.append(global_idx)
        if len(surviving) == 0:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(surviving).astype(np.int64)

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
        logger.debug(f"set_filter_samples: {self._filter_samples.size} samples")

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
    def num_samples(self) -> int:
        """Total samples in the store (raw axis length, before any
        sample selection or null-sample filtering)."""
        return int(self.root["sample_id"].shape[0])

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
        start: int = 0,
    ):
        """Yield dict[str, np.ndarray] per variant chunk that passes the
        current variants/samples/variant-filter selection.

        ``start`` is an offset into ``variant_chunk_plan``; iteration
        begins at ``variant_chunk_plan[start]``. ``start=0`` (default)
        iterates the full plan. ``start >= len(variant_chunk_plan)``
        yields no chunks. Negative ``start`` raises ``ValueError``.

        The per-chunk flow:

        1. Iterate ``variant_chunk_plan[start:]``; each entry's
           ``selection`` pre-slices the chunk's variant axis.
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
        if start < 0:
            raise ValueError(f"start must be >= 0 (got {start})")
        if fields is not None and len(fields) == 0:
            return

        # Snapshot the filter so a mid-iteration set_variant_filter
        # can't change behaviour for this generator.
        variant_filter = self.variant_filter
        query_fields = self._resolve_query_fields(fields)
        filter_fields = frozenset(
            variant_filter.referenced_fields if variant_filter is not None else ()
        )
        if self._filter_samples is None:
            # Default: filter axis IS the subset axis. Covers non-empty
            # subsets, the no-subset default, and ``--drop-genotypes``
            # (empty subset collapses to an empty plan — no reads,
            # zero-column call_* output via CachedVariantChunk._empty_call_array).
            sample_chunk_plan = self.sample_chunk_plan
            output_columns = None
        else:
            # "bcftools view" mode: filter axis differs from subset. Read
            # on the filter axis; remap columns to produce the subset
            # output. Assumes ``samples_selection`` ⊆ ``filter_samples``.
            sample_chunk_plan = self.filter_sample_chunk_plan
            output_columns = np.searchsorted(
                self._filter_samples, self.samples_selection
            )

        # Split referenced fields into static (read once on the reader)
        # and dynamic (prefetched per variant chunk).
        referenced = list(dict.fromkeys([*filter_fields, *query_fields]))
        referenced_static_fields = {
            name: self._load_static_field(name)
            for name in referenced
            if not _has_variants_axis(self.root[name])
        }
        read_fields = [
            name for name in referenced if name not in referenced_static_fields
        ]
        readahead_bytes = (
            self.readahead_bytes
            if self.readahead_bytes is not None
            else DEFAULT_READAHEAD_BYTES
        )

        variant_chunk_plan = self.variant_chunk_plan
        if start > 0:
            variant_chunk_plan = variant_chunk_plan[start:]

        logger.info(
            f"variant_chunks: starting iteration "
            f"({len(query_fields)} query fields, {len(filter_fields)} filter fields, "
            f"{len(referenced_static_fields)} static fields, "
            f"{len(read_fields)} read fields, "
            f"{len(variant_chunk_plan)} variant chunks, "
            f"{len(sample_chunk_plan.chunk_reads)} sample chunks, "
            f"readahead_bytes={_fmt_bytes(readahead_bytes)}, "
            f"workers={self._readahead_workers})"
        )

        pipeline = ReadaheadPipeline(
            self.root,
            variant_chunk_plan,
            sample_chunk_plan,
            output_columns,
            read_fields,
            readahead_bytes=readahead_bytes,
            executor=self._executor,
        )
        chunks_visited = 0
        chunks_yielded = 0
        variants_yielded = 0
        bytes_yielded = 0
        iter_start = time.perf_counter()
        for chunk in pipeline:
            chunks_visited += 1
            chunk_start = time.perf_counter()
            read_seconds = pipeline.last_chunk_read_seconds or 0.0
            bytes_yielded += pipeline.last_chunk_bytes or 0
            # variants_selection: 1-D bool over the chunk's variant axis, or
            # None meaning "no filter, keep every variant".
            # sample_filter_pass: 2-D bool over (surviving variants, output
            # samples) for sample-scope filters only; published so query.py
            # can emit only matching samples in FORMAT-loop queries.
            variants_selection = None
            sample_filter_pass = None
            if variant_filter is not None:
                filter_data = {
                    f: referenced_static_fields[f]
                    if f in referenced_static_fields
                    else chunk.filter_view(f)
                    for f in filter_fields
                }
                filter_result = variant_filter.evaluate(filter_data)
                if filter_result.ndim == 1:
                    # Variant-scope filter: one bool per variant.
                    variants_selection = filter_result
                    logger.debug(
                        f"chunk {chunk.variant_chunk.index}: filter pass "
                        f"{int(filter_result.sum())}/{filter_result.size} variants"
                    )
                else:
                    # Sample-scope filter: a variant survives if at least one
                    # sample matched; the surviving rows are kept so the query
                    # layer can emit only matching samples.
                    variants_selection = filter_result.any(axis=1)
                    sample_filter_pass = filter_result[variants_selection]
                    if output_columns is not None:
                        # Filter ran on the real-sample axis but output is
                        # the user's subset axis; reindex columns to match.
                        sample_filter_pass = sample_filter_pass[:, output_columns]
                    logger.debug(
                        f"chunk {chunk.variant_chunk.index}: filter pass "
                        f"{int(variants_selection.sum())}/"
                        f"{variants_selection.size} variants, "
                        f"{int(filter_result.sum())}/{filter_result.size} "
                        f"sample cells"
                    )

            if variants_selection is not None and not variants_selection.any():
                continue

            chunk_data = {}
            for field in query_fields:
                if field in referenced_static_fields:
                    chunk_data[field] = referenced_static_fields[field]
                    continue
                value = chunk.output_view(field)
                if variants_selection is not None:
                    value = value[variants_selection]
                chunk_data[field] = value
            if sample_filter_pass is not None:
                chunk_data["sample_filter_pass"] = sample_filter_pass

            # Count surviving variants from a dynamic (variants-axis)
            # query field if there is one; static fields have the
            # store-wide axis length, not the per-chunk variant count.
            non_static_query = [
                f for f in query_fields if f not in referenced_static_fields
            ]
            if len(non_static_query) > 0:
                chunk_variants = len(chunk_data[non_static_query[0]])
            elif variants_selection is not None:
                chunk_variants = int(variants_selection.sum())
            else:
                chunk_variants = 0
            variants_yielded += chunk_variants
            chunks_yielded += 1
            total_seconds = time.perf_counter() - chunk_start + read_seconds
            assemble_seconds = max(0.0, total_seconds - read_seconds)
            logger.debug(
                f"chunk {chunk.variant_chunk.index}: yielded {chunk_variants} "
                f"variants in {total_seconds:.2f}s "
                f"(read {read_seconds:.2f}s, assemble {assemble_seconds:.2f}s)"
            )
            yield chunk_data
        elapsed = time.perf_counter() - iter_start
        mib = bytes_yielded / (1024 * 1024)
        rate = mib / elapsed if elapsed > 0 else 0.0
        logger.info(
            f"variant_chunks: iteration done in {elapsed:.2f}s "
            f"({chunks_visited} chunks visited, {chunks_yielded} yielded, "
            f"{variants_yielded} variants, "
            f"{mib:.1f} MiB retrieved, {rate:.1f} MiB/s)"
        )

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
