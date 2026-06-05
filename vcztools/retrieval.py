"""VCZ reader: chunked-iteration entry points for VCF Zarr stores.

The variants axis is iterated in *stream* chunks, sized per query as
the minimum ``chunks[0]`` across the read fields (or ``min_chunk`` —
the ``call_*`` chunk size — when no read fields are involved). For a
query touching ``call_*``, the stream chunk size equals ``min_chunk``;
for queries touching only variant-only fields with larger chunks, the
stream chunk equals that field's chunk size. The canonical
``variant_chunk_plan`` on :class:`VczReader` stays in ``min_chunk``
units; per query, :func:`vcztools.utils.rebucket_to_stream_plan`
re-expresses it in stream-chunk units.

:class:`StreamReader` drives the per-query iteration. Each unique Zarr
block is submitted to the reader's thread pool exactly once: the
reader walks ``stream_plan`` in order, derives the per-(field,
sample-chunk) block records for the current chunk on the fly, submits
any block whose ``block_key`` is not already live, and evicts blocks
once the next stream chunk no longer references them. The per-chunk
``CachedLogicalVariantsChunk`` view assembler receives blocks that have already
been intra-sliced to the stream chunk's variant rows.
"""

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
from vcztools import virtual_fields as virtual_fields_mod
from vcztools.utils import (
    _as_fixed_length_string,
)

logger = logging.getLogger(__name__)

# Sub-DEBUG level for very-high-cardinality per-block / per-chunk
# scheduler events (one or more lines per variant chunk). Enable with
# logging.getLogger("vcztools.retrieval").setLevel(retrieval.TRACE).
TRACE = logging.DEBUG - 5


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


def _freeze(arr: np.ndarray) -> np.ndarray:
    """Mark an array read-only and return it. Used at every site where
    ``VczReader`` hands an internally-cached array back to a caller —
    a mutation by the caller would corrupt the cache or a sibling
    view into a shared buffer.
    """
    arr.flags.writeable = False
    return arr


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


@dataclasses.dataclass(frozen=True, slots=True)
class FieldSpec:
    """Per-query, per-field constants for stream-reader block derivation.

    Built once at :class:`StreamReader` init from each entry in the
    read-fields list. For non-``call_*`` fields one spec covers all
    stream chunks; for ``call_*`` fields the same spec covers every
    sample chunk (the sample chunk index is a runtime parameter of
    :meth:`StreamReader._derive_blocks`).

    ``multiplier`` is ``arr.chunks[0] // stream_chunk_size`` (exact
    integer — :func:`vcztools.utils.compute_stream_chunk_size` returns
    a value that divides every read field's chunk size). It is ``1``
    for any field whose chunk size equals ``stream_chunk_size``
    (always the case for ``call_*`` when ``call_*`` is in
    ``read_fields``, since ``call_*`` chunks define ``min_chunk`` and
    ``stream_chunk_size`` collapses to ``min_chunk`` then); it is
    ``>1`` for variant-only fields with bigger chunks. When
    ``multiplier > 1``, one Zarr block spans ``multiplier`` consecutive
    stream chunks and gets intra-sliced to the current chunk's rows.

    ``block_size_estimate`` is ``dtype.itemsize * prod(arr.chunks)`` —
    the full uncompressed block-shape footprint, used for the readahead
    byte budget. Partial-edge blocks (last variant block, last sample
    chunk) are smaller in practice and the budget over-estimates them.
    Variable-length string fields (object / StringDType) report only
    the cell-metadata size here, *not* the content bytes; the budget
    under-counts those.

    ``block_index_suffix`` is the part of the Zarr ``block_index``
    *after* the variants and (for ``call_*``) sample-chunk slots —
    ``(slice(None),) * (ndim - 1)`` for non-``call_*`` fields and
    ``(slice(None),) * (ndim - 2)`` for ``call_*`` fields.
    """

    field: str
    arr: object
    multiplier: int
    is_call: bool
    block_size_estimate: int
    block_index_suffix: tuple


def _make_field_specs(root, read_fields, stream_chunk_size: int) -> list[FieldSpec]:
    """Resolve ``read_fields`` to one :class:`FieldSpec` per field.

    Every field must be variants-axis. Static (no-variants-axis) fields
    are handled by the reader's static-field cache and never reach the
    stream reader.
    """
    specs = []
    for field in read_fields:
        arr = root[field]
        assert utils.has_variants_axis(arr), (
            f"non-variants-axis field in stream reader: {field}"
        )
        chunks = arr.chunks
        if int(chunks[0]) % stream_chunk_size != 0:
            raise ValueError(
                f"field {field!r} chunks[0]={chunks[0]} is not a multiple of "
                f"stream_chunk_size={stream_chunk_size}"
            )
        multiplier = int(chunks[0]) // stream_chunk_size
        is_call = field.startswith("call_")
        block_size = int(arr.dtype.itemsize)
        for d in chunks:
            block_size *= int(d)
        if is_call:
            suffix = (slice(None),) * (arr.ndim - 2)
        else:
            suffix = (slice(None),) * (arr.ndim - 1)
        specs.append(
            FieldSpec(
                field=field,
                arr=arr,
                multiplier=multiplier,
                is_call=is_call,
                block_size_estimate=block_size,
                block_index_suffix=suffix,
            )
        )
    return specs


@dataclasses.dataclass(frozen=True, slots=True)
class _BlockRecord:
    """One Zarr block needed by a stream chunk.

    Transient — built fresh each time :meth:`StreamReader._derive_blocks`
    runs (memoised per stream-chunk index in ``StreamReader._derived``).
    ``block_key`` deduplicates across stream chunks; ``dest_key`` is
    the slot in :class:`CachedLogicalVariantsChunk`'s blocks dict.
    """

    block_key: tuple
    dest_key: tuple
    arr: object
    block_index: tuple
    intra_slice: slice
    size_estimate: int


class StreamReader:
    """Cross-stream-chunk readahead controller for
    ``VczReader.variant_chunks``.

    Walks ``stream_plan`` in order; for each stream chunk derives the
    set of unique Zarr blocks needed (one per (field, sample-chunk)
    pair) on the fly. Each unique block — identified by its
    ``block_key`` — is submitted to the reader-owned thread pool
    exactly once: blocks shared between consecutive stream chunks
    (variant-only fields with ``multiplier > 1``) are reused, never
    re-submitted. The consumer waits per-stream-chunk on the specific
    futures it needs (no ``as_completed`` loop), so a shared block
    contributes its bytes to ``last_chunk_bytes`` only the first time
    it's read.

    Memory bookkeeping is rolling:

    - ``_live`` maps ``block_key`` to ``(Future, size_estimate)``.
    - ``_in_flight_bytes`` is the running sum of live size_estimates.
    - After each yield, blocks whose ``block_key`` is in the current
      stream chunk's records but not in the next stream chunk's
      records are dropped from ``_live`` (their last reference is
      this chunk). For one field, ``stream_plan`` positions that
      reference variant-block ``K`` form a contiguous range, so a
      single peek at the next stream chunk's records is enough to
      decide eviction.

    The executor is supplied by the caller (typically
    :class:`VczReader`) and lives across stream readers. When iteration
    is abandoned mid-stream (consumer breaks early, generator closed,
    exception propagates), the reader cancels its own pending futures;
    the executor itself outlives the reader.

    ``readahead_bytes=0`` pins depth at 1: the consumer's current
    stream chunk plus exactly one prefetched ahead. The reader never
    goes below depth 1 (otherwise the consumer would wait for every
    chunk's I/O on the request thread).
    """

    def __init__(
        self,
        root,
        stream_plan: list[utils.ChunkRead],
        sample_chunk_plan: "samples_mod.SampleChunkPlan",
        output_columns: np.ndarray | None,
        read_fields,
        *,
        readahead_bytes: int,
        executor: cf.ThreadPoolExecutor,
        stream_chunk_size: int,
    ):
        self.root = root
        self._stream_plan = stream_plan
        self._sample_chunk_plan = sample_chunk_plan
        self._output_columns = output_columns
        self._executor = executor
        self._readahead_bytes = readahead_bytes
        self._stream_chunk_size = stream_chunk_size
        self._field_specs = _make_field_specs(root, read_fields, stream_chunk_size)
        # block_key -> (Future, size_estimate).
        self._live: dict[tuple, tuple[cf.Future, int]] = {}
        self._in_flight_bytes = 0
        self._submit_cursor = 0
        # Memo of derived block records per stream-chunk index. Populated
        # by _submit_more, popped by the yield loop after eviction is
        # decided. Bounded by depth (submit_cursor - yield_idx + 1).
        self._derived: dict[int, list[_BlockRecord]] = {}
        # Block-keys whose actual bytes have been counted into a
        # last_chunk_bytes report. Used to attribute per-chunk new-IO
        # bytes correctly when a block is shared across stream chunks.
        # Evicted alongside ``_live`` so growth is bounded by depth.
        self._billed: set[tuple] = set()
        # Wall-clock seconds spent on the most recent chunk's block reads;
        # consumed by VczReader.variant_chunks to attribute per-chunk time
        # into "read" vs. "assemble".
        self.last_chunk_read_seconds: float | None = None
        # Sum of utils.array_memory_bytes() over the most recent chunk's
        # *newly resolved* blocks; consumed by VczReader.variant_chunks
        # to accumulate retrieval-side throughput stats.
        self.last_chunk_bytes: int | None = None
        # Peak ``len(_live)`` observed across the iteration; the
        # consumer reads it after iteration to assess how effective
        # the readahead window was at staying ahead of demand.
        self.max_in_flight = 0
        logger.debug(
            f"StreamReader init: {len(read_fields)} read_fields, "
            f"stream_chunk_size={stream_chunk_size}, "
            f"stream_plan_len={len(stream_plan)}, "
            f"readahead_bytes={_fmt_bytes(readahead_bytes)}"
        )

    def _derive_blocks(self, sc: utils.ChunkRead) -> list[_BlockRecord]:
        """Build the per-(field, sample-chunk) block records for one
        stream chunk. The derivation is field-spec driven and stateless:
        the same ``sc`` produces the same list every time, modulo
        identity of the returned objects."""
        records: list[_BlockRecord] = []
        stream_chunk_size = self._stream_chunk_size
        sc_index = int(sc.index)
        for spec in self._field_specs:
            multiplier = spec.multiplier
            variant_block_idx = sc_index // multiplier
            intra_start = (sc_index % multiplier) * stream_chunk_size
            intra_slice = slice(intra_start, intra_start + stream_chunk_size)
            if spec.is_call:
                for cr in self._sample_chunk_plan.chunk_reads:
                    dest_key = (spec.field, cr.index)
                    block_index = (
                        variant_block_idx,
                        cr.index,
                    ) + spec.block_index_suffix
                    records.append(
                        _BlockRecord(
                            block_key=(dest_key, variant_block_idx),
                            dest_key=dest_key,
                            arr=spec.arr,
                            block_index=block_index,
                            intra_slice=intra_slice,
                            size_estimate=spec.block_size_estimate,
                        )
                    )
            else:
                dest_key = (spec.field,)
                block_index = (variant_block_idx,) + spec.block_index_suffix
                records.append(
                    _BlockRecord(
                        block_key=(dest_key, variant_block_idx),
                        dest_key=dest_key,
                        arr=spec.arr,
                        block_index=block_index,
                        intra_slice=intra_slice,
                        size_estimate=spec.block_size_estimate,
                    )
                )
        return records

    def _get_derived(self, sc_idx: int) -> list[_BlockRecord]:
        cached = self._derived.get(sc_idx)
        if cached is not None:
            return cached
        records = self._derive_blocks(self._stream_plan[sc_idx])
        self._derived[sc_idx] = records
        return records

    def _submit_more(self, must_advance_through: int) -> None:
        """Submit unique blocks for upcoming stream chunks under the
        byte budget. Always submits at least through
        ``must_advance_through`` (the next stream-chunk index the
        consumer will demand), even if that overshoots the budget — the
        consumer has no later opportunity to receive those blocks, and
        refusing here would leave the iterator looking up an
        unsubmitted block_key. The budget is therefore a soft cap: a
        large variant-only block that stays live across stream chunks
        can force a single-chunk overshoot when its bytes plus the next
        chunk's new-block bytes exceed ``readahead_bytes``."""
        while self._submit_cursor < len(self._stream_plan):
            sc_idx = self._submit_cursor
            records = self._get_derived(sc_idx)
            new_records = [r for r in records if r.block_key not in self._live]
            new_bytes = sum(r.size_estimate for r in new_records)
            over_budget = (
                new_bytes > 0
                and self._in_flight_bytes + new_bytes > self._readahead_bytes
            )
            if sc_idx > must_advance_through and over_budget:
                return
            for rec in new_records:
                fut = self._executor.submit(_read_block, rec.arr, rec.block_index)
                self._live[rec.block_key] = (fut, rec.size_estimate)
                self._in_flight_bytes += rec.size_estimate
            if len(self._live) > self.max_in_flight:
                self.max_in_flight = len(self._live)
            self._submit_cursor += 1
            logger.log(
                TRACE,
                f"submitted stream chunk {sc_idx}: "
                f"{len(new_records)} new blocks "
                f"({len(records) - len(new_records)} already live)",
            )

    def __iter__(self):
        try:
            self._submit_more(must_advance_through=0)
            for sc_idx in range(len(self._stream_plan)):
                records = self._get_derived(sc_idx)
                read_start = time.perf_counter()
                blocks: dict[tuple, np.ndarray] = {}
                chunk_bytes = 0
                for rec in records:
                    fut, _ = self._live[rec.block_key]
                    raw = fut.result()
                    if rec.block_key not in self._billed:
                        self._billed.add(rec.block_key)
                        chunk_bytes += utils.array_memory_bytes(raw)
                    blocks[rec.dest_key] = raw[rec.intra_slice]
                read_seconds = time.perf_counter() - read_start
                self.last_chunk_read_seconds = read_seconds
                self.last_chunk_bytes = chunk_bytes
                stream_chunk = self._stream_plan[sc_idx]
                logger.debug(
                    f"stream chunk {stream_chunk.index} read complete in "
                    f"{read_seconds:.2f}s ({len(blocks)} blocks, "
                    f"{_fmt_bytes(chunk_bytes)})"
                )
                yield CachedLogicalVariantsChunk(
                    self.root,
                    stream_chunk,
                    sample_chunk_plan=self._sample_chunk_plan,
                    output_columns=self._output_columns,
                    blocks=blocks,
                    stream_chunk_size=self._stream_chunk_size,
                )
                current_keys = {rec.block_key for rec in records}
                if sc_idx + 1 < len(self._stream_plan):
                    next_keys = {rec.block_key for rec in self._get_derived(sc_idx + 1)}
                else:
                    next_keys = set()
                self._derived.pop(sc_idx, None)
                for key in current_keys - next_keys:
                    _, size = self._live.pop(key)
                    self._in_flight_bytes -= size
                    self._billed.discard(key)
                self._submit_more(must_advance_through=sc_idx + 1)
        finally:
            cancelled = 0
            for fut, _ in self._live.values():
                if fut.cancel():
                    cancelled += 1
            if cancelled > 0:
                logger.debug(f"cancelled {cancelled} pending futures")


class CachedLogicalVariantsChunk:
    """View assembler over prefetched, intra-sliced blocks for one
    stream chunk visit.

    Constructed by :class:`StreamReader` once its block reads have
    completed and been intra-sliced to the stream chunk's variant
    rows; performs no I/O itself.

    Constructor inputs:

    - ``blocks`` — ``{dest_key: ndarray}`` keyed by ``(field,)`` for
      variants-axis non-``call_*`` reads and ``(field, sci)`` for one
      ``call_*`` sample-chunk read. Each ndarray has already been
      sliced to the stream chunk's variant rows; the caller does not
      apply an additional intra-slice.
    - ``sample_chunk_plan`` — the sample chunks the prefetch covers
      for every ``call_*`` field. In subset-mode this is the subset
      plan; in view-mode it is the non-null-samples plan. An empty
      plan (no ``chunk_reads``) is valid and produces zero-sample-
      column arrays without any prefetched ``call_*`` blocks.
      Non-``call_*`` fields ignore it.
    - ``output_columns`` — indices into the read-plan axis that
      produce the subset axis. ``None`` when the read plan is already
      the subset axis (subset-mode); :meth:`output_view` returns the
      assembled read untouched.
    - ``stream_chunk_size`` — used by :meth:`_empty_call_array` when
      the variant chunk's selection is ``None`` (raw block).

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
        stream_chunk_size: int,
    ):
        self.root = root
        self.variant_chunk = variant_chunk
        self._sample_chunk_plan = sample_chunk_plan
        self._output_columns = output_columns
        self._blocks = blocks
        self._stream_chunk_size = stream_chunk_size
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
        if field.startswith("call_"):
            result = self._assemble_call(field)
        else:
            result = self._slice_variants(self._blocks[(field,)])
        #
        # Workaround for numpy bug: https://github.com/numpy/numpy/issues/31415
        #
        # Under proportional chunking a single Zarr block can back many
        # stream chunks; basic-indexing slices return a view that shares
        # its numpy 2 StringDType arena with the block buffer in
        # _live. The consumer thread's StringDType operations on that
        # view then race the prefetch worker's parallel operations on a
        # sibling view of the same block, causing heap corruption. Force
        # a copy so each chunk's StringDType array has its own arena.
        if result.dtype.kind == "T" and not result.flags.owndata:
            result = result.copy()
        return result

    def _slice_variants(self, data):
        sel = self.variant_chunk.selection
        if sel is None:
            return data
        return data[sel]

    def _assemble_call(self, field: str) -> np.ndarray:
        plan = self._sample_chunk_plan
        if len(plan.chunk_reads) == 0:
            return self._empty_call_array(field)
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

    def _empty_call_array(self, field: str) -> np.ndarray:
        """Zero-sample-column array for a call_* field, without I/O.
        Used when the sample chunk plan is empty (e.g. ``set_samples([])``)."""
        arr = self.root[field]
        sel = self.variant_chunk.selection
        if sel is None:
            chunk_start = self.variant_chunk.index * self._stream_chunk_size
            n_variants = min(self._stream_chunk_size, int(arr.shape[0]) - chunk_start)
        else:
            n_variants = int(self.variant_chunk.num_selected)
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

    Owns the zarr root and provides metadata properties and variant
    iteration at both chunk and row granularity, via
    :meth:`variant_chunks` and :meth:`variants`.

    The reader starts with no configured selection — every real sample
    and every variant is iterated by default. Call :meth:`set_samples`
    to restrict the sample selection **before** iterating.

    Use as a context manager (``with VczReader(root) as reader:``) so
    the reader's resources are released deterministically on exit.

    Virtual fields
    --------------
    Besides stored arrays, these computed variant-axis fields may be
    named in :meth:`variant_chunks` ``fields`` or in a filter expression.
    Availability depends on the store; see :attr:`virtual_field_names`.
    They are addressable by name but never auto-emitted when ``fields``
    is ``None``.

    ``variant_index``     Global (store-wide) 0-based index of the variant.
    ``variant_AC``        Allele count in genotypes.
    ``variant_AN``        Total number of alleles in called genotypes.
    ``variant_AF``        Allele frequency.
    ``variant_NS``        Number of samples with data.
    ``variant_N_ALT``     Number of non-empty ALT alleles.
    ``variant_N_MISSING`` Number of samples with all-missing genotypes.
    ``variant_F_MISSING`` Fraction of samples with all-missing genotypes.

    Parameters
    ----------
    root
        An already-opened :class:`zarr.Group` pointing at the VCZ
        dataset. Use :func:`vcztools.open_zarr` to open a path
        (local, remote, or zip) with the desired backend before
        constructing the reader.
    readahead_workers
        Worker count for the readahead thread pool. ``None``
        (default) uses ``32``. The pool is created at construction;
        this parameter has no post-init knob.
    readahead_bytes
        Cap, in bytes, on the cross-chunk readahead window. ``None``
        (default) uses 256 MiB. ``0`` pins pipeline depth at 1 (one
        chunk prefetched ahead of the consumer); the pipeline cannot
        go lower.
    """

    def __init__(
        self,
        root,
        *,
        readahead_workers: int | None = None,
        readahead_bytes: int | None = None,
    ):
        self.root = root
        utils.validate_variants_axis_chunking(
            root, utils.compute_min_variants_chunk_size(root)
        )
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
        self._bcftools_semantics = False
        self._full_sample_filter = False
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
        value = _freeze(self.root[name][:])
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
        self._samples_selection = _freeze(np.flatnonzero(raw_sample_ids != ""))
        self._sample_ids = _freeze(raw_sample_ids[self._samples_selection])
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
        """Selected sample IDs as a numpy array, in selection order."""
        self._resolve_samples_if_needed()
        return self._sample_ids

    @property
    def sample_chunk_plan(self):
        self._resolve_samples_if_needed()
        return self._sample_chunk_plan

    @property
    def variant_chunk_plan(self):
        if self._variant_chunk_plan is None:
            chunk_size = self.variants_chunk_size
            num_variants = self.num_variants
            self._variant_chunk_plan = utils.ChunkRead.simple_plan(
                num_variants, chunk_size
            )
            logger.debug(
                f"Built default variant_chunk_plan: "
                f"{len(self._variant_chunk_plan)} chunks"
            )
        return self._variant_chunk_plan

    def set_samples(self, samples) -> None:
        """Configure the sample selection.

        Accepts a list or ndarray of integer indexes into the VCZ
        ``sample_id`` array, in the order the caller wants. An empty
        sequence is valid and means "no samples in output". Out-of-range
        indexes raise ``ValueError``; duplicates are permitted. Must be
        called before iterating; raises ``RuntimeError`` if the selection
        is already configured.

        Selecting a proper subset makes sample-dependent virtual fields
        (``AC``/``AN``/``AF``/``NS`` …) recompute to reflect the subset
        on the next iteration. See :meth:`variant_chunks`.
        """
        if self._samples_selection is not None:
            raise RuntimeError("samples already configured")
        samples_selection = self._normalize_sample_indexes(samples, label="sample")
        self._samples_selection = _freeze(samples_selection)
        self._sample_ids = _freeze(self.raw_sample_ids[samples_selection])
        self._sample_chunk_plan = samples_mod.build_chunk_plan(
            samples_selection,
            samples_chunk_size=self.samples_chunk_size,
        )
        logger.debug(
            f"set_samples: {samples_selection.size} samples, "
            f"{len(self._sample_chunk_plan.chunk_reads)} sample chunks"
        )

    def variant_counts_per_chunk(self) -> np.ndarray:
        """Number of variants contributed by each entry in
        :attr:`variant_chunk_plan`.

        Returns a 1-D ``int64`` array of length
        ``len(variant_chunk_plan)``. Reads the count stored on each
        :class:`~vcztools.utils.ChunkRead` — no Zarr access. Useful
        for consumers (e.g. PLINK BED encoders) that need to size
        per-chunk output without reading any data.
        """
        return _freeze(
            np.array(
                [cr.num_selected for cr in self.variant_chunk_plan], dtype=np.int64
            )
        )

    def set_variants(self, variants) -> None:
        """Configure the variant selection.

        Accepts a sorted 1-D array of global variant indexes, which is
        bucketed into a chunk plan internally. A pre-built chunk plan
        (a ``list`` of ``ChunkRead``) may also be passed for callers
        that already have one, e.g. from a region/target query.

        May be called multiple times; each call replaces the prior
        selection. A :meth:`variant_chunks` generator already iterating
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
                min_chunk=self.variants_chunk_size,
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
        :class:`~vcztools.VariantFilter` protocol (e.g. a
        :class:`~vcztools.BcftoolsFilter`), or ``None`` to clear a
        previously-set filter. By default a sample-scope filter
        evaluates over the user's sample selection (``bcftools query``
        FMT-scope post-subset semantics).

        May be called multiple times; each call replaces the prior
        filter. A :meth:`variant_chunks` generator already iterating is
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

        Iterates :meth:`variant_chunks` collecting the global indexes
        of surviving variants, then replaces ``(variant_filter,
        variant_chunk_plan)`` with a chunk plan over those variants.
        No-op if no filter is configured.

        Only variant-scope filters are supported. Sample-scope filters
        require the per-cell mask emitted during iteration; resolve
        them by iterating :meth:`variant_chunks` directly. Raises
        ``ValueError`` on a sample-scope filter.
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
        surviving = []
        for chunk_data in self.variant_chunks(fields=["variant_index"]):
            surviving.append(chunk_data["variant_index"])
        if len(surviving) == 0:
            indexes = np.empty(0, dtype=np.int64)
        else:
            indexes = np.concatenate(surviving)
        plan = regions_mod.chunk_plan_from_indexes(
            indexes, min_chunk=self.variants_chunk_size
        )
        logger.info(
            f"materialise_variant_filter: {int(indexes.size)} variants survive "
            f"({len(plan)} chunks)"
        )
        self.set_variant_filter(None)
        self.set_variants(plan)

    def set_bcftools_semantics(self, *, full_sample_filter: bool = False) -> None:
        """Opt into bcftools filter / INFO evaluation semantics.

        By default the reader is *subset-first*: once :meth:`set_samples`
        selects a sample subset, variant filters evaluate over that
        subset and a virtual field (``AC``/``AN``/``AF``/``NS`` …)
        resolves the same way wherever it is used, so a filter on the
        field and its emitted value always agree.

        bcftools instead evaluates ``-i/-e`` filters against the file's
        stored ``INFO`` values (a virtual field is computed for a filter
        only when there is no stored counterpart), and recomputes
        ``INFO`` output only as named by :meth:`variant_chunks`'
        ``force_recompute``. Both ``bcftools view`` and ``query`` filter
        on the stored ``INFO``; they differ only in the sample axis a
        sample-scope (FORMAT) filter sees, set by ``full_sample_filter``:
        ``False`` (default, ``bcftools query``) uses the user's subset;
        ``True`` (``bcftools view``) evaluates over the full (pre-subset)
        non-null sample set, then subsets for output.

        One-shot: a second call raises ``RuntimeError``.
        """
        if self._bcftools_semantics:
            raise RuntimeError("bcftools semantics already configured")
        self._bcftools_semantics = True
        self._full_sample_filter = full_sample_filter
        logger.debug(f"set_bcftools_semantics: full_sample_filter={full_sample_filter}")

    def _normalize_sample_indexes(self, value, *, label: str) -> np.ndarray:
        """Validate and convert a sample-index input to a 1-D int64 ndarray.

        Checks input type (via :func:`_validate_samples_input`) and that
        every index is in ``[0, num_samples)``. ``label`` is interpolated
        into error messages.
        """
        _validate_samples_input(value)
        # ``np.array`` (not ``asarray``): always produce an owned copy so
        # freezing the result for caching never locks the caller's input.
        arr = np.array(value, dtype=np.int64)
        if arr.size > 0:
            lo = arr.min()
            hi = arr.max()
            raw_size = self.raw_sample_ids.size
            if lo < 0 or hi >= raw_size:
                raise ValueError(
                    f"{label} index out of range: must be in [0, {raw_size})"
                )
        return arr

    @property
    def filter_sample_chunk_plan(self) -> samples_mod.SampleChunkPlan:
        """Chunk plan for reading the filter sample axis. This matches
        :attr:`sample_chunk_plan` unless :meth:`set_bcftools_semantics`
        enabled ``full_sample_filter``, in which case it is built from
        the full (pre-subset) non-null sample set."""
        if not self._full_sample_filter:
            return self.sample_chunk_plan
        return samples_mod.build_chunk_plan(
            self.non_null_sample_indices,
            samples_chunk_size=self.samples_chunk_size,
        )

    @functools.cached_property
    def contig_ids(self):
        """Contig IDs as a numpy StringDType array."""
        return _freeze(
            np.array(self.root["contig_id"][:], dtype=np.dtypes.StringDType())
        )

    @functools.cached_property
    def filter_ids(self):
        """Filter IDs as raw zarr strings."""
        return _freeze(self.root["filter_id"][:])

    @functools.cached_property
    def contigs(self):
        """Contig IDs as fixed-length bytes (for VcfEncoder)."""
        return _freeze(_as_fixed_length_string(self.root["contig_id"][:]))

    @functools.cached_property
    def filters(self):
        """Filter IDs as fixed-length bytes (for VcfEncoder).
        Returns a single ``b"PASS"`` entry when the store has no
        ``filter_id`` array."""
        return _freeze(_get_filter_ids(self.root))

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
        """Minimum chunk size on the variants axis: the iteration unit.

        ``call_*`` fields define this minimum (they are the memory-budget
        driver). Variant-only fields may have a chunk size that is a
        positive integer multiple of it. When no ``call_*`` field is
        present (e.g. a drop-genotypes-only store), falls back to the
        minimum chunk size across variant-axis fields.
        """
        return utils.compute_min_variants_chunk_size(self.root)

    @functools.cached_property
    def samples_chunk_size(self) -> int:
        """Chunk size along the samples axis."""
        return int(self.root["sample_id"].chunks[0])

    @functools.cached_property
    def raw_sample_ids(self) -> np.ndarray:
        """Full ``sample_id`` array from the store, including any
        null-string entries. For the post-subset order used when
        encoding rows, see :attr:`sample_ids`."""
        return _freeze(self.root["sample_id"][:])

    @functools.cached_property
    def non_null_sample_indices(self) -> np.ndarray:
        """Global indices of non-null samples in ``sample_id``. Sorted;
        empty if every entry is null."""
        return _freeze(np.flatnonzero(self.raw_sample_ids != ""))

    @functools.cached_property
    def contig_lengths(self) -> np.ndarray | None:
        """``contig_length`` array, or ``None`` if absent."""
        if "contig_length" not in self.root:
            return None
        return _freeze(self.root["contig_length"][:])

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
        return _freeze(self.root["region_index"][:])

    @functools.cached_property
    def filter_descriptions(self) -> np.ndarray | None:
        """Per-filter descriptions, or ``None`` if absent."""
        if "filter_description" not in self.root:
            return None
        return _freeze(self.root["filter_description"][:])

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
        """Set of real (Zarr-backed) field names present in the store.

        Virtual fields (e.g. ``variant_AC`` on a store without a stored
        ``variant_AC`` array) are not included here even though
        :meth:`variant_chunks` can yield them — they are addressable by
        name but not auto-discovered by the default-emit path. See
        :attr:`virtual_field_names` for the available virtual set.
        """
        return frozenset(self.root)

    @functools.cached_property
    def _virtual_fields(self) -> dict:
        """The subset of :data:`virtual_fields.REGISTRY` whose deps
        are satisfied by this store (with degenerate fallbacks chosen
        where the primary's deps are absent). Resolved lazily so the
        constructor stays cheap and store-touching tests can introspect
        the reader without paying for a metadata walk."""
        return virtual_fields_mod.resolve_for_root(self.root)

    @functools.cached_property
    def virtual_field_names(self) -> frozenset[str]:
        """Names of virtual fields whose dependencies are satisfied by
        the current store.

        A name appears here when its primary form's dependencies are
        present, or — for ``variant_N_MISSING`` / ``variant_F_MISSING``
        on annotations-only stores — when the degenerate fallback's
        dependencies are present. These names can be requested in the
        ``fields`` argument of :meth:`variant_chunks`.
        """
        return frozenset(self._virtual_fields)

    @functools.cached_property
    def _field_info_cache(self) -> dict[str, FieldInfo]:
        return {}

    def get_field_info(self, name: str) -> FieldInfo:
        """Return a :class:`FieldInfo` snapshot for the named field.
        Reads Zarr metadata on first access, then memoizes per-field.
        Raises ``KeyError`` if the field is absent.

        For a stored field, returns the real on-disk metadata.
        For a virtual field with no stored counterpart, synthesises
        :class:`FieldInfo` from the registry entry. For a virtual field
        that *also* exists as a stored array, the stored array wins
        (its ``attrs.description`` is preserved); the registry only
        substitutes its description if the stored array has none.
        """
        cache = self._field_info_cache
        if name in cache:
            return cache[name]
        if name in self.root:
            arr = self.root[name]
            attrs = dict(arr.attrs)
            vf = self._virtual_fields.get(name)
            if vf is not None and "description" not in attrs:
                attrs["description"] = vf.description
            cache[name] = FieldInfo(
                name=name,
                dtype=arr.dtype,
                shape=tuple(arr.shape),
                dims=tuple(utils.array_dims(arr)),
                attrs=attrs,
            )
            return cache[name]
        if name in self._virtual_fields:
            vf = self._virtual_fields[name]
            source = self.root[vf.shape_from]
            num_variants = source.shape[0]
            if len(vf.dims) == 1:
                shape = (num_variants,)
            else:
                shape = (num_variants, source.shape[1] - 1)
            cache[name] = FieldInfo(
                name=name,
                dtype=vf.dtype,
                shape=shape,
                dims=vf.dims,
                attrs={"description": vf.description},
            )
            return cache[name]
        raise KeyError(name)

    def variant_chunks(
        self,
        *,
        fields: list[str] | None = None,
        start: int = 0,
        force_recompute=False,
    ):
        """Yield ``dict[str, np.ndarray]`` per variant chunk that passes
        the current sample and variant selection.

        ``fields`` names the fields to read; ``None`` (default) emits
        every stored field. A field's value spans the variants axis (and,
        for FORMAT fields, the selected samples axis).

        ``start`` is an offset into the sequence of variant chunks:
        iteration begins at the ``start``-th chunk. ``start=0`` (default)
        iterates every chunk; a ``start`` past the last chunk yields
        nothing; negative ``start`` raises ``ValueError``.

        ``force_recompute`` controls recomputation of virtual fields (see
        :attr:`virtual_field_names`) that have a same-named stored array.
        ``True`` recomputes every requested virtual field; an iterable
        scopes it to the named fields; ``False`` (default) forces none.
        A virtual field with no stored counterpart is always computed.
        When a sample subset is active, a sample-dependent virtual field
        (``AC``, ``AN``, ``AF``, ``NS``, ``N_MISSING``, ``F_MISSING``) is
        recomputed to reflect the subset even with
        ``force_recompute=False``, so a filter on the field and its
        emitted value always agree.

        The returned iterator overlaps the consumer's per-chunk work with
        the assembly of the next chunk; ``close()`` it promptly to
        release the in-flight chunk. Argument validation is eager:
        ``start < 0`` and ``fields == []`` are reported on the call
        itself rather than on the first ``next()``.

        Virtual fields (see :attr:`virtual_field_names`) may be named in
        ``fields`` alongside stored arrays. ``"variant_index"`` is one:
        its per-chunk array holds the global (store-wide) ``int64`` index
        of each surviving variant. Virtual fields are not auto-emitted
        when ``fields`` is ``None``.
        """
        if start < 0:
            raise ValueError(f"start must be >= 0 (got {start})")
        if fields is not None and len(fields) == 0:
            return iter(())

        return utils.PrefetchIterator(
            self._variant_chunks_gen(
                fields=fields, start=start, force_recompute=force_recompute
            )
        )

    def _variant_chunks_gen(
        self,
        *,
        fields: list[str] | None = None,
        start: int = 0,
        force_recompute=False,
    ):
        """Inner generator backing :meth:`variant_chunks`. The public
        entry point validates arguments eagerly and wraps this
        generator in a one-deep prefetch iterator; tests that need
        the raw single-threaded behaviour (e.g. for deterministic
        in-thread state inspection) can drive this directly."""
        # Snapshot the filter so a mid-iteration set_variant_filter
        # can't change behaviour for this generator.
        variant_filter = self.variant_filter
        query_fields = self._resolve_query_fields(fields)
        filter_fields = frozenset(
            variant_filter.referenced_fields if variant_filter is not None else ()
        )
        if not self._full_sample_filter:
            # Filter axis IS the subset axis. Covers non-empty subsets,
            # the no-subset default, and ``--drop-genotypes`` (empty
            # subset collapses to an empty plan — no reads, zero-column
            # call_* output via
            # CachedLogicalVariantsChunk._empty_call_array).
            sample_chunk_plan = self.sample_chunk_plan
            output_columns = None
        else:
            # "bcftools view" mode: filter axis is the full (pre-subset)
            # non-null sample set. Read on that axis; remap columns to
            # produce the subset output.
            sample_chunk_plan = self.filter_sample_chunk_plan
            output_columns = np.searchsorted(
                self.non_null_sample_indices, self.samples_selection
            )

        # Split referenced fields into static (read once on the reader)
        # and dynamic (prefetched per stream chunk).
        #
        # Virtual fields go through the registry and never enter the
        # Zarr-backed split when computed; they are produced per chunk
        # from their dependencies (or, for ``variant_index``, from the
        # chunk's plan state). How a field resolves (stored array vs
        # registry compute) depends on the mode.
        #
        # Default (subset-first) mode: one rule for filter and output,
        # so a filter on a field and its emitted value always agree. A
        # virtual field is computed when it has no stored counterpart,
        # when ``force_recompute`` names it, or when a sample subset is
        # active and the field is sample-dependent (its value reflects
        # the genotypes, so the stored full-file value would be stale).
        # Otherwise the stored array is used.
        #
        # bcftools mode: the two paths diverge, matching bcftools. The
        # filter always reads the stored array when it exists (both
        # ``view`` and ``query`` evaluate ``-i INFO/AC>10`` against the
        # file's INFO column, unaffected by recompute); output reads the
        # stored array unless ``force_recompute`` names the field.
        force_set = self._resolve_force_recompute(force_recompute)
        subset_active = self.samples_selection.size > 0 and not np.array_equal(
            self.samples_selection, self.non_null_sample_indices
        )

        def _sample_dependent(name):
            return "call_genotype" in self._virtual_fields[name].deps

        def _is_virtual_default(name):
            if name not in self._virtual_fields:
                return False
            if name not in self.root:
                return True
            if name in force_set:
                return True
            return subset_active and _sample_dependent(name)

        def _is_virtual_bcftools_output(name):
            if name not in self._virtual_fields:
                return False
            if name not in self.root:
                return True
            return name in force_set

        def _is_virtual_bcftools_filter(name):
            return name in self._virtual_fields and name not in self.root

        if self._bcftools_semantics:
            is_virtual_output = _is_virtual_bcftools_output
            is_virtual_filter = _is_virtual_bcftools_filter
        else:
            is_virtual_output = _is_virtual_default
            is_virtual_filter = _is_virtual_default

        virtual_query_fields = frozenset(
            f for f in query_fields if is_virtual_output(f)
        )
        virtual_filter_fields = frozenset(
            f for f in filter_fields if is_virtual_filter(f)
        )

        def _expand_virtual_deps(names, predicate):
            out = []
            for n in names:
                if predicate(n):
                    out.extend(self._virtual_fields[n].deps)
                else:
                    out.append(n)
            return out

        real_query_fields = [f for f in query_fields if f not in virtual_query_fields]
        real_filter_fields = [
            f for f in filter_fields if f not in virtual_filter_fields
        ]
        referenced = list(
            dict.fromkeys(
                [
                    *real_filter_fields,
                    *_expand_virtual_deps(
                        list(virtual_filter_fields), is_virtual_filter
                    ),
                    *real_query_fields,
                    *_expand_virtual_deps(
                        list(virtual_query_fields), is_virtual_output
                    ),
                ]
            )
        )
        referenced_static_fields = {
            name: self._load_static_field(name)
            for name in referenced
            if not utils.has_variants_axis(self.root[name])
        }
        read_fields = [
            name for name in referenced if name not in referenced_static_fields
        ]
        readahead_bytes = (
            self.readahead_bytes
            if self.readahead_bytes is not None
            else DEFAULT_READAHEAD_BYTES
        )

        min_chunk = self.variants_chunk_size
        stream_chunk_size = utils.compute_stream_chunk_size(
            self.root, read_fields, min_chunk
        )
        stream_plan = utils.rebucket_to_stream_plan(
            self.variant_chunk_plan, min_chunk, stream_chunk_size
        )
        if start > 0:
            stream_plan = stream_plan[start:]

        logger.info(
            f"variant_chunks: starting iteration "
            f"({len(query_fields)} query fields, {len(filter_fields)} filter fields, "
            f"{len(referenced_static_fields)} static fields, "
            f"{len(read_fields)} read fields, "
            f"min_chunk={min_chunk}, stream_chunk_size={stream_chunk_size}, "
            f"{len(stream_plan)} stream chunks, "
            f"{len(sample_chunk_plan.chunk_reads)} sample chunks, "
            f"readahead_bytes={_fmt_bytes(readahead_bytes)}, "
            f"workers={self._readahead_workers}); "
            f"query_fields={list(query_fields)}, "
            f"read_fields={read_fields}"
        )

        pipeline = StreamReader(
            self.root,
            stream_plan,
            sample_chunk_plan,
            output_columns,
            read_fields,
            readahead_bytes=readahead_bytes,
            executor=self._executor,
            stream_chunk_size=stream_chunk_size,
        )
        chunks_visited = 0
        chunks_yielded = 0
        variants_yielded = 0
        bytes_yielded = 0
        # Per-iteration time accounting. consumer_wait isolates the gap
        # between yielding chunk N and the consumer pulling chunk N+1
        # (minus the producer's own read wait for N+1), exposing the
        # downstream encoder/writer cost which the iterator otherwise
        # can't see.
        producer_assemble_total = 0.0
        producer_read_total = 0.0
        consumer_wait_total = 0.0
        last_yield_t: float | None = None
        iter_start = time.perf_counter()
        try:
            for chunk in pipeline:
                chunks_visited += 1
                chunk_start = time.perf_counter()
                read_seconds = pipeline.last_chunk_read_seconds or 0.0
                producer_read_total += read_seconds
                if last_yield_t is not None:
                    gap = chunk_start - last_yield_t
                    consumer_wait_total += max(0.0, gap - read_seconds)
                bytes_yielded += pipeline.last_chunk_bytes or 0
                # variants_selection: 1-D bool over the chunk's variant axis, or
                # None meaning "no filter, keep every variant".
                # sample_filter_pass: 2-D bool over (surviving variants, output
                # samples) for sample-scope filters only; published so query.py
                # can emit only matching samples in FORMAT-loop queries.
                variants_selection = None
                sample_filter_pass = None
                # Per-chunk cache for virtual-field outputs. Compute
                # functions may publish sibling values into it (e.g.
                # the AC kernel pass also yields AN) so a later request
                # for the sibling short-circuits. Virtual fields are
                # computed on the output (subset) sample axis. When the
                # filter axis is the subset axis output_view is
                # filter_view, so a filter and the INFO column see the
                # same value; with full_sample_filter the non-virtual
                # filter fields read the full pre-subset axis via
                # filter_view.
                virtual_cache: dict = {}
                chunk_context = virtual_fields_mod.ChunkContext(
                    chunk.variant_chunk, stream_chunk_size
                )

                def _virtual_value(name, view_fn, cache, context):
                    if name in cache:
                        return cache[name]
                    vf = self._virtual_fields[name]
                    deps = {dep: view_fn(dep) for dep in vf.deps}
                    value = vf.compute(deps, cache, context)
                    cache[name] = value
                    return value

                if variant_filter is not None:
                    filter_data = {}
                    for f in filter_fields:
                        if f in referenced_static_fields:
                            filter_data[f] = referenced_static_fields[f]
                        elif f in virtual_filter_fields:
                            filter_data[f] = _virtual_value(
                                f, chunk.output_view, virtual_cache, chunk_context
                            )
                        else:
                            filter_data[f] = chunk.filter_view(f)
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
                    if field in virtual_query_fields:
                        value = _virtual_value(
                            field, chunk.output_view, virtual_cache, chunk_context
                        )
                    else:
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
                producer_assemble_total += assemble_seconds
                logger.debug(
                    f"chunk {chunk.variant_chunk.index}: yielded {chunk_variants} "
                    f"variants in {total_seconds:.2f}s "
                    f"(read {read_seconds:.2f}s, assemble {assemble_seconds:.2f}s)"
                )
                last_yield_t = time.perf_counter()
                # Lock the dict's arrays read-only at the chokepoint so a
                # consumer cannot mutate a view into a shared raw block,
                # a cached static field, or a sibling StringDType arena.
                for value in chunk_data.values():
                    value.flags.writeable = False
                yield chunk_data
        finally:
            elapsed = time.perf_counter() - iter_start
            mib = bytes_yielded / (1024 * 1024)
            rate = mib / elapsed if elapsed > 0 else 0.0
            logger.info(
                f"variant_chunks: iteration done in {elapsed:.2f}s "
                f"({chunks_visited} chunks visited, {chunks_yielded} yielded, "
                f"{variants_yielded} variants, "
                f"{mib:.1f} MiB retrieved, {rate:.1f} MiB/s, "
                f"max readahead depth {pipeline.max_in_flight}); "
                f"producer_assemble={producer_assemble_total:.2f}s, "
                f"producer_read_wait={producer_read_total:.2f}s, "
                f"consumer_wait={consumer_wait_total:.2f}s"
            )

    def _resolve_query_fields(self, fields):
        if fields is not None:
            return list(fields)
        return [
            key
            for key in self.field_names
            if key.startswith("variant_") or key.startswith("call_")
        ]

    def _resolve_force_recompute(self, value) -> frozenset[str]:
        """Normalise the :meth:`variant_chunks` ``force_recompute``
        argument into the set of virtual field names to recompute even
        when a stored counterpart exists.

        ``True`` expands to every available virtual name; ``False`` to
        the empty set; any iterable is taken verbatim (caller-supplied
        names need not be among the available virtuals — unknown names
        have no effect, since the virtual check is a separate gate).
        """
        if value is True:
            return self.virtual_field_names
        if value is False:
            return frozenset()
        return frozenset(value)

    def variants(
        self,
        *,
        fields: list[str] | None = None,
        force_recompute=False,
    ):
        """Yield dict[str, scalar/1d-array] per variant row.

        ``force_recompute`` is forwarded to :meth:`variant_chunks`."""
        chunks = self.variant_chunks(fields=fields, force_recompute=force_recompute)
        for chunk_data in chunks:
            first_field = next(iter(chunk_data.values()))
            num_variants = len(first_field)
            for i in range(num_variants):
                yield {name: chunk_data[name][i] for name in chunk_data}
