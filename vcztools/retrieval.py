import concurrent.futures as cf
import dataclasses
import functools
import itertools
import logging
import os
import threading
import time
import weakref

import anyio
import anyio.from_thread
import numpy as np

from vcztools import regions as regions_mod
from vcztools import samples as samples_mod
from vcztools import utils, zarr_direct
from vcztools import variant_filter as variant_filter_mod
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


DEFAULT_READAHEAD_BYTES = 256 * 1024 * 1024
# Default cap on concurrent ``store.get`` calls per variant_chunks()
# iteration. Stores are async-native so this is a coroutine count, not
# a thread count.
DEFAULT_IO_CONCURRENCY = 32
# Default size of the per-reader decode thread pool. CPU-bound work
# (zstd, blosc, etc. — all GIL-releasing C) runs in this pool inside
# the reader's anyio portal.
DEFAULT_DECODE_THREADS = os.cpu_count() or 1


async def _read_block_async(
    reader: zarr_direct.BlockReader,
    block_index: tuple,
    io_limiter: anyio.CapacityLimiter,
    decode_limiter: anyio.CapacityLimiter,
) -> np.ndarray:
    """Fetch the chunks covered by ``block_index`` and assemble them
    into a single ndarray, matching the result of ``arr.blocks[idx]``.

    ``block_index`` may mix integer chunk indices with ``slice``
    objects (typically ``slice(None)`` over a non-variants axis).
    Slices are resolved to their concrete chunk-coord ranges via
    :attr:`BlockReader.cdata_shape`, every chunk is fetched
    concurrently inside an :func:`anyio.create_task_group`, and the
    results are assembled with :func:`numpy.block`.
    """
    coord_ranges = []
    for d, idx in enumerate(block_index):
        if isinstance(idx, slice):
            n_chunks = reader.cdata_shape[d]
            coord_ranges.append(list(range(*idx.indices(n_chunks))))
        else:
            coord_ranges.append([idx])
    coords_list = list(itertools.product(*coord_ranges))

    async def fetch_one(coords):
        async with io_limiter:
            return await reader.read_chunk(coords, decode_limiter)

    if len(coords_list) == 1:
        return await fetch_one(coords_list[0])
    fetched: dict[tuple, np.ndarray] = {}
    async with anyio.create_task_group() as tg:
        for coords in coords_list:

            async def one(c=coords):
                fetched[c] = await fetch_one(c)

            tg.start_soon(one)

    def build(axis, prefix):
        if axis == len(coord_ranges):
            return fetched[tuple(prefix)]
        return [build(axis + 1, prefix + [c]) for c in coord_ranges[axis]]

    return np.block(build(0, []))


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


# Query-only pseudo-fields recognised by :meth:`VczReader.variant_chunks`.
# Each is emitted from per-chunk plan state, never from a Zarr array.
_PSEUDO_QUERY_FIELDS = frozenset({"variant_index"})


def _absolute_variant_indexes(
    entry: utils.ChunkRead, variants_chunk_size: int
) -> np.ndarray:
    """Global variant indexes contributed by ``entry``.

    Maps each variant the chunk read selects to its store-wide variant
    index. Used to materialise the ``variant_index`` pseudo-field in
    :meth:`VczReader.variant_chunks`.
    """
    chunk_offset = entry.index * variants_chunk_size
    sel = entry.selection
    if sel is None:
        local = np.arange(entry.num_selected, dtype=np.int64)
    elif isinstance(sel, slice):
        local = np.arange(*sel.indices(variants_chunk_size), dtype=np.int64)
    else:
        local = np.asarray(sel, dtype=np.int64)
    return chunk_offset + local


@dataclasses.dataclass(frozen=True, slots=True)
class BlockReadTemplate:
    """Variant-chunk-independent read pattern for one block.

    ``block_index_suffix`` is the part of the Zarr ``block_index``
    *after* the variant chunk index slot — empty tuple for a 1-D
    variant-axis field, ``(slice(None),) * (ndim - 1)`` for higher-D
    non-call fields, ``(sci, slice(None) * (ndim - 2))`` for ``call_*``.
    """

    key: tuple
    block_reader: zarr_direct.BlockReader
    block_index_suffix: tuple


def create_chunk_read_list(
    root,
    sample_chunk_plan: "samples_mod.SampleChunkPlan",
    fields,
    *,
    get_block_reader,
) -> list[BlockReadTemplate]:
    """Resolve ``fields`` to a list of :class:`BlockReadTemplate`
    once per query, before any variant chunk is visited.

    Each template carries the variant-chunk-independent parts of one
    block read — the cache key, a :class:`vcztools.zarr_direct.BlockReader`
    bound to the field's array, and the suffix of ``block_index``
    that follows the variant chunk index slot.
    :func:`update_chunk_read_list` substitutes a specific variant
    chunk index to produce executor-ready
    ``(key, block_reader, block_index)`` tuples.

    For a ``call_*`` field the template list fans out one entry per
    sample chunk in ``sample_chunk_plan.chunk_reads``; for any other
    field exactly one entry is produced.

    Every field must be variant-axis. Static (no-variants-axis) fields
    are handled by the reader's static-field cache, not the pipeline.

    ``get_block_reader`` is a callable ``str -> BlockReader`` that
    typically routes through :meth:`VczReader._get_block_reader` so
    BlockReader instances are cached for the reader's lifetime.
    """
    templates = []
    for field in fields:
        arr = root[field]
        assert _has_variants_axis(arr), f"non-variants-axis field in pipeline: {field}"
        reader = get_block_reader(field)
        if not field.startswith("call_"):
            suffix = (slice(None),) * (arr.ndim - 1)
            templates.append(
                BlockReadTemplate(
                    key=(field,), block_reader=reader, block_index_suffix=suffix
                )
            )
        else:
            for cr in sample_chunk_plan.chunk_reads:
                suffix = (cr.index,) + (slice(None),) * (arr.ndim - 2)
                templates.append(
                    BlockReadTemplate(
                        key=(field, cr.index),
                        block_reader=reader,
                        block_index_suffix=suffix,
                    )
                )
    return templates


def update_chunk_read_list(
    templates: list[BlockReadTemplate],
    variant_chunk_index: int,
) -> list[tuple]:
    """Substitute ``variant_chunk_index`` into each template, returning
    the ``[(key, block_reader, block_index), ...]`` list that
    :func:`_produce_variant_chunks` issues fetches against. The
    template list itself is unchanged.
    """
    reads = []
    for t in templates:
        block_index = (variant_chunk_index,) + t.block_index_suffix
        reads.append((t.key, t.block_reader, block_index))
    return reads


@dataclasses.dataclass(frozen=True, slots=True)
class _VariantChunksContext:
    """Frozen snapshot of every per-iteration parameter for the
    variant_chunks producer. Built on the calling thread; consumed by
    the producer task running on the reader's anyio portal so that
    mid-iteration mutations of reader state do not affect the
    in-flight iteration.
    """

    root: object
    variant_chunk_plan: list
    sample_chunk_plan: "samples_mod.SampleChunkPlan"
    output_columns: np.ndarray | None
    read_fields: tuple
    query_fields: tuple
    filter_fields: frozenset
    referenced_static_fields: dict
    variant_filter: variant_filter_mod.VariantFilter | None
    variants_chunk_size: int
    io_limiter: anyio.CapacityLimiter
    decode_limiter: anyio.CapacityLimiter
    get_block_reader: object
    readahead_bytes: int


async def _create_memory_channel(buffer_size: int):
    """Return an anyio (send, recv) pair sized to ``buffer_size``."""
    return anyio.create_memory_object_stream[dict](max_buffer_size=buffer_size)


def _close_portal_cm(portal_cm) -> None:
    """Exit the BlockingPortal context manager, swallowing any
    teardown error so a misbehaving portal can't wedge GC. Used both
    by :meth:`VczReader.close` and by the weakref finalizer that
    arms close on garbage collection.
    """
    try:
        portal_cm.__exit__(None, None, None)
    except Exception:
        pass


async def _produce_variant_chunks(send_channel, ctx, telemetry):
    """Async producer for :meth:`VczReader.variant_chunks`.

    Iterates ``ctx.variant_chunk_plan`` in order, fetching every
    chunk's blocks concurrently inside an inner task group, applying
    the variant filter, materialising the output dict, and sending it
    through ``send_channel``. Backpressure is byte-budget controlled:
    the in-flight window expands until measured per-chunk bytes times
    in-flight count would exceed ``ctx.readahead_bytes``.

    ``telemetry`` is a shared dict that the iterator reads after
    iteration completes. Updated keys: ``max_in_flight``,
    ``last_chunk_bytes``, ``chunks_visited``, ``chunks_yielded``,
    ``variants_yielded``, ``bytes_yielded``, ``producer_assemble_total``,
    ``producer_read_total``.
    """
    templates = create_chunk_read_list(
        ctx.root,
        ctx.sample_chunk_plan,
        ctx.read_fields,
        get_block_reader=ctx.get_block_reader,
    )
    plan_iter = iter(ctx.variant_chunk_plan)
    in_flight: list[dict] = []
    per_chunk_bytes: int | None = None
    iter_start = time.perf_counter()

    try:
        async with send_channel, anyio.create_task_group() as tg:

            async def fetch_one(slot):
                vc = slot["vc"]
                t0 = time.perf_counter()
                blocks: dict[tuple, np.ndarray] = {}
                async with anyio.create_task_group() as inner:
                    for key, reader, block_index in update_chunk_read_list(
                        templates, vc.index
                    ):

                        async def one(k=key, r=reader, bi=block_index):
                            blocks[k] = await _read_block_async(
                                r, bi, ctx.io_limiter, ctx.decode_limiter
                            )

                        inner.start_soon(one)
                slot["blocks"] = blocks
                slot["read_seconds"] = time.perf_counter() - t0
                slot["bytes"] = sum(
                    utils.array_memory_bytes(v) for v in blocks.values()
                )
                slot["done"].set()

            def schedule_one() -> bool:
                try:
                    vc = next(plan_iter)
                except StopIteration:
                    return False
                slot = {"vc": vc, "done": anyio.Event()}
                in_flight.append(slot)
                if len(in_flight) > telemetry["max_in_flight"]:
                    telemetry["max_in_flight"] = len(in_flight)
                tg.start_soon(fetch_one, slot)
                logger.log(
                    TRACE,
                    f"schedule chunk {vc.index}: {len(templates)} blocks submitted",
                )
                return True

            def refill():
                # Until the first chunk has been measured we can't size the
                # window — schedule exactly one chunk and wait for its reads
                # to land. Subsequent refills fall through to the budget loop.
                if per_chunk_bytes is None:
                    if not in_flight:
                        schedule_one()
                    return
                # Always keep at least one chunk in flight; otherwise honour
                # the byte budget (use an effective per-chunk cost of at
                # least 1 to avoid an infinite loop when read_fields is
                # empty).
                pcb = max(1, per_chunk_bytes)
                while not in_flight or (len(in_flight) * pcb < ctx.readahead_bytes):
                    if not schedule_one():
                        return

            refill()
            while in_flight:
                slot = in_flight.pop(0)
                await slot["done"].wait()
                vc = slot["vc"]
                blocks = slot["blocks"]
                chunk_bytes = slot["bytes"]
                read_seconds = slot["read_seconds"]
                telemetry["last_chunk_bytes"] = chunk_bytes
                telemetry["bytes_yielded"] += chunk_bytes
                telemetry["producer_read_total"] += read_seconds
                telemetry["chunks_visited"] += 1
                if per_chunk_bytes is None:
                    per_chunk_bytes = chunk_bytes
                    if ctx.readahead_bytes > 0 and chunk_bytes > 0:
                        window_chunks = max(
                            1, ctx.readahead_bytes // max(1, chunk_bytes)
                        )
                    else:
                        window_chunks = 1
                    logger.info(
                        f"Per-chunk read size: {_fmt_bytes(chunk_bytes)} "
                        f"(chunk {vc.index}); window will hold "
                        f"~{window_chunks} chunks under budget "
                        f"{_fmt_bytes(ctx.readahead_bytes)}"
                    )
                    if (
                        ctx.readahead_bytes > 0
                        and chunk_bytes > ctx.readahead_bytes / 2
                    ):
                        logger.warning(
                            f"Readahead budget is single-chunk-bound: "
                            f"per-chunk {_fmt_bytes(chunk_bytes)} > "
                            f"half of {_fmt_bytes(ctx.readahead_bytes)}; "
                            f"the prefetch window will be capped at ~1 "
                            f"in flight regardless of worker count. "
                            f"Increase readahead_bytes to widen the window."
                        )
                logger.debug(
                    f"chunk {vc.index} read complete in {read_seconds:.2f}s "
                    f"({len(blocks)} blocks, {_fmt_bytes(chunk_bytes)})"
                )

                assemble_start = time.perf_counter()
                cached = CachedVariantChunk(
                    ctx.root,
                    vc,
                    sample_chunk_plan=ctx.sample_chunk_plan,
                    output_columns=ctx.output_columns,
                    blocks=blocks,
                )

                variants_selection = None
                sample_filter_pass = None
                if ctx.variant_filter is not None:
                    filter_data = {
                        f: ctx.referenced_static_fields[f]
                        if f in ctx.referenced_static_fields
                        else cached.filter_view(f)
                        for f in ctx.filter_fields
                    }
                    filter_result = ctx.variant_filter.evaluate(filter_data)
                    if filter_result.ndim == 1:
                        variants_selection = filter_result
                        logger.debug(
                            f"chunk {vc.index}: filter pass "
                            f"{int(filter_result.sum())}/{filter_result.size} "
                            f"variants"
                        )
                    else:
                        variants_selection = filter_result.any(axis=1)
                        sample_filter_pass = filter_result[variants_selection]
                        if ctx.output_columns is not None:
                            sample_filter_pass = sample_filter_pass[
                                :, ctx.output_columns
                            ]
                        logger.debug(
                            f"chunk {vc.index}: filter pass "
                            f"{int(variants_selection.sum())}/"
                            f"{variants_selection.size} variants, "
                            f"{int(filter_result.sum())}/{filter_result.size} "
                            f"sample cells"
                        )

                if variants_selection is not None and not variants_selection.any():
                    telemetry["producer_assemble_total"] += (
                        time.perf_counter() - assemble_start
                    )
                    refill()
                    continue

                chunk_data: dict[str, np.ndarray] = {}
                for field in ctx.query_fields:
                    if field in ctx.referenced_static_fields:
                        chunk_data[field] = ctx.referenced_static_fields[field]
                        continue
                    if field == "variant_index":
                        value = _absolute_variant_indexes(vc, ctx.variants_chunk_size)
                    else:
                        value = cached.output_view(field)
                    if variants_selection is not None:
                        value = value[variants_selection]
                    chunk_data[field] = value
                if sample_filter_pass is not None:
                    chunk_data["sample_filter_pass"] = sample_filter_pass

                non_static_query = [
                    f for f in ctx.query_fields if f not in ctx.referenced_static_fields
                ]
                if len(non_static_query) > 0:
                    chunk_variants = len(chunk_data[non_static_query[0]])
                elif variants_selection is not None:
                    chunk_variants = int(variants_selection.sum())
                else:
                    chunk_variants = 0
                telemetry["variants_yielded"] += chunk_variants
                telemetry["chunks_yielded"] += 1

                assemble_seconds = time.perf_counter() - assemble_start
                telemetry["producer_assemble_total"] += assemble_seconds
                logger.debug(
                    f"chunk {vc.index}: assembled {chunk_variants} variants in "
                    f"{assemble_seconds:.2f}s"
                )

                await send_channel.send(chunk_data)
                refill()
    finally:
        elapsed = time.perf_counter() - iter_start
        mib = telemetry["bytes_yielded"] / (1024 * 1024)
        rate = mib / elapsed if elapsed > 0 else 0.0
        logger.info(
            f"variant_chunks: iteration done in {elapsed:.2f}s "
            f"({telemetry['chunks_visited']} chunks visited, "
            f"{telemetry['chunks_yielded']} yielded, "
            f"{telemetry['variants_yielded']} variants, "
            f"{mib:.1f} MiB retrieved, {rate:.1f} MiB/s, "
            f"max readahead depth {telemetry['max_in_flight']}); "
            f"producer_assemble={telemetry['producer_assemble_total']:.2f}s, "
            f"producer_read_wait={telemetry['producer_read_total']:.2f}s"
        )


class _AsyncBackedIterator:
    """Sync iterator over an anyio memory channel populated by
    :func:`_produce_variant_chunks` running on the reader's portal.

    Bridges async → sync via ``portal.call(recv.receive)``. ``close()``
    cancels the producer and shuts the channel; ``__del__`` closes
    defensively. Producer exceptions surface on ``__next__`` once the
    channel is drained.

    ``max_in_flight`` and ``last_chunk_bytes`` expose telemetry the
    producer updates as it runs; they are intended for diagnostics
    and tests, not for the user-facing chunk dicts themselves.
    """

    def __init__(self, portal, recv, fut, telemetry):
        self._portal = portal
        self._recv = recv
        self._fut = fut
        self._telemetry = telemetry
        self._closed = False

    @property
    def max_in_flight(self) -> int:
        return self._telemetry["max_in_flight"]

    @property
    def last_chunk_bytes(self) -> int | None:
        return self._telemetry["last_chunk_bytes"]

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        try:
            return self._portal.call(self._recv.receive)
        except anyio.EndOfStream:
            self._closed = True
            try:
                self._fut.result()
            except BaseExceptionGroup as eg:
                # anyio wraps every producer-side error in an
                # ExceptionGroup via the task group's __aexit__. Unwrap
                # a single-exception group so callers see the original
                # error type (handle_exception in cli.py matches against
                # ValueError, not BaseExceptionGroup).
                if len(eg.exceptions) == 1:
                    raise eg.exceptions[0] from None
                raise
            raise StopIteration from None

    def close(self):
        if self._closed:
            return
        self._closed = True
        # Cancel the producer; close the receive end so a producer that's
        # blocked on send.send() unblocks via BrokenResourceError; wait
        # (bounded) for the task to actually finish so __del__ can't wedge
        # interpreter shutdown.
        self._fut.cancel()
        try:
            self._portal.call(self._recv.aclose)
        except Exception:
            pass
        try:
            self._fut.result(timeout=5)
        except (cf.CancelledError, cf.TimeoutError, BaseException):
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class CachedVariantChunk:
    """View assembler over prefetched blocks for one variant chunk visit.

    Constructed by :func:`_produce_variant_chunks` once its block reads
    have completed; performs no I/O itself. The constructor takes:

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

    The reader owns a single anyio :class:`BlockingPortal` (started
    lazily on first ``variant_chunks()``) plus two
    :class:`anyio.CapacityLimiter` knobs sized by ``io_concurrency``
    and ``decode_threads``. Use as a context manager
    (``with VczReader(root) as reader:``) so the portal is shut down
    deterministically on exit. Multiple concurrent
    ``variant_chunks()`` callers share the portal — startup is
    guarded by an internal lock.

    Parameters
    ----------
    root
        An already-opened :class:`zarr.Group` pointing at the VCZ
        dataset. Use :func:`vcztools.open_zarr` to open a path
        (local, remote, or zip) with the desired backend before
        constructing the reader.
    io_concurrency
        Cap on concurrent ``store.get`` calls per iteration. ``None``
        (default) uses :data:`DEFAULT_IO_CONCURRENCY` (``32``). Each
        store fetch is async-native, so this is a coroutine count,
        not a thread count.
    readahead_bytes
        Cap, in bytes, on the cross-chunk readahead window. ``None``
        (default) uses :data:`DEFAULT_READAHEAD_BYTES` (256 MiB).
        ``0`` pins window depth at 1 (one chunk prefetched ahead of
        the consumer).
    decode_threads
        Size of the decode thread pool that runs codec ``_decode_sync``
        calls inside the reader's portal. ``None`` (default) uses
        :data:`DEFAULT_DECODE_THREADS` (``os.cpu_count()``).
    """

    def __init__(
        self,
        root,
        *,
        io_concurrency: int | None = None,
        readahead_bytes: int | None = None,
        decode_threads: int | None = None,
    ):
        self.root = root
        self.readahead_bytes = readahead_bytes
        self._io_concurrency = (
            io_concurrency if io_concurrency is not None else DEFAULT_IO_CONCURRENCY
        )
        self._decode_threads = (
            decode_threads if decode_threads is not None else DEFAULT_DECODE_THREADS
        )
        # Portal + limiters are started lazily on first variant_chunks()
        # call — readers that only consume static metadata never pay
        # the asyncio-thread cost. Concurrent variant_chunks() callers
        # racing to start the portal would otherwise leak event-loop
        # threads, so the bring-up is guarded by ``_portal_lock``.
        self._portal_lock = threading.Lock()
        self._portal_cm = None
        self._portal: anyio.from_thread.BlockingPortal | None = None
        self._io_limiter: anyio.CapacityLimiter | None = None
        self._decode_limiter: anyio.CapacityLimiter | None = None
        self._finalizer: weakref.finalize | None = None
        self._block_readers: dict[str, zarr_direct.BlockReader] = {}
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
            f"io_concurrency={self._io_concurrency}, "
            f"decode_threads={self._decode_threads}, "
            f"readahead_bytes={readahead_bytes}"
        )

    def _ensure_portal(self):
        """Start the anyio portal and limiter pair on first access;
        return ``(portal, io_limiter, decode_limiter)``. Idempotent and
        thread-safe — concurrent first callers see the same portal.

        A :func:`weakref.finalize` arms ``close`` so the portal thread
        is shut down even if the caller doesn't use the reader as a
        context manager — without it, the portal's daemon thread stays
        alive past interpreter shutdown and joins on the asyncio
        default executor's non-daemon workers can wedge process exit.
        """
        with self._portal_lock:
            if self._portal is None:
                self._portal_cm = anyio.from_thread.start_blocking_portal(
                    backend="asyncio", name="vcztools-portal"
                )
                self._portal = self._portal_cm.__enter__()
                self._io_limiter = anyio.CapacityLimiter(self._io_concurrency)
                self._decode_limiter = anyio.CapacityLimiter(self._decode_threads)
                if self._finalizer is None:
                    self._finalizer = weakref.finalize(
                        self,
                        _close_portal_cm,
                        self._portal_cm,
                    )
            return self._portal, self._io_limiter, self._decode_limiter

    def _get_block_reader(self, name: str) -> zarr_direct.BlockReader:
        """Lazily construct and cache one :class:`BlockReader` per field."""
        cached = self._block_readers.get(name)
        if cached is not None:
            return cached
        reader = zarr_direct.BlockReader(self.root[name])
        self._block_readers[name] = reader
        return reader

    def close(self) -> None:
        """Tear down owned resources: the anyio portal (if started)."""
        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None
        if self._portal_cm is not None:
            _close_portal_cm(self._portal_cm)
            self._portal_cm = None
            self._portal = None
            self._io_limiter = None
            self._decode_limiter = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
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

    def variant_counts_per_chunk(self) -> np.ndarray:
        """Number of variants contributed by each entry in
        :attr:`variant_chunk_plan`.

        Returns a 1-D ``int64`` array of length
        ``len(variant_chunk_plan)``. Reads the count stored on each
        :class:`~vcztools.utils.ChunkRead` — no Zarr access. Useful
        for consumers (e.g. PLINK BED encoders) that need to size
        per-chunk output without reading any data.
        """
        return np.array(
            [cr.num_selected for cr in self.variant_chunk_plan], dtype=np.int64
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
        chunk_size = self.variants_chunk_size
        plan = []
        surviving_total = 0
        for chunk_data in self.variant_chunks(fields=["variant_index"]):
            abs_idx = chunk_data["variant_index"]
            chunk_idx = int(abs_idx[0]) // chunk_size
            local_sel = abs_idx - chunk_idx * chunk_size
            plan.append(
                utils.ChunkRead(
                    index=chunk_idx,
                    num_selected=int(abs_idx.size),
                    selection=utils.normalise_local_selection(local_sel, chunk_size),
                )
            )
            surviving_total += int(abs_idx.size)
        logger.info(
            f"materialise_variant_filter: {surviving_total} variants survive "
            f"({len(plan)} chunks)"
        )
        self.set_variant_filter(None)
        self.set_variants(plan)

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

        The returned iterator runs the chunk pipeline in a dedicated
        background thread so that the consumer's per-chunk work and
        the producer's per-chunk assembly overlap. Peak in-flight
        memory grows by one extra chunk's worth of arrays for the
        duration of iteration; ``close()`` the iterator promptly to
        release it. Argument validation is eager: ``start < 0`` and
        ``fields == []`` are detected on the call itself rather than
        on the first ``next()``.

        The reserved name ``"variant_index"`` may be passed in
        ``fields`` as a pseudo-field. It does not correspond to a Zarr
        array; instead the yielded per-chunk array contains the global
        (store-wide) variant index of each surviving variant in that
        chunk, dtype ``int64``. Pseudo-fields are not auto-discovered
        when ``fields`` is ``None``.
        """
        if start < 0:
            raise ValueError(f"start must be >= 0 (got {start})")
        if fields is not None and len(fields) == 0:
            return iter(())

        # Snapshot the filter so a mid-iteration set_variant_filter
        # can't change behaviour for this iteration.
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
        # and dynamic (prefetched per variant chunk). Pseudo-fields
        # (e.g. ``variant_index``) are query-only and never enter the
        # Zarr-backed split; they are emitted directly from per-chunk
        # plan state.
        real_query_fields = [f for f in query_fields if f not in _PSEUDO_QUERY_FIELDS]
        referenced = list(dict.fromkeys([*filter_fields, *real_query_fields]))
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
            f"io_concurrency={self._io_concurrency}, "
            f"decode_threads={self._decode_threads}); "
            f"query_fields={list(query_fields)}, "
            f"read_fields={read_fields}"
        )

        portal, io_limiter, decode_limiter = self._ensure_portal()
        ctx = _VariantChunksContext(
            root=self.root,
            variant_chunk_plan=variant_chunk_plan,
            sample_chunk_plan=sample_chunk_plan,
            output_columns=output_columns,
            read_fields=tuple(read_fields),
            query_fields=tuple(query_fields),
            filter_fields=filter_fields,
            referenced_static_fields=referenced_static_fields,
            variant_filter=variant_filter,
            variants_chunk_size=self.variants_chunk_size,
            io_limiter=io_limiter,
            decode_limiter=decode_limiter,
            get_block_reader=self._get_block_reader,
            readahead_bytes=readahead_bytes,
        )
        telemetry: dict = {
            "max_in_flight": 0,
            "last_chunk_bytes": None,
            "chunks_visited": 0,
            "chunks_yielded": 0,
            "variants_yielded": 0,
            "bytes_yielded": 0,
            "producer_assemble_total": 0.0,
            "producer_read_total": 0.0,
        }
        send, recv = portal.call(_create_memory_channel, 1)
        fut = portal.start_task_soon(_produce_variant_chunks, send, ctx, telemetry)
        return _AsyncBackedIterator(portal, recv, fut, telemetry)

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
