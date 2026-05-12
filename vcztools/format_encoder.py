"""Abstract base class for fixed-size, random-access byte-stream encoders
over a VCZ store.

Subclasses (:class:`vcztools.plink.BedEncoder`,
:class:`vcztools.bgen.BgenEncoder`) implement the format-specific encoding
of a single variant chunk; everything else — POSIX-style ``read(off, size)``
semantics, chunk-resident state machine, iterator restart vs. advance
arbitration, thread-pool lifecycle, per-format prefix/header serving — lives
here.

Shared contract:

* Construction is I/O-free: byte offsets are derived from the reader's chunk
  plan via :meth:`~vcztools.retrieval.VczReader.variant_counts_per_chunk`.
* :meth:`read` is POSIX-style: reads past EOF return ``b""``, ``size`` is
  clamped to the end of the stream, negative ``off``/``size`` raise.
* Reads whose start lies in the loaded chunk or the immediately-next plan
  chunk advance the running iterator one chunk at a time. Further offsets
  tear down and rebuild the iterator at the chunk containing ``off``.
* Per-chunk encoding may parallelise across variant-axis sub-blocks via
  :meth:`_parallel_encode`. Each encoder owns its own
  :class:`~concurrent.futures.ThreadPoolExecutor`, created in ``__init__``
  and joined in :meth:`close` / :meth:`__exit__`.
* Concurrency: a single encoder instance is **not** thread-safe. Multiple
  instances may share one :class:`~vcztools.retrieval.VczReader` safely;
  each runs an independent variant-chunk iteration. The caller owns the
  reader's lifetime — :meth:`close` tears down the encoder's iterator only,
  not the reader.

Subclass authoring:

* Pass ``bytes_per_variant``, ``prefix_bytes``, and ``iterator_fields`` to
  ``super().__init__()``. The base class does not call back into the
  subclass during construction.
* Implement :meth:`_encode_chunk` to produce ``bytes`` for a single chunk
  yielded by the reader's variant iterator. Per-format validation
  (e.g. biallelic / diploid checks) belongs at the top of this method.
* Optionally override :meth:`_close_hook` for format-specific close-time
  logging (e.g. mixed-phase variant warnings).
* Log messages are emitted under ``logging.getLogger(type(self).__module__)``
  so existing tests that scope ``caplog.at_level(logger="vcztools.bgen" |
  "vcztools.plink")`` continue to work after the refactor.
"""

import abc
import concurrent.futures as cf
import logging
from collections.abc import Callable

import numpy as np

from vcztools import retrieval


class FormatEncoder(abc.ABC):
    """Abstract base class for fixed-size streaming VCZ encoders.

    See module docstring for the shared contract and subclass authoring
    guidelines. The only mandatory hook is :meth:`_encode_chunk`.
    """

    def __init__(
        self,
        reader: retrieval.VczReader,
        *,
        bytes_per_variant: int,
        prefix_bytes: bytes,
        iterator_fields: list[str],
        encode_threads: int | None = None,
        encode_block_bytes: int | None = None,
    ):
        if encode_threads is None:
            encode_threads = 4
        if encode_block_bytes is None:
            encode_block_bytes = 10 * 1024 * 1024

        if reader.variant_filter is not None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support readers with a "
                "set_variant_filter() configured. Apply the filter "
                "externally and pass the resulting reader, or use "
                "set_variants() to materialise the surviving indices."
            )
        if encode_threads < 1:
            raise ValueError(f"encode_threads must be >= 1 (got {encode_threads})")
        if encode_block_bytes < 1:
            raise ValueError(
                f"encode_block_bytes must be >= 1 (got {encode_block_bytes})"
            )

        self._reader = reader
        self._closed = False

        self._num_samples = int(reader.sample_ids.size)
        self._bytes_per_variant = int(bytes_per_variant)

        self._prefix_bytes = bytes(prefix_bytes)
        self._prefix_size = len(self._prefix_bytes)

        self._iterator_fields = list(iterator_fields)

        counts = reader.variant_counts_per_chunk()
        self._num_variants = int(counts.sum())
        # _chunk_byte_offsets: cumulative byte offset per plan entry,
        # starting at prefix_size (post-prefix). Length = len(plan) + 1;
        # the trailing entry equals total_size.
        self._chunk_byte_offsets = np.empty(len(counts) + 1, dtype=np.int64)
        self._chunk_byte_offsets[0] = self._prefix_size
        np.cumsum(counts * self._bytes_per_variant, out=self._chunk_byte_offsets[1:])
        self._chunk_byte_offsets[1:] += self._prefix_size
        self._total_size = int(self._chunk_byte_offsets[-1])

        self._iterator = None
        self._chunk_bytes: bytes | None = None
        self._chunk_start = 0
        self._chunk_plan_pos = -1
        # Counts only true restarts (offset jumps). The first read() after
        # construction is mechanically a restart-from-None but is logged
        # as an "init" event and does not increment this counter.
        self._restart_count = 0

        self._encode_threads = encode_threads
        self._encode_block_bytes = encode_block_bytes
        # Logger resolves under the subclass's module so caplog scoping
        # by "vcztools.bgen" / "vcztools.plink" still captures records.
        self._logger = logging.getLogger(type(self).__module__)
        thread_tag = type(self).__name__.removesuffix("Encoder").lower()
        self._executor = cf.ThreadPoolExecutor(
            max_workers=encode_threads,
            thread_name_prefix=f"vcztools-encode-{thread_tag}",
        )

    # --- Abstract hook ---

    @abc.abstractmethod
    def _encode_chunk(self, chunk: dict) -> bytes:
        """Encode a single chunk yielded by the reader's variant iterator.

        Receives the dict produced by ``reader.variant_chunks(fields=...)``
        for the field list configured at construction. Must return exactly
        ``num_variants_in_chunk * bytes_per_variant`` bytes. Per-format
        validation (biallelic, diploid, ...) belongs at the top of this
        method. May dispatch to the encode thread pool via
        :meth:`_parallel_encode`.
        """

    # --- Optional override ---

    def _close_hook(self) -> None:  # noqa: B027 — intentional no-op default
        """Hook called once at the end of :meth:`close` for format-specific
        close-time logging. Default: no-op. Called exactly once across the
        encoder's lifetime (idempotent ``close()`` does not re-invoke it).
        """

    # --- Properties ---

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("encoder closed")

    @property
    def num_variants(self) -> int:
        self._check_open()
        return self._num_variants

    @property
    def num_samples(self) -> int:
        self._check_open()
        return self._num_samples

    @property
    def bytes_per_variant(self) -> int:
        self._check_open()
        return self._bytes_per_variant

    @property
    def prefix_size(self) -> int:
        self._check_open()
        return self._prefix_size

    @property
    def total_size(self) -> int:
        self._check_open()
        return self._total_size

    # --- Read pipeline ---

    def read(self, off: int, size: int) -> bytes:
        """Return up to ``size`` bytes from the virtual stream at ``off``.

        POSIX-read semantics:

        - ``b""`` if ``off >= total_size`` or ``size == 0``
        - ``size`` clamped to the end of the stream
        - ``off < 0`` or ``size < 0`` raises ``ValueError``

        Reads whose start falls in the loaded chunk or the immediately-
        next plan chunk are served by slicing chunk-resident bytes,
        advancing the running iterator one chunk at a time as needed.
        Reads whose start is further away rebuild the iterator at the
        chunk containing ``off``.
        """
        self._check_open()
        if off < 0:
            raise ValueError(f"off must be >= 0 (got {off})")
        if size < 0:
            raise ValueError(f"size must be >= 0 (got {size})")
        if off >= self._total_size or size == 0:
            return b""
        end = min(off + size, self._total_size)

        out = bytearray()
        if off < self._prefix_size:
            out.extend(self._prefix_bytes[off : min(end, self._prefix_size)])
            off = min(end, self._prefix_size)
        if off < end:
            out.extend(self._read_data(off, end - off))
        return bytes(out)

    def _read_data(self, off: int, size: int) -> bytes:
        # Caller guarantees off >= prefix_size, size > 0,
        # off + size <= total_size. Off is reachable without restart iff
        # it lies in the loaded chunk or the immediately-next plan chunk:
        # at most one advance gets us to the chunk containing off.
        # Anything further skips a chunk's bytes that nobody asked for,
        # so restart is cheaper than advance.
        in_range = False
        if self._chunk_bytes is not None and self._chunk_start <= off:
            next_plan_end_idx = self._chunk_plan_pos + 2
            if next_plan_end_idx < len(self._chunk_byte_offsets):
                reachable_end = int(self._chunk_byte_offsets[next_plan_end_idx])
            else:
                reachable_end = self._chunk_start + len(self._chunk_bytes)
            in_range = off < reachable_end
        if not in_range:
            self._restart(off)

        out = bytearray()
        while len(out) < size:
            # If off has crossed into a chunk we haven't loaded yet, roll
            # forward via the running iterator. The >= covers both the
            # exact trailing-edge boundary and the case where off sits
            # inside the next chunk past its start.
            if off >= self._chunk_start + len(self._chunk_bytes):
                self._advance()
            local = off - self._chunk_start
            take = min(size - len(out), len(self._chunk_bytes) - local)
            out.extend(self._chunk_bytes[local : local + take])
            off += take
        return bytes(out)

    def _advance(self) -> None:
        chunk = next(self._iterator)
        encoded = self._encode_chunk(chunk)
        self._chunk_plan_pos += 1
        self._chunk_start = int(self._chunk_byte_offsets[self._chunk_plan_pos])
        self._chunk_bytes = bytes(encoded)

    def _restart(self, off: int) -> None:
        prev_plan_pos = self._chunk_plan_pos
        self._teardown_iterator()
        # searchsorted(side="right") - 1: largest plan position whose
        # start offset is <= off. _chunk_byte_offsets has len(plan)+1
        # entries (the trailing entry is total_size); off < total_size is
        # guaranteed by read(), so the index is in range.
        plan_pos = int(np.searchsorted(self._chunk_byte_offsets, off, side="right") - 1)
        self._iterator = self._reader.variant_chunks(
            fields=self._iterator_fields,
            start=plan_pos,
        )
        self._chunk_plan_pos = plan_pos - 1
        self._advance()
        label = type(self).__name__
        if prev_plan_pos == -1:
            self._logger.debug(f"{label} iterator init: off={off}, plan_pos={plan_pos}")
        else:
            self._restart_count += 1
            self._logger.info(
                f"{label} restart #{self._restart_count}: "
                f"off={off}, plan_pos={prev_plan_pos} → {plan_pos}"
            )

    def _teardown_iterator(self) -> None:
        if self._iterator is not None:
            self._iterator.close()
            self._iterator = None
        self._chunk_bytes = None
        self._chunk_plan_pos = -1

    # --- Lifecycle ---

    def close(self) -> None:
        """Tear down the active chunk iterator, shut down the encode
        thread pool, drop iterator state, and invoke :meth:`_close_hook`.
        Does not close the underlying reader. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._teardown_iterator()
        self._executor.shutdown(wait=True)
        if self._restart_count > 0:
            self._logger.debug(
                f"{type(self).__name__} closed: {self._restart_count} "
                f"restarts over {self._total_size} bytes"
            )
        self._close_hook()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # --- Parallel encode helper ---

    def _parallel_encode(
        self,
        *,
        num_variants: int,
        block_variants: int,
        sequential_threshold_bytes: int,
        encode_range: Callable[[int, int], bytes],
    ) -> bytes:
        """Encode ``num_variants`` worth of variant-axis output, optionally
        splitting across the thread pool.

        Sub-block layout: ``encode_range(start, end)`` must produce
        exactly ``(end - start) * bytes_per_variant`` bytes for variants
        ``[start, end)``. Sub-blocks are assembled into the chunk's final
        byte buffer at deterministic offsets, so caller-side ordering
        does not matter — workers complete in any order.

        Threshold semantics are caller-determined:
        ``sequential_threshold_bytes`` is what the caller compares to
        ``self._encode_block_bytes`` to decide between sequential and
        parallel. BGEN passes the chunk's output bytes
        (``num_variants * bytes_per_variant``); BED passes the chunk's
        input genotype bytes (``G.nbytes``). The two semantics are
        intentionally distinct — do not unify here.
        """
        if (
            self._encode_threads <= 1
            or sequential_threshold_bytes <= self._encode_block_bytes
        ):
            return encode_range(0, num_variants)

        bpv = self._bytes_per_variant
        output = bytearray(num_variants * bpv)
        future_to_range = {}
        for start in range(0, num_variants, block_variants):
            end = min(start + block_variants, num_variants)
            future = self._executor.submit(encode_range, start, end)
            future_to_range[future] = (start, end)
        for future in cf.as_completed(future_to_range):
            start, end = future_to_range[future]
            output[start * bpv : end * bpv] = future.result()
        return bytes(output)
