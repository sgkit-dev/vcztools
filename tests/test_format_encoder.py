"""
Unit tests for :mod:`vcztools.format_encoder`.

These exercise the abstract :class:`FormatEncoder` base class through a
minimal in-test concrete subclass, :class:`_FakeEncoder`, which emits a
deterministic, oracle-checkable byte stream per variant. The tests do
not go through ``BedEncoder``/``BgenEncoder`` so that they stay coupled
only to the base-class machinery and not to any format's encoding.

Goal: 100% line + branch coverage of ``vcztools/format_encoder.py``.
"""

import io
import logging
import threading
import time

import numpy as np
import pytest

from tests import vcz_builder
from vcztools import bcftools_filter, format_encoder, retrieval, utils

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _emit_range(G, start, end, bpv):
    """Per-variant output: ``bpv`` copies of ``G[i, 0, 0] & 0xFF``."""
    out = bytearray()
    for i in range(start, end):
        val = int(G[i, 0, 0]) & 0xFF
        out.extend(bytes([val]) * bpv)
    return bytes(out)


class _FakeEncoder(format_encoder.FormatEncoder):
    """Minimal concrete encoder for testing the abstract base class.

    Per-variant output is ``bpv`` copies of ``call_genotype[i, 0, 0] & 0xFF``,
    so when the fixture sets ``call_genotype[v, 0, 0] = v`` the encoded
    bytes are globally unique and easy to oracle.
    """

    def __init__(
        self,
        reader,
        *,
        prefix=b"FAKE",
        bpv=4,
        use_parallel=False,
        encode_threads=None,
        encode_block_bytes=None,
    ):
        self._fake_use_parallel = use_parallel
        self._encode_calls = 0
        self._close_hook_calls = 0
        self._chunks_received = []
        self._encode_thread_names: list[str] = []
        self._encode_raise_once = False
        self._encode_raise_in_worker = False
        # When set, _encode_chunk raises if the chunk's first variant
        # has this value in call_genotype[0, 0, 0]. Used for tests that
        # need a deterministic failure trigger independent of prefetch
        # ordering (since _encode_raise_once may be consumed by an
        # in-flight prefetched chunk during teardown).
        self._fail_if_first_variant: int | None = None
        super().__init__(
            reader,
            bytes_per_variant=bpv,
            prefix_bytes=prefix,
            iterator_fields=["call_genotype"],
            encode_threads=encode_threads,
            encode_block_bytes=encode_block_bytes,
        )

    def _encode_chunk(self, chunk):
        self._encode_calls += 1
        self._encode_thread_names.append(threading.current_thread().name)
        self._chunks_received.append(tuple(sorted(chunk.keys())))
        if self._encode_raise_once:
            self._encode_raise_once = False
            raise ValueError("fake encode failure")
        G = chunk["call_genotype"]
        if (
            self._fail_if_first_variant is not None
            and int(G[0, 0, 0]) == self._fail_if_first_variant
        ):
            raise ValueError("fake encode failure")
        num_variants = G.shape[0]
        bpv = self._bytes_per_variant
        if self._fake_use_parallel:

            def encode_range(start, end, out_view):
                if self._encode_raise_in_worker:
                    raise RuntimeError("worker boom")
                out_view[:] = np.frombuffer(
                    _emit_range(G, start, end, bpv), dtype=np.uint8
                )

            return self._parallel_encode(
                num_variants=num_variants,
                encode_range=encode_range,
            )
        return _emit_range(G, 0, num_variants, bpv)

    def _close_hook(self):
        self._close_hook_calls += 1


def _build_reader(
    *,
    num_variants=8,
    num_samples=2,
    variants_chunk_size=4,
    **overrides,
):
    if "call_genotype" not in overrides:
        cg = np.zeros((num_variants, num_samples, 2), dtype=np.int8)
        cg[:, 0, 0] = np.arange(num_variants, dtype=np.int8)
        overrides["call_genotype"] = cg
    defaults = dict(
        variant_contig=[0] * num_variants,
        variant_position=list(range(100, 100 + num_variants)),
        alleles=[("A", "T")] * num_variants,
        num_samples=num_samples,
        variants_chunk_size=variants_chunk_size,
    )
    defaults.update(overrides)
    root = vcz_builder.make_vcz(**defaults)
    return retrieval.VczReader(root)


def _oracle(num_variants, bpv, prefix=b"FAKE"):
    data = bytearray(prefix)
    for v in range(num_variants):
        data.extend(bytes([v & 0xFF]) * bpv)
    return bytes(data)


def _drain(encoder, step):
    out = bytearray()
    off = 0
    total = encoder.total_size
    while off < total:
        buf = encoder.read(off, step)
        out.extend(buf)
        off += len(buf)
    return bytes(out)


def _wrap_restart(enc):
    counter = {"n": 0}
    original = enc._restart

    def counting(off):
        counter["n"] += 1
        original(off)

    enc._restart = counting
    return counter


def _wrap_submit(enc):
    counter = {"n": 0}
    real = enc._executor.submit

    def counting(*args, **kwargs):
        counter["n"] += 1
        return real(*args, **kwargs)

    enc._executor.submit = counting
    return counter


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def test_defaults_applied(self):
        enc = _FakeEncoder(_build_reader())
        assert enc._encode_threads == 4
        assert enc._encode_block_bytes == 10 * 1024 * 1024
        enc.close()

    def test_explicit_values_stored(self):
        enc = _FakeEncoder(
            _build_reader(),
            encode_threads=2,
            encode_block_bytes=4096,
        )
        assert enc._encode_threads == 2
        assert enc._encode_block_bytes == 4096
        enc.close()

    def test_encode_threads_zero_raises(self):
        with pytest.raises(ValueError, match="encode_threads must be >= 1"):
            _FakeEncoder(_build_reader(), encode_threads=0)

    def test_encode_threads_negative_raises(self):
        with pytest.raises(ValueError, match="encode_threads must be >= 1"):
            _FakeEncoder(_build_reader(), encode_threads=-1)

    def test_encode_block_bytes_zero_raises(self):
        with pytest.raises(ValueError, match="encode_block_bytes must be >= 1"):
            _FakeEncoder(_build_reader(), encode_block_bytes=0)

    def test_encode_block_bytes_negative_raises(self):
        with pytest.raises(ValueError, match="encode_block_bytes must be >= 1"):
            _FakeEncoder(_build_reader(), encode_block_bytes=-1)

    def test_variant_filter_rejected(self):
        reader = _build_reader()
        reader.set_variant_filter(
            bcftools_filter.BcftoolsFilter(reader, include="N_ALT <= 1")
        )
        with pytest.raises(NotImplementedError, match="_FakeEncoder"):
            _FakeEncoder(reader)

    def test_executor_thread_prefix_from_classname(self):
        enc = _FakeEncoder(_build_reader())
        # Prefix is derived from type(self).__name__ with the "Encoder"
        # suffix stripped and the rest lowercased — "_fake" for this
        # test subclass; "bed" / "bgen" for the production encoders.
        assert enc._executor._thread_name_prefix == "vcztools-encode-_fake"
        enc.close()

    def test_logger_name_from_subclass_module(self):
        enc = _FakeEncoder(_build_reader())
        assert enc._logger.name == _FakeEncoder.__module__
        enc.close()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_num_variants(self):
        enc = _FakeEncoder(_build_reader(num_variants=10))
        assert enc.num_variants == 10
        enc.close()

    def test_num_samples(self):
        enc = _FakeEncoder(_build_reader(num_samples=5))
        assert enc.num_samples == 5
        enc.close()

    def test_bytes_per_variant(self):
        enc = _FakeEncoder(_build_reader(), bpv=7)
        assert enc.bytes_per_variant == 7
        enc.close()

    def test_prefix_size(self):
        enc = _FakeEncoder(_build_reader(), prefix=b"HEADERBYTES")
        assert enc.prefix_size == 11
        enc.close()

    def test_total_size(self):
        enc = _FakeEncoder(_build_reader(num_variants=10), bpv=3, prefix=b"AB")
        assert enc.total_size == 2 + 10 * 3
        enc.close()

    def test_chunk_byte_offsets_layout(self):
        enc = _FakeEncoder(
            _build_reader(num_variants=10, variants_chunk_size=4),
            bpv=3,
            prefix=b"AB",
        )
        offsets = enc._chunk_byte_offsets
        # 10 variants / chunk size 4 => 3 chunks: [4, 4, 2]
        assert offsets.tolist() == [2, 2 + 4 * 3, 2 + 8 * 3, 2 + 10 * 3]
        assert offsets[0] == enc.prefix_size
        assert offsets[-1] == enc.total_size
        enc.close()

    def test_properties_after_close_raise(self):
        enc = _FakeEncoder(_build_reader())
        enc.close()
        for name in (
            "num_variants",
            "num_samples",
            "bytes_per_variant",
            "prefix_size",
            "total_size",
        ):
            with pytest.raises(RuntimeError, match="encoder closed"):
                getattr(enc, name)


# ---------------------------------------------------------------------------
# read() edge cases
# ---------------------------------------------------------------------------


class TestReadEdges:
    def test_negative_off_raises(self):
        enc = _FakeEncoder(_build_reader())
        with pytest.raises(ValueError, match="off must be >= 0"):
            enc.read(-1, 10)
        enc.close()

    def test_negative_size_raises(self):
        enc = _FakeEncoder(_build_reader())
        with pytest.raises(ValueError, match="size must be >= 0"):
            enc.read(0, -1)
        enc.close()

    def test_off_at_end_returns_empty(self):
        enc = _FakeEncoder(_build_reader())
        assert enc.read(enc.total_size, 10) == b""
        enc.close()

    def test_off_past_end_returns_empty(self):
        enc = _FakeEncoder(_build_reader())
        assert enc.read(enc.total_size + 5, 10) == b""
        enc.close()

    def test_size_zero_returns_empty(self):
        enc = _FakeEncoder(_build_reader())
        assert enc.read(0, 0) == b""
        enc.close()

    def test_size_clamped_to_end(self):
        enc = _FakeEncoder(_build_reader(num_variants=4), bpv=2)
        # total_size = 4 (prefix) + 4*2 = 12. Request 100 bytes starting at 8.
        result = enc.read(8, 100)
        assert len(result) == enc.total_size - 8
        enc.close()

    def test_prefix_only_read(self):
        enc = _FakeEncoder(_build_reader(), prefix=b"HEADER")
        assert enc.read(0, 3) == b"HEA"
        assert enc.read(2, 4) == b"ADER"
        enc.close()

    def test_prefix_and_data_read(self):
        enc = _FakeEncoder(_build_reader(num_variants=4), bpv=2, prefix=b"AB")
        result = enc.read(0, enc.total_size)
        assert result == _oracle(4, 2, prefix=b"AB")
        enc.close()

    def test_data_only_read(self):
        enc = _FakeEncoder(_build_reader(num_variants=4), bpv=2, prefix=b"AB")
        result = enc.read(enc.prefix_size, 4)
        assert result == _oracle(4, 2, prefix=b"AB")[2:6]
        enc.close()

    def test_off_exactly_at_prefix_boundary(self):
        enc = _FakeEncoder(_build_reader(num_variants=4), bpv=2, prefix=b"AB")
        # off == prefix_size: skips prefix branch, lands at first chunk start
        result = enc.read(2, 2)
        assert result == _oracle(4, 2, prefix=b"AB")[2:4]
        enc.close()

    def test_off_plus_size_equals_total_size_exactly(self):
        enc = _FakeEncoder(_build_reader(num_variants=4), bpv=2, prefix=b"AB")
        # Drain-to-end without clamping
        result = enc.read(0, enc.total_size)
        assert len(result) == enc.total_size
        enc.close()

    def test_tail_byte_read(self):
        enc = _FakeEncoder(_build_reader(num_variants=4), bpv=2, prefix=b"AB")
        oracle = _oracle(4, 2, prefix=b"AB")
        assert enc.read(enc.total_size - 1, 1) == oracle[-1:]
        enc.close()

    def test_multiple_reads_past_eof_stable(self):
        enc = _FakeEncoder(_build_reader())
        for _ in range(3):
            assert enc.read(enc.total_size + 100, 50) == b""
        enc.close()

    def test_read_after_close_raises(self):
        enc = _FakeEncoder(_build_reader())
        enc.close()
        with pytest.raises(RuntimeError, match="encoder closed"):
            enc.read(0, 10)

    def test_prefixless_encoder_read(self):
        enc = _FakeEncoder(_build_reader(num_variants=4), bpv=2, prefix=b"")
        assert enc.prefix_size == 0
        oracle = _oracle(4, 2, prefix=b"")
        # read(0, N) takes the data-only path (prefix branch skipped)
        assert enc.read(0, enc.total_size) == oracle
        enc.close()


# ---------------------------------------------------------------------------
# Sequential drain
# ---------------------------------------------------------------------------


class TestSequentialDrain:
    @pytest.mark.parametrize("step", [1, 7, 17, 4096, 1 << 17])
    def test_drain_at_step_matches_oracle(self, step):
        num_variants = 10
        bpv = 4
        enc = _FakeEncoder(
            _build_reader(num_variants=num_variants, variants_chunk_size=3),
            bpv=bpv,
        )
        assert _drain(enc, step) == _oracle(num_variants, bpv)
        enc.close()

    def test_single_read_full_stream(self):
        num_variants = 10
        bpv = 4
        enc = _FakeEncoder(
            _build_reader(num_variants=num_variants, variants_chunk_size=3),
            bpv=bpv,
        )
        assert enc.read(0, enc.total_size) == _oracle(num_variants, bpv)
        enc.close()


# ---------------------------------------------------------------------------
# Chunk-resident reads (no-restart invariant)
# ---------------------------------------------------------------------------


class TestChunkResidentReads:
    """Reads whose start lies in the loaded chunk or the immediately-next
    plan chunk advance via the running iterator without restart. Mirrors
    the invariant exercised in test_plink::TestBedEncoderChunkResidentReads.
    """

    NUM_VARIANTS = 20
    CHUNK_SIZE = 4
    BPV = 4

    def _enc_and_oracle(self):
        enc = _FakeEncoder(
            _build_reader(
                num_variants=self.NUM_VARIANTS,
                variants_chunk_size=self.CHUNK_SIZE,
            ),
            bpv=self.BPV,
        )
        oracle = _oracle(self.NUM_VARIANTS, self.BPV)
        return enc, oracle

    def test_in_chunk_forward_jump_no_restart(self):
        enc, oracle = self._enc_and_oracle()
        prefix = enc.prefix_size
        bpv = enc.bytes_per_variant
        counter = _wrap_restart(enc)
        first = enc.read(prefix, 1)
        second = enc.read(prefix + 2 * bpv, 5)
        assert first == oracle[prefix : prefix + 1]
        assert second == oracle[prefix + 2 * bpv : prefix + 2 * bpv + 5]
        assert counter["n"] == 1  # init only
        enc.close()

    def test_in_chunk_backward_jump_no_restart(self):
        enc, oracle = self._enc_and_oracle()
        prefix = enc.prefix_size
        bpv = enc.bytes_per_variant
        counter = _wrap_restart(enc)
        enc.read(prefix + 2 * bpv, 5)
        enc.read(prefix + bpv, 5)
        assert counter["n"] == 1
        enc.close()

    def test_cross_one_boundary_no_restart(self):
        enc, oracle = self._enc_and_oracle()
        prefix = enc.prefix_size
        bpv = enc.bytes_per_variant
        counter = _wrap_restart(enc)
        enc.read(prefix, 1)
        # Span variants 3..6: trailing edge of chunk 0 plus first two of chunk 1
        result = enc.read(prefix + 3 * bpv, 3 * bpv)
        assert result == oracle[prefix + 3 * bpv : prefix + 6 * bpv]
        assert counter["n"] == 1
        enc.close()

    def test_off_exactly_at_next_chunk_start_no_restart(self):
        enc, oracle = self._enc_and_oracle()
        prefix = enc.prefix_size
        bpv = enc.bytes_per_variant
        counter = _wrap_restart(enc)
        enc.read(prefix, 1)
        # off at chunk 1 start (variant index 4)
        result = enc.read(prefix + 4 * bpv, 2 * bpv)
        assert result == oracle[prefix + 4 * bpv : prefix + 6 * bpv]
        assert counter["n"] == 1
        enc.close()

    def test_off_mid_next_chunk_no_restart(self):
        enc, oracle = self._enc_and_oracle()
        prefix = enc.prefix_size
        bpv = enc.bytes_per_variant
        counter = _wrap_restart(enc)
        enc.read(prefix, 1)
        # off lands one variant into chunk 1; size stays inside chunk 1
        result = enc.read(prefix + 5 * bpv, bpv)
        assert result == oracle[prefix + 5 * bpv : prefix + 6 * bpv]
        assert counter["n"] == 1
        enc.close()

    def test_span_all_chunks_one_read_no_restart(self):
        enc, oracle = self._enc_and_oracle()
        counter = _wrap_restart(enc)
        assert enc.read(0, enc.total_size) == oracle
        assert counter["n"] == 1
        enc.close()

    def test_multi_chunk_advance_via_read_loop(self):
        # off mid chunk 1; size spans chunks 1, 2, 3 — runs the
        # advance-via-loop branch repeatedly with a single restart.
        enc, oracle = self._enc_and_oracle()
        prefix = enc.prefix_size
        bpv = enc.bytes_per_variant
        counter = _wrap_restart(enc)
        enc.read(prefix, 1)
        result = enc.read(prefix + 6 * bpv, 8 * bpv)
        assert result == oracle[prefix + 6 * bpv : prefix + 14 * bpv]
        assert counter["n"] == 1
        enc.close()


def _wrap_advance(enc):
    counter = {"n": 0}
    original = enc._advance

    def counting():
        counter["n"] += 1
        original()

    enc._advance = counting
    return counter


class TestTryCachedRead:
    """try_cached_read returns bytes iff the requested range can be
    served from prefix bytes and/or the currently-loaded chunk
    without advancing the variant iterator. Otherwise None. Same
    arg validation and EOF semantics as read."""

    NUM_VARIANTS = 20
    CHUNK_SIZE = 4
    BPV = 4

    def _enc_and_oracle(self):
        enc = _FakeEncoder(
            _build_reader(
                num_variants=self.NUM_VARIANTS,
                variants_chunk_size=self.CHUNK_SIZE,
            ),
            bpv=self.BPV,
        )
        oracle = _oracle(self.NUM_VARIANTS, self.BPV)
        return enc, oracle

    def test_returns_none_before_any_read(self):
        enc, _ = self._enc_and_oracle()
        # No chunk loaded yet; the request lies outside the prefix.
        assert enc.try_cached_read(enc.prefix_size, self.BPV) is None
        enc.close()

    def test_returns_prefix_bytes_with_no_chunk_loaded(self):
        enc, oracle = self._enc_and_oracle()
        # Reads contained entirely in the prefix don't need a chunk.
        result = enc.try_cached_read(0, enc.prefix_size)
        assert result == oracle[: enc.prefix_size]
        enc.close()

    def test_returns_eof_immediately(self):
        enc, _ = self._enc_and_oracle()
        assert enc.try_cached_read(enc.total_size, 100) == b""
        assert enc.try_cached_read(enc.total_size + 1000, 1) == b""
        assert enc.try_cached_read(0, 0) == b""
        enc.close()

    def test_returns_bytes_inside_loaded_chunk(self):
        enc, oracle = self._enc_and_oracle()
        prefix = enc.prefix_size
        bpv = enc.bytes_per_variant
        enc.read(prefix, 1)  # warm chunk 0
        result = enc.try_cached_read(prefix + bpv, bpv)
        assert result == oracle[prefix + bpv : prefix + 2 * bpv]
        enc.close()

    def test_returns_full_loaded_chunk_range(self):
        enc, oracle = self._enc_and_oracle()
        prefix = enc.prefix_size
        bpv = enc.bytes_per_variant
        chunk_bytes_total = self.CHUNK_SIZE * bpv
        enc.read(prefix, 1)
        result = enc.try_cached_read(prefix, chunk_bytes_total)
        assert result == oracle[prefix : prefix + chunk_bytes_total]
        enc.close()

    def test_returns_none_for_range_beyond_loaded_chunk(self):
        enc, _ = self._enc_and_oracle()
        prefix = enc.prefix_size
        bpv = enc.bytes_per_variant
        chunk_bytes_total = self.CHUNK_SIZE * bpv
        enc.read(prefix, 1)  # warm chunk 0
        # First byte of chunk 1 is out of range.
        assert enc.try_cached_read(prefix + chunk_bytes_total, 1) is None
        enc.close()

    def test_returns_none_for_range_before_loaded_chunk(self):
        enc, _ = self._enc_and_oracle()
        prefix = enc.prefix_size
        bpv = enc.bytes_per_variant
        chunk_bytes_total = self.CHUNK_SIZE * bpv
        # Warm chunk 2.
        enc.read(prefix + 2 * chunk_bytes_total, 1)
        # Chunk 0's first variant is before the loaded chunk; even
        # though it would be reachable via restart, try_cached_read
        # refuses to advance.
        assert enc.try_cached_read(prefix, bpv) is None
        enc.close()

    def test_returns_bytes_spanning_prefix_and_loaded_chunk(self):
        enc, oracle = self._enc_and_oracle()
        prefix = enc.prefix_size
        bpv = enc.bytes_per_variant
        enc.read(prefix, 1)  # warm chunk 0
        # Read straddles prefix → chunk 0.
        result = enc.try_cached_read(prefix - 2, 2 + bpv)
        assert result == oracle[prefix - 2 : prefix + bpv]
        enc.close()

    def test_returns_none_when_range_straddles_chunk_boundary(self):
        enc, _ = self._enc_and_oracle()
        prefix = enc.prefix_size
        bpv = enc.bytes_per_variant
        chunk_bytes_total = self.CHUNK_SIZE * bpv
        enc.read(prefix, 1)  # warm chunk 0
        # Read spans last byte of chunk 0 + first byte of chunk 1.
        result = enc.try_cached_read(prefix + chunk_bytes_total - 1, 2)
        assert result is None
        enc.close()

    def test_size_clamped_to_stream_end_within_loaded_chunk(self):
        # When the loaded chunk includes the trailing edge of the
        # stream, an over-large size is clamped — try_cached_read
        # returns the bytes through total_size.
        enc, oracle = self._enc_and_oracle()
        prefix = enc.prefix_size
        bpv = enc.bytes_per_variant
        chunk_bytes_total = self.CHUNK_SIZE * bpv
        last_chunk_first_byte = prefix + (self.NUM_VARIANTS - self.CHUNK_SIZE) * bpv
        enc.read(last_chunk_first_byte, 1)
        result = enc.try_cached_read(last_chunk_first_byte, chunk_bytes_total + 1000)
        assert result == oracle[last_chunk_first_byte:]
        enc.close()

    def test_no_iterator_advance_on_hit_or_miss(self):
        enc, _ = self._enc_and_oracle()
        prefix = enc.prefix_size
        bpv = enc.bytes_per_variant
        chunk_bytes_total = self.CHUNK_SIZE * bpv
        enc.read(prefix, 1)  # one advance to load chunk 0
        advance_ctr = _wrap_advance(enc)
        restart_ctr = _wrap_restart(enc)
        # Hits: prefix, in-chunk, EOF, size=0.
        enc.try_cached_read(0, enc.prefix_size)
        enc.try_cached_read(prefix + bpv, bpv)
        enc.try_cached_read(enc.total_size, 100)
        enc.try_cached_read(0, 0)
        # Misses: beyond chunk, across boundary, fresh-encoder-style
        # (no chunk loaded — simulate via teardown of state would
        # require touching internals; this still exercises a miss).
        assert enc.try_cached_read(prefix + chunk_bytes_total, 1) is None
        assert enc.try_cached_read(prefix + chunk_bytes_total - 1, 2) is None
        assert advance_ctr["n"] == 0
        assert restart_ctr["n"] == 0
        enc.close()

    def test_validates_negative_off(self):
        enc, _ = self._enc_and_oracle()
        with pytest.raises(ValueError, match="off must be >= 0"):
            enc.try_cached_read(-1, 4)
        enc.close()

    def test_validates_negative_size(self):
        enc, _ = self._enc_and_oracle()
        with pytest.raises(ValueError, match="size must be >= 0"):
            enc.try_cached_read(0, -1)
        enc.close()

    def test_after_close_raises(self):
        enc, _ = self._enc_and_oracle()
        enc.close()
        with pytest.raises(RuntimeError, match="encoder closed"):
            enc.try_cached_read(0, 1)

    def test_advance_publishes_new_chunk(self):
        # After read() advances to chunk 2, try_cached_read snapshots
        # the new chunk — confirms the publish hook in _advance is
        # invoked on every step, not just the first.
        enc, oracle = self._enc_and_oracle()
        prefix = enc.prefix_size
        bpv = enc.bytes_per_variant
        chunk_bytes_total = self.CHUNK_SIZE * bpv
        enc.read(prefix, 1)  # chunk 0
        enc.read(prefix + chunk_bytes_total, 1)  # advance to chunk 1
        enc.read(prefix + 2 * chunk_bytes_total, 1)  # advance to chunk 2
        # Chunk 2 covers variants 8..11.
        result = enc.try_cached_read(prefix + 9 * bpv, bpv)
        assert result == oracle[prefix + 9 * bpv : prefix + 10 * bpv]
        # And chunk 0 is no longer cached.
        assert enc.try_cached_read(prefix, bpv) is None
        enc.close()

    def test_restart_clears_publication(self):
        # _teardown_iterator (called from _restart) must clear
        # _published_chunk; otherwise stale bytes could be served
        # after a restart but before the new chunk's encode finishes.
        enc, _ = self._enc_and_oracle()
        prefix = enc.prefix_size
        bpv = enc.bytes_per_variant
        enc.read(prefix, 1)  # chunk 0 loaded
        # _teardown_iterator clears the publication; verify directly.
        enc._teardown_iterator()
        assert enc._published_chunk is None
        # try_cached_read against the previously-cached range now
        # returns None.
        assert enc.try_cached_read(prefix, bpv) is None
        enc.close()


class TestWriteTo:
    """write_to(out, off, size) streams the encoded bytes through out's
    write(bytes) method. Defaults to the full stream; off/size limit
    to a sub-range with the same validation and EOF clamping as
    read."""

    NUM_VARIANTS = 20
    CHUNK_SIZE = 4
    BPV = 4

    def _enc_and_oracle(self):
        enc = _FakeEncoder(
            _build_reader(
                num_variants=self.NUM_VARIANTS,
                variants_chunk_size=self.CHUNK_SIZE,
            ),
            bpv=self.BPV,
        )
        oracle = _oracle(self.NUM_VARIANTS, self.BPV)
        return enc, oracle

    def test_default_writes_full_stream(self):
        enc, oracle = self._enc_and_oracle()
        buf = io.BytesIO()
        written = enc.write_to(buf)
        assert written == enc.total_size
        assert buf.getvalue() == oracle
        enc.close()

    def test_off_only_writes_from_offset_to_end(self):
        enc, oracle = self._enc_and_oracle()
        off = enc.prefix_size + 3 * self.BPV
        buf = io.BytesIO()
        written = enc.write_to(buf, off=off)
        assert written == enc.total_size - off
        assert buf.getvalue() == oracle[off:]
        enc.close()

    def test_size_only_writes_from_start(self):
        enc, oracle = self._enc_and_oracle()
        size = enc.prefix_size + 2 * self.BPV
        buf = io.BytesIO()
        written = enc.write_to(buf, size=size)
        assert written == size
        assert buf.getvalue() == oracle[:size]
        enc.close()

    def test_off_and_size_writes_sub_range(self):
        enc, oracle = self._enc_and_oracle()
        off = enc.prefix_size + 2 * self.BPV
        size = 5 * self.BPV
        buf = io.BytesIO()
        written = enc.write_to(buf, off=off, size=size)
        assert written == size
        assert buf.getvalue() == oracle[off : off + size]
        enc.close()

    def test_size_clamped_to_total(self):
        enc, oracle = self._enc_and_oracle()
        off = enc.total_size - 4
        buf = io.BytesIO()
        written = enc.write_to(buf, off=off, size=10_000)
        assert written == 4
        assert buf.getvalue() == oracle[off:]
        enc.close()

    def test_off_past_eof_writes_nothing(self):
        enc, _ = self._enc_and_oracle()
        buf = io.BytesIO()
        written = enc.write_to(buf, off=enc.total_size + 100)
        assert written == 0
        assert buf.getvalue() == b""
        enc.close()

    def test_size_zero_writes_nothing(self):
        enc, _ = self._enc_and_oracle()
        buf = io.BytesIO()
        written = enc.write_to(buf, size=0)
        assert written == 0
        assert buf.getvalue() == b""
        enc.close()

    def test_validates_negative_off(self):
        enc, _ = self._enc_and_oracle()
        with pytest.raises(ValueError, match="off must be >= 0"):
            enc.write_to(io.BytesIO(), off=-1)
        enc.close()

    def test_validates_negative_size(self):
        enc, _ = self._enc_and_oracle()
        with pytest.raises(ValueError, match="size must be >= 0"):
            enc.write_to(io.BytesIO(), size=-1)
        enc.close()

    def test_after_close_raises(self):
        enc, _ = self._enc_and_oracle()
        enc.close()
        with pytest.raises(RuntimeError, match="encoder closed"):
            enc.write_to(io.BytesIO())

    def test_writes_to_real_file(self, tmp_path):
        enc, oracle = self._enc_and_oracle()
        path = tmp_path / "out.bin"
        with open(path, "wb") as f:
            written = enc.write_to(f)
        assert written == enc.total_size
        assert path.read_bytes() == oracle
        enc.close()

    def test_block_size_uses_encode_block_bytes(self):
        # Force a tiny encode_block_bytes so write_to issues many
        # reads; assert the byte stream is still correct.
        enc = _FakeEncoder(
            _build_reader(
                num_variants=self.NUM_VARIANTS,
                variants_chunk_size=self.CHUNK_SIZE,
            ),
            bpv=self.BPV,
            encode_block_bytes=3,
        )
        oracle = _oracle(self.NUM_VARIANTS, self.BPV)
        buf = io.BytesIO()
        written = enc.write_to(buf)
        assert written == enc.total_size
        assert buf.getvalue() == oracle
        enc.close()


# ---------------------------------------------------------------------------
# Restart behaviour
# ---------------------------------------------------------------------------


class TestRestart:
    def test_forward_skip_past_loaded_chunk_restarts(self):
        enc = _FakeEncoder(
            _build_reader(num_variants=20, variants_chunk_size=4),
            bpv=4,
        )
        counter = _wrap_restart(enc)
        enc.read(enc.prefix_size, 1)  # loads chunk 0
        # Skip chunk 1 entirely: read in chunk 2
        enc.read(enc.prefix_size + 8 * 4, 4)
        assert counter["n"] == 2
        enc.close()

    def test_backward_jump_past_loaded_chunk_restarts(self):
        enc = _FakeEncoder(
            _build_reader(num_variants=20, variants_chunk_size=4),
            bpv=4,
        )
        counter = _wrap_restart(enc)
        enc.read(enc.prefix_size + 8 * 4, 4)  # loads chunk 2
        enc.read(enc.prefix_size + 4, 4)  # back to chunk 0
        assert counter["n"] == 2
        enc.close()

    def test_three_scattered_reads_three_restarts(self):
        enc = _FakeEncoder(
            _build_reader(num_variants=20, variants_chunk_size=4),
            bpv=4,
        )
        counter = _wrap_restart(enc)
        bpv = enc.bytes_per_variant
        prefix = enc.prefix_size
        enc.read(prefix + 16 * bpv, 4)  # chunk 4
        enc.read(prefix + 0 * bpv, 4)  # back to chunk 0
        enc.read(prefix + 8 * bpv, 4)  # forward to chunk 2
        assert counter["n"] == 3
        assert enc._restart_count == 2
        enc.close()

    def test_restart_after_failed_advance_emits_restart_not_init(self, caplog):
        """If a prior restart already succeeded (so ``_chunk_plan_pos > -1``)
        and a subsequent ``_advance`` call from a new restart raises in
        ``_encode_chunk``, the retry from a later read sees
        ``prev_plan_pos != -1`` and emits ``"restart #N"`` — not
        ``"iterator init"``. Locks in the state-machine invariant.
        """
        enc = _FakeEncoder(
            _build_reader(num_variants=20, variants_chunk_size=4),
            bpv=4,
        )
        prefix = enc.prefix_size
        bpv = enc.bytes_per_variant
        # First read loads chunk 0 successfully → _chunk_plan_pos = 0
        enc.read(prefix, 4)
        assert enc._chunk_plan_pos == 0

        # Force the encode of chunk 2 (variants [8:12]) to raise. Chunk-content-
        # based so the failure is robust against prefetch-induced extra encode
        # calls on chunks the consumer never observes.
        enc._fail_if_first_variant = 8
        with pytest.raises(ValueError, match="fake encode failure"):
            enc.read(prefix + 8 * bpv, 4)  # skip to chunk 2

        enc._fail_if_first_variant = None
        with caplog.at_level(logging.DEBUG, logger=_FakeEncoder.__module__):
            enc.read(prefix + 12 * bpv, 4)  # retry into chunk 3
        messages = [
            r.getMessage() for r in caplog.records if "_FakeEncoder" in r.getMessage()
        ]
        assert any("_FakeEncoder restart" in m for m in messages)
        assert not any("iterator init" in m for m in messages)
        enc.close()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestLogging:
    def test_first_read_emits_iterator_init_at_debug(self, caplog):
        enc = _FakeEncoder(_build_reader())
        with caplog.at_level(logging.DEBUG, logger=_FakeEncoder.__module__):
            enc.read(enc.prefix_size, 4)
        debug_msgs = [
            r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG
        ]
        assert any("_FakeEncoder iterator init" in m for m in debug_msgs)
        enc.close()

    def test_forced_restart_logs_at_info_with_plan_position(self, caplog):
        enc = _FakeEncoder(
            _build_reader(num_variants=20, variants_chunk_size=4),
            bpv=4,
        )
        prefix = enc.prefix_size
        bpv = enc.bytes_per_variant
        with caplog.at_level(logging.DEBUG, logger=_FakeEncoder.__module__):
            enc.read(prefix, 1)
            enc.read(prefix + 8 * bpv, 5)
        info_lines = [
            r.getMessage()
            for r in caplog.records
            if r.levelno == logging.INFO and "_FakeEncoder restart" in r.getMessage()
        ]
        assert len(info_lines) == 1
        assert "_FakeEncoder restart #1" in info_lines[0]
        assert "plan_pos=0 → 2" in info_lines[0]
        enc.close()

    def test_close_summary_only_when_restarts_happened(self, caplog):
        reader = _build_reader(num_variants=12, variants_chunk_size=4)
        # Drain sequentially: no restarts → no summary
        enc_a = _FakeEncoder(reader, bpv=4)
        with caplog.at_level(logging.DEBUG, logger=_FakeEncoder.__module__):
            enc_a.read(0, enc_a.total_size)
            enc_a.close()
        assert "_FakeEncoder closed" not in caplog.text

        caplog.clear()
        # Force a restart → close summary appears
        enc_b = _FakeEncoder(reader, bpv=4)
        prefix = enc_b.prefix_size
        bpv = enc_b.bytes_per_variant
        with caplog.at_level(logging.DEBUG, logger=_FakeEncoder.__module__):
            enc_b.read(prefix, 1)
            enc_b.read(prefix + 8 * bpv, 1)
            enc_b.close()
        close_lines = [
            r.getMessage()
            for r in caplog.records
            if "_FakeEncoder closed" in r.getMessage()
        ]
        assert len(close_lines) == 1
        assert "1 restarts" in close_lines[0]

    def test_log_record_name_is_subclass_module(self, caplog):
        enc = _FakeEncoder(_build_reader())
        with caplog.at_level(logging.DEBUG, logger=_FakeEncoder.__module__):
            enc.read(enc.prefix_size, 4)
        names = {r.name for r in caplog.records if "_FakeEncoder" in r.getMessage()}
        assert names == {_FakeEncoder.__module__}
        enc.close()


# ---------------------------------------------------------------------------
# Close / context manager
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_idempotent(self):
        enc = _FakeEncoder(_build_reader())
        enc.close()
        enc.close()  # second call must not raise
        assert enc._closed is True

    def test_close_tears_down_iterator(self):
        enc = _FakeEncoder(_build_reader())
        enc.read(enc.prefix_size, 4)  # bootstrap iterator
        assert enc._iterator is not None
        enc.close()
        assert enc._iterator is None
        assert enc._chunk_bytes is None
        assert enc._chunk_plan_pos == -1

    def test_close_shuts_down_executor(self):
        enc = _FakeEncoder(_build_reader())
        enc.close()
        assert enc._executor._shutdown is True

    def test_close_hook_called_once_on_first_close(self):
        enc = _FakeEncoder(_build_reader())
        enc.close()
        assert enc._close_hook_calls == 1
        enc.close()  # idempotent — must not re-invoke hook
        assert enc._close_hook_calls == 1

    def test_close_hook_called_when_no_reads_happened(self):
        enc = _FakeEncoder(_build_reader())
        enc.close()
        assert enc._close_hook_calls == 1

    def test_reader_usable_after_encoder_close(self):
        reader = _build_reader(num_variants=4)
        enc = _FakeEncoder(reader)
        enc.read(enc.prefix_size, 4)
        enc.close()
        # Reader's variant_chunks should still work after the encoder's close
        chunks = list(reader.variant_chunks(fields=["call_genotype"]))
        assert len(chunks) >= 1


class TestContextManager:
    def test_enter_returns_self(self):
        enc = _FakeEncoder(_build_reader())
        assert enc.__enter__() is enc
        enc.close()

    def test_exit_closes_on_normal_exit(self):
        with _FakeEncoder(_build_reader()) as enc:
            pass
        assert enc._closed is True
        assert enc._close_hook_calls == 1

    def test_exit_closes_and_propagates_exception(self):
        enc = _FakeEncoder(_build_reader())
        with pytest.raises(RuntimeError, match="body boom"):  # noqa: PT012
            with enc:
                raise RuntimeError("body boom")
        assert enc._closed is True


# ---------------------------------------------------------------------------
# Parallel encode helper
# ---------------------------------------------------------------------------


class TestParallelEncode:
    def test_threads_one_never_submits(self):
        enc = _FakeEncoder(
            _build_reader(num_variants=20, variants_chunk_size=4),
            bpv=4,
            use_parallel=True,
            encode_threads=1,
        )
        counter = _wrap_submit(enc)
        enc.read(0, enc.total_size)
        assert counter["n"] == 0
        enc.close()

    def test_below_threshold_never_submits(self):
        enc = _FakeEncoder(
            _build_reader(num_variants=20, variants_chunk_size=4),
            bpv=4,
            use_parallel=True,
            encode_threads=4,
            encode_block_bytes=10 * 1024 * 1024,  # well above 4 vars * 4 bpv = 16
        )
        counter = _wrap_submit(enc)
        enc.read(0, enc.total_size)
        assert counter["n"] == 0
        enc.close()

    def test_above_threshold_submits_and_output_matches_sequential(self):
        reader = _build_reader(num_variants=20, variants_chunk_size=8)
        sequential = _FakeEncoder(reader, bpv=4, use_parallel=False)
        sequential_out = _drain(sequential, 1 << 17)
        sequential.close()

        parallel = _FakeEncoder(
            reader,
            bpv=4,
            use_parallel=True,
            encode_threads=4,
            encode_block_bytes=4,  # forces fan-out for 8-variant chunks
        )
        counter = _wrap_submit(parallel)
        parallel_out = _drain(parallel, 1 << 17)
        assert counter["n"] > 0
        assert parallel_out == sequential_out
        parallel.close()

    def test_remainder_block_assembly(self):
        # 10 variants / chunk size 4 => last chunk has 2 variants, not 4.
        # encode_block_bytes=4 (= 1 variant) forces blocks of 1 variant.
        enc = _FakeEncoder(
            _build_reader(num_variants=10, variants_chunk_size=4),
            bpv=4,
            use_parallel=True,
            encode_threads=4,
            encode_block_bytes=4,
        )
        out = _drain(enc, 1 << 17)
        assert out == _oracle(10, 4)
        enc.close()

    def test_tiny_block_bytes_floors_block_variants_to_one(self):
        # encode_block_bytes=1, bpv=100 ⇒ 1//100 = 0 ⇒ max(1, 0) = 1.
        # Output must still match the sequential reference byte-for-byte.
        reader = _build_reader(num_variants=12, variants_chunk_size=4)
        sequential = _FakeEncoder(reader, bpv=100, use_parallel=False)
        sequential_out = _drain(sequential, 1 << 17)
        sequential.close()

        parallel = _FakeEncoder(
            reader,
            bpv=100,
            use_parallel=True,
            encode_threads=4,
            encode_block_bytes=1,
        )
        assert _drain(parallel, 1 << 17) == sequential_out
        parallel.close()

    def test_exception_in_worker_propagates(self):
        enc = _FakeEncoder(
            _build_reader(num_variants=8, variants_chunk_size=4),
            bpv=4,
            use_parallel=True,
            encode_threads=4,
            encode_block_bytes=4,  # force fan-out
        )
        enc._encode_raise_in_worker = True
        with pytest.raises(RuntimeError, match="worker boom"):
            enc.read(0, enc.total_size)
        enc.close()


# ---------------------------------------------------------------------------
# Subclass hook contract
# ---------------------------------------------------------------------------


class TestSubclassHookContract:
    def test_encode_chunk_called_once_per_chunk(self):
        enc = _FakeEncoder(
            _build_reader(num_variants=12, variants_chunk_size=4),
            bpv=4,
        )
        enc.read(0, enc.total_size)  # 3 chunks
        assert enc._encode_calls == 3
        enc.close()

    def test_prefix_only_reads_do_not_call_encode_chunk(self):
        enc = _FakeEncoder(
            _build_reader(num_variants=12, variants_chunk_size=4),
            bpv=4,
            prefix=b"HEADERBYTES",
        )
        enc.read(0, 3)  # within prefix
        enc.read(2, 5)  # within prefix
        assert enc._encode_calls == 0
        enc.close()

    def test_encode_chunk_receives_iterator_fields(self):
        enc = _FakeEncoder(_build_reader(num_variants=4, variants_chunk_size=4))
        enc.read(0, enc.total_size)
        assert enc._chunks_received[0] == ("call_genotype",)
        enc.close()

    def test_default_close_hook_is_no_op(self):
        class _NoHookEncoder(format_encoder.FormatEncoder):
            def __init__(self, reader):
                super().__init__(
                    reader,
                    bytes_per_variant=1,
                    prefix_bytes=b"",
                    iterator_fields=["call_genotype"],
                )

            def _encode_chunk(self, chunk):
                return b"\x00" * chunk["call_genotype"].shape[0]

        enc = _NoHookEncoder(_build_reader(num_variants=4))
        enc.close()  # default _close_hook is a pass; must not raise


# ---------------------------------------------------------------------------
# Reader passthrough
# ---------------------------------------------------------------------------


class TestReaderPassthrough:
    def test_set_variants_reflected_in_total_size(self):
        reader = _build_reader(num_variants=12, variants_chunk_size=4)
        reader.set_variants(np.array([0, 2, 4, 6], dtype=np.int64))
        enc = _FakeEncoder(reader, bpv=4)
        assert enc.num_variants == 4
        assert enc.total_size == enc.prefix_size + 4 * 4
        enc.close()

    def test_set_samples_reflected_in_num_samples(self):
        reader = _build_reader(num_variants=4, num_samples=5)
        reader.set_samples([0, 2])
        enc = _FakeEncoder(reader)
        assert enc.num_samples == 2
        enc.close()

    def test_encoder_does_not_close_reader(self):
        reader = _build_reader(num_variants=4)
        enc = _FakeEncoder(reader)
        enc.close()
        # reader.sample_ids should still be accessible
        assert reader.sample_ids.size > 0

    def test_two_encoders_share_one_reader_independently(self):
        reader = _build_reader(num_variants=12, variants_chunk_size=4)
        e1 = _FakeEncoder(reader, bpv=4)
        e2 = _FakeEncoder(reader, bpv=4)
        out1 = _drain(e1, 17)
        out2 = _drain(e2, 17)
        assert out1 == out2 == _oracle(12, 4)
        e1.close()
        e2.close()


# ---------------------------------------------------------------------------
# Empty store
# ---------------------------------------------------------------------------


class TestEmptyStore:
    def _zero_variant_reader(self):
        # vcz_builder.make_vcz infers region_index shape from row count,
        # so for 0 variants we pass an explicit (0, 6) array — same
        # pattern as test_bgen.py / test_plink.py.
        root = vcz_builder.make_vcz(
            variant_contig=np.zeros(0, dtype=np.int32),
            variant_position=np.zeros(0, dtype=np.int32),
            alleles=np.zeros((0, 2), dtype="<U16"),
            num_samples=3,
            sample_id=["s0", "s1", "s2"],
            call_genotype=np.zeros((0, 3, 2), dtype=np.int8),
            region_index=np.zeros((0, 6), dtype=np.int32),
        )
        return retrieval.VczReader(root)

    def test_total_size_equals_prefix_size(self):
        enc = _FakeEncoder(self._zero_variant_reader(), prefix=b"AB", bpv=4)
        assert enc.num_variants == 0
        assert enc.total_size == 2
        enc.close()

    def test_read_returns_prefix_then_empty(self):
        enc = _FakeEncoder(self._zero_variant_reader(), prefix=b"AB", bpv=4)
        assert enc.read(0, 2) == b"AB"
        assert enc.read(2, 10) == b""
        enc.close()

    def test_prefixless_empty_store_reads_return_empty(self):
        enc = _FakeEncoder(self._zero_variant_reader(), prefix=b"", bpv=4)
        assert enc.total_size == 0
        assert enc.read(0, 100) == b""
        assert enc.read(0, 0) == b""
        enc.close()

    def test_close_works_without_reads(self):
        enc = _FakeEncoder(self._zero_variant_reader())
        enc.close()
        assert enc._closed is True


# ---------------------------------------------------------------------------
# Encode readahead (PrefetchIterator wrap)
# ---------------------------------------------------------------------------


def _prefetch_threads():
    return [t for t in threading.enumerate() if "vcztools-prefetch" in t.name]


def _wait_for_thread_count(target, timeout=1.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if len(_prefetch_threads()) <= target:
            return
        time.sleep(0.01)


class TestReadahead:
    """The encoded-chunk iterator is wrapped in
    :class:`utils.PrefetchIterator`, so ``_encode_chunk`` runs on a
    background prefetch worker one chunk ahead of the consumer.
    """

    def test_iterator_is_wrapped_after_first_read(self):
        enc = _FakeEncoder(_build_reader(num_variants=20, variants_chunk_size=4))
        try:
            enc.read(enc.prefix_size, 4)
            assert isinstance(enc._iterator, utils.PrefetchIterator)
        finally:
            enc.close()

    def test_encode_runs_on_prefetch_worker_not_caller(self):
        # Drain the full stream, then assert every _encode_chunk call
        # ran on a vcztools-prefetch-named thread, never on MainThread.
        enc = _FakeEncoder(_build_reader(num_variants=20, variants_chunk_size=4))
        try:
            _drain(enc, step=enc.bytes_per_variant)
            assert len(enc._encode_thread_names) >= 5  # 20 variants / 4 = 5 chunks
            assert all("vcztools-prefetch" in name for name in enc._encode_thread_names)
            assert not any("MainThread" in name for name in enc._encode_thread_names)
        finally:
            enc.close()

    def test_worker_exception_propagates_to_consumer_read(self):
        # An encode failure on chunk N surfaces on the consumer's read()
        # that crosses chunk N — the PrefetchIterator stashes the
        # worker exception in its future and re-raises on next().
        enc = _FakeEncoder(_build_reader(num_variants=20, variants_chunk_size=4), bpv=4)
        try:
            prefix = enc.prefix_size
            bpv = enc.bytes_per_variant
            # First read loads chunk 0 successfully.
            enc.read(prefix, 4)
            # Arm a chunk-content failure for chunk 2 (variants [8:12]).
            enc._fail_if_first_variant = 8
            with pytest.raises(ValueError, match="fake encode failure"):
                # Drain forward across chunks 1 and 2; chunk 2 raises.
                _drain(enc, step=bpv * 2)
        finally:
            # close() must still join cleanly even after a worker
            # exception — proves no hang on teardown.
            enc.close()

    def test_close_joins_prefetch_worker(self):
        before = len(_prefetch_threads())
        enc = _FakeEncoder(_build_reader(num_variants=20, variants_chunk_size=4))
        enc.read(enc.prefix_size, 4)  # spawn prefetch worker via _restart
        enc.close()
        _wait_for_thread_count(before)
        assert len(_prefetch_threads()) <= before

    def test_no_thread_leak_across_many_construct_close_cycles(self):
        # Build and tear down 100 encoders; the prefetch worker count
        # must return to baseline after each close.
        before = len(_prefetch_threads())
        for _ in range(100):
            enc = _FakeEncoder(_build_reader(num_variants=8, variants_chunk_size=4))
            enc.read(enc.prefix_size, 4)
            enc.close()
        _wait_for_thread_count(before, timeout=2.0)
        assert len(_prefetch_threads()) <= before
