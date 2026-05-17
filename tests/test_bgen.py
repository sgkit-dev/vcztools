"""
Unit tests for vcztools.bgen.

These exercise the encoder layers (header, sample-id block, genotype
block, variant block, .sample text, .bgi index, end-to-end write_bgen)
directly against in-memory VCZ groups built with
:func:`tests.vcz_builder.make_vcz`.

Round-trip checks parse the bytes produced by ``write_bgen`` with the
``bgen-reader`` reference reader.
"""

import io
import logging
import sqlite3
import struct
import zlib

import bgen_reader as br
import numpy as np
import numpy.testing as nt
import pytest

from tests import vcz_builder
from vcztools import _vcztools, bcftools_filter, bgen, regions, retrieval


def _build_geno_block_reference(g, variant_phased):
    """Pure-Python reference for one BGEN Layout-2 genotype block.

    ``g`` has shape ``(num_samples, 2)`` int8; ``variant_phased`` is a
    bool. Returns the raw bytes (variable length under mixed ploidy).

    Sentinel rules per sample ``(a, b)``. Only ``{-2, -1, 0, 1}`` are
    accepted; any other value raises ``ValueError`` (BGEN is biallelic).

      * ``a in {0, 1}, b in {0, 1}``  -> diploid call, 0x02, 2 prob bytes
      * ``a in {0, 1}, b == -2``      -> haploid call, 0x01, 1 prob byte
      * ``a == -1, b == -1``          -> missing diploid, 0x82, 2 zero bytes
      * ``a == -1, b == -2``          -> missing haploid, 0x81, 1 zero byte
      * half-missing diploid          -> 0x82, 2 zero bytes
      * ``a == -2`` (any b)           -> raises ValueError (zero-ploidy)
      * any other value               -> raises ValueError (invalid allele)

    Block layout:

        uint32 N            n_samples
        uint16 K = 2        n_alleles
        uint8  P_min        actual min ploidy in this variant
        uint8  P_max        actual max ploidy in this variant
        N bytes             ploidy/missing per sample
        uint8  phased       per-variant flag
        uint8  B = 8        bits per probability
        sum_s K_s bytes     per-sample probability bytes
    """
    num_samples = g.shape[0]
    ploidy_bytes = bytearray()
    prob_bytes = bytearray()
    pmin, pmax = 2, 1

    for s in range(num_samples):
        a = int(g[s, 0])
        b = int(g[s, 1])
        if a not in (-2, -1, 0, 1) or b not in (-2, -1, 0, 1):
            raise ValueError(f"reference: invalid allele at sample {s}: ({a}, {b})")
        if a == -2:
            raise ValueError(f"reference: -2 in slot 0 at sample {s}")
        if b == -2:
            if a == -1:
                ploidy_bytes.append(0x81)
                prob_bytes.append(0x00)
            else:
                ploidy_bytes.append(0x01)
                prob_bytes.append(0xFF if a == 0 else 0x00)
            pmin = min(pmin, 1)
        elif a < 0 or b < 0:
            ploidy_bytes.append(0x82)
            prob_bytes.extend([0x00, 0x00])
            pmax = max(pmax, 2)
        else:
            ploidy_bytes.append(0x02)
            if variant_phased:
                prob_bytes.append(0xFF if a == 0 else 0x00)
                prob_bytes.append(0xFF if b == 0 else 0x00)
            else:
                prob_bytes.append(0xFF if (a == 0 and b == 0) else 0x00)
                het = (a == 0 and b == 1) or (a == 1 and b == 0)
                prob_bytes.append(0xFF if het else 0x00)
            pmax = max(pmax, 2)

    if num_samples == 0:
        pmin, pmax = 2, 2

    row = bytearray()
    row.extend(struct.pack("<IHBB", num_samples, 2, pmin, pmax))
    row.extend(ploidy_bytes)
    row.append(1 if variant_phased else 0)
    row.append(bgen.BITS_PER_PROB)
    row.extend(prob_bytes)
    return bytes(row)


def _build_geno_blocks_reference(G, phased):
    """List of per-variant byte blocks; one call to
    :func:`_build_geno_block_reference` per variant."""
    blocks = []
    for v in range(G.shape[0]):
        blocks.append(_build_geno_block_reference(G[v], bool(phased[v])))
    return blocks


def _build_reader(*, num_variants=2, num_samples=3, **overrides):
    defaults = dict(
        variant_contig=[0] * num_variants,
        variant_position=list(range(100, 100 + num_variants)),
        alleles=[("A", "T")] * num_variants,
        num_samples=num_samples,
        call_genotype=np.zeros((num_variants, num_samples, 2), dtype=np.int8),
    )
    defaults.update(overrides)
    root = vcz_builder.make_vcz(**defaults)
    return retrieval.VczReader(root)


# ---------------------------------------------------------------------------
# Sidecar generation
# ---------------------------------------------------------------------------


def _sample_string(reader):
    buf = io.StringIO()
    bgen.write_sample(reader, buf)
    return buf.getvalue()


class TestWriteSample:
    def test_minimal(self):
        reader = _build_reader(num_samples=3)
        text = _sample_string(reader)
        lines = text.splitlines()
        assert lines[0] == "ID_1 ID_2 missing"
        assert lines[1] == "0 0 0"
        assert lines[2] == "sample_0 sample_0 0"
        assert lines[3] == "sample_1 sample_1 0"
        assert lines[4] == "sample_2 sample_2 0"
        assert text.endswith("\n")

    def test_zero_samples(self):
        reader = _build_reader(num_samples=0)
        assert _sample_string(reader) == "ID_1 ID_2 missing\n0 0 0\n"

    def test_whitespace_rejected(self):
        reader = _build_reader(
            num_samples=2,
            sample_id=np.array(["ok", "has space"], dtype="<U16"),
        )
        with pytest.raises(ValueError, match="whitespace"):
            bgen.write_sample(reader, io.StringIO())

    def test_sample_subset_round_trips(self):
        reader = _build_reader(num_samples=4)
        reader.set_samples([0, 2])
        text = _sample_string(reader)
        lines = text.splitlines()
        assert lines[2] == "sample_0 sample_0 0"
        assert lines[3] == "sample_2 sample_2 0"
        assert len(lines) == 4

    def test_path_output_matches_buffer(self, tmp_path):
        reader = _build_reader(num_samples=3)
        out = tmp_path / "x.sample"
        bgen.write_sample(reader, out)
        assert out.read_text() == _sample_string(reader)


class TestBuildSampleIdBlock:
    def test_layout(self):
        block = bgen._build_sample_id_block(["alice", "bob"])
        # L_SI (4) + N (4) + 2 * (uint16 + utf8)
        expected = (
            struct.pack("<II", 4 + 4 + 2 + 5 + 2 + 3, 2)
            + struct.pack("<H", 5)
            + b"alice"
            + struct.pack("<H", 3)
            + b"bob"
        )
        assert block == expected

    def test_empty(self):
        block = bgen._build_sample_id_block([])
        assert block == struct.pack("<II", 8, 0)

    def test_unicode(self):
        # UTF-8: id length is bytes, not characters.
        block = bgen._build_sample_id_block(["Ω"])
        body = "Ω".encode()
        assert len(body) == 2
        expected = struct.pack("<II", 8 + 2 + 2, 1) + struct.pack("<H", 2) + body
        assert block == expected

    def test_oversized_id_raises(self):
        big = "x" * 70000
        with pytest.raises(ValueError, match="65535"):
            bgen._build_sample_id_block([big])


class TestBuildHeader:
    def test_layout(self):
        sample_block = bgen._build_sample_id_block(["s1"])
        header = bgen._build_header(
            num_variants=10, num_samples=1, sample_id_block=sample_block
        )
        # First 4 bytes: offset = HEADER_LENGTH + len(sample_block)
        (offset,) = struct.unpack_from("<I", header, 0)
        assert offset == bgen.HEADER_LENGTH + len(sample_block)
        # Next 20 bytes: header block.
        (h_len, m, n) = struct.unpack_from("<III", header, 4)
        assert h_len == bgen.HEADER_LENGTH
        assert m == 10
        assert n == 1
        assert header[16:20] == b"bgen"
        (flags,) = struct.unpack_from("<I", header, 20)
        assert flags & 0b11 == bgen.COMPRESSION_ZLIB
        assert (flags >> 2) & 0xF == bgen.LAYOUT_2
        assert flags & bgen.SAMPLE_IDS_PRESENT
        # Sample-id block follows.
        assert header[24:] == sample_block

    def test_no_embedded_samples(self):
        header = bgen._build_header(
            num_variants=10,
            num_samples=2,
            sample_id_block=b"",
            embed_samples=False,
        )
        # No sample-id block: offset = HEADER_LENGTH and the header
        # ends at byte 24.
        (offset,) = struct.unpack_from("<I", header, 0)
        assert offset == bgen.HEADER_LENGTH
        assert len(header) == 4 + bgen.HEADER_LENGTH
        (flags,) = struct.unpack_from("<I", header, 20)
        assert flags & 0b11 == bgen.COMPRESSION_ZLIB
        assert (flags >> 2) & 0xF == bgen.LAYOUT_2
        assert (flags & bgen.SAMPLE_IDS_PRESENT) == 0


class TestBgenVariantBlockSize:
    """:func:`vcztools._vcztools.bgen_variant_block_size` exposes the
    C kernel's exact per-variant byte count to Python, so the encoder
    never has to reimplement the size formula.

    Two layers of coverage:

    1. :func:`test_matches_kernel_emit` — drives the chunk-slice C
       kernel for one variant at boundary sizes (where Python's
       ``zlib.compress(_, 0)`` over-predicts by 5 bytes) and asserts
       the bytes the kernel wrote line up with what
       ``bgen_variant_block_size`` predicted.
    2. :func:`test_round_trips_through_python_zlib_decompress` —
       extracts the kernel's stored-zlib payload and round-trips it
       through ``zlib.decompress``, locking the kernel's stream to
       "valid zlib" status at the same boundary sizes.
    3. :func:`test_invalid_uniform_ploidy_rejected` — covers the
       ``uniform_ploidy ∈ {1, 2}`` validation in the wrapper.
    """

    def _emit_chunk_slice(self, num_samples, uniform_ploidy):
        """Drive the chunk-slice C kernel for one variant at
        ``num_samples`` × ``uniform_ploidy``; return the raw output
        bytes plus the per-variant offsets needed to walk the wire."""
        num_variants = 1
        varid_max = rsid_max = chrom_max = 1
        allele_max = 1
        geno_size = 10 + (uniform_ploidy + 1) * num_samples
        G = np.zeros((num_variants, num_samples, 2), dtype=np.int8)
        if uniform_ploidy == 1:
            # Mark every sample haploid via the -2 sentinel in slot 1.
            G[:, :, 1] = -2
        phased = np.zeros(num_variants, dtype=bool)
        varid = np.zeros((num_variants, varid_max), dtype=np.uint8)
        rsid = np.zeros((num_variants, rsid_max), dtype=np.uint8)
        chrom = np.zeros((num_variants, chrom_max), dtype=np.uint8)
        allele1 = np.zeros((num_variants, allele_max), dtype=np.uint8)
        allele2 = np.zeros((num_variants, allele_max), dtype=np.uint8)
        position = np.array([100], dtype=np.int32)
        bpv = _vcztools.bgen_variant_block_size(
            num_samples=num_samples,
            uniform_ploidy=uniform_ploidy,
            varid_max=varid_max,
            rsid_max=rsid_max,
            chrom_max=chrom_max,
            allele_max=allele_max,
        )
        out = np.zeros(bpv, dtype=np.uint8)
        _vcztools.encode_bgen_chunk_slice_level0(
            varid,
            rsid,
            chrom,
            allele1,
            allele2,
            position,
            G,
            phased,
            out,
            uniform_ploidy,
        )
        header_overhead = (
            2
            + varid_max
            + 2
            + rsid_max
            + 2
            + chrom_max
            + 4
            + 2
            + 4
            + allele_max
            + 4
            + allele_max
        )
        return bytes(out), header_overhead, geno_size, bpv

    @pytest.mark.parametrize(
        ("num_samples", "uniform_ploidy"),
        [
            (1, 2),  # geno_size = 13
            (4, 2),  # geno_size = 22 (small single-block)
            (21841, 2),  # geno_size = 65533 (boundary band, diploid)
            (21844, 2),  # geno_size = 65542 (just past boundary)
            (32761, 1),  # geno_size = 65532 (boundary band, haploid)
            (32762, 1),  # geno_size = 65534 (boundary band, haploid)
            (22000, 2),  # geno_size = 66010 (just past boundary, 2 blocks)
            (43690, 2),  # geno_size = 131080 (multi-block)
        ],
        ids=lambda v: str(v),
    )
    def test_matches_kernel_emit(self, num_samples, uniform_ploidy):
        # The chunk-slice kernel writes exactly bytes_per_variant
        # bytes per variant. Read the per-variant C (payload length)
        # field straight off the wire and assert that the kernel's
        # declared variant-block tail matches what
        # bgen_variant_block_size predicted.
        wire, header_overhead, _geno_size, bpv = self._emit_chunk_slice(
            num_samples, uniform_ploidy
        )
        assert len(wire) == bpv
        (C,) = struct.unpack_from("<I", wire, header_overhead)
        # The C field is the byte count from the start of the D field
        # to the end of the payload; bpv = header_overhead + 4 + C.
        assert header_overhead + 4 + C == bpv

    @pytest.mark.parametrize(
        ("num_samples", "uniform_ploidy"),
        [
            (21841, 2),
            (32761, 1),
            (32762, 1),
            (43690, 2),
            (1, 2),
        ],
        ids=lambda v: str(v),
    )
    def test_round_trips_through_python_zlib_decompress(
        self, num_samples, uniform_ploidy
    ):
        # The C kernel writes its own zlib stored-DEFLATE stream
        # (different block layout from Python's zlib.compress, same
        # algorithm). Python's zlib.decompress must accept that
        # stream, recover exactly D bytes of uncompressed geno block,
        # and the recovered bytes must equal what the geno-block
        # builder produced for the same input. Locks the kernel's
        # stream to "valid zlib" status at the boundary sizes.
        wire, header_overhead, geno_size, _bpv = self._emit_chunk_slice(
            num_samples, uniform_ploidy
        )
        (C,) = struct.unpack_from("<I", wire, header_overhead)
        (D,) = struct.unpack_from("<I", wire, header_overhead + 4)
        assert D == geno_size
        payload = wire[header_overhead + 8 : header_overhead + 4 + C]
        unpacked = zlib.decompress(payload)
        assert len(unpacked) == geno_size
        # Cross-check the recovered bytes against the geno-block
        # builder's reference output for the same all-zero inputs.
        G = np.zeros((1, num_samples, 2), dtype=np.int8)
        if uniform_ploidy == 1:
            G[:, :, 1] = -2
        phased = np.zeros(1, dtype=bool)
        ref_blocks, ref_lens = _vcztools.encode_bgen_geno_blocks(G, phased)
        assert int(ref_lens[0]) == geno_size
        assert unpacked == bytes(ref_blocks[0, :geno_size])


# ---------------------------------------------------------------------------
# Genotype block / variant block byte-level encoding
# ---------------------------------------------------------------------------


def _single_variant_geno_block(G_single, *, phased=False):
    """Build a one-variant chunk, run `_prepare_chunk`, and return the
    uncompressed genotype-block bytes for that variant, trimmed to the
    actual length (variable under mixed-ploidy). ``G_single`` has shape
    ``(num_samples, 2)`` or ``(num_samples, 1)``."""
    G = G_single[None, ...]
    n = G_single.shape[0]
    chunk = {
        "call_genotype": G,
        "variant_allele": np.array([["A", "T"]]),
        "variant_contig": np.array([0]),
        "variant_position": np.array([100]),
    }
    if phased:
        chunk["call_genotype_phased"] = np.ones((1, n), dtype=bool)
    chunk_strings = bgen._prepare_chunk_strings(chunk, contig_ids=np.array(["chr1"]))
    prep = bgen._prepare_chunk(chunk, chunk_strings, start=0, end=1)
    geno_blocks, geno_block_lens = _vcztools.encode_bgen_geno_blocks(
        prep.genotypes, prep.phased
    )
    block_len = int(geno_block_lens[0])
    return bytes(geno_blocks[0, :block_len])


class TestPrepareChunkGenoBlocks:
    """Spec-level byte assertions on the uncompressed genotype-block
    bytes produced by ``_vcztools.encode_bgen_geno_blocks`` (the same
    geno-block builder that BgenEncoder's C kernel runs internally).
    Layout 2 / 8-bit / biallelic with per-sample ploidy."""

    def test_unphased_basic(self):
        # Three samples: hom-ref, het, hom-alt.
        G = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.int8)
        block = _single_variant_geno_block(G)
        # uint32 N=3, uint16 K=2, uint8 P_min=2, uint8 P_max=2 -> 8 bytes
        (n, k, p_min, p_max) = struct.unpack_from("<IHBB", block, 0)
        assert (n, k, p_min, p_max) == (3, 2, 2, 2)
        # 3 ploidy bytes, all 0x02 (diploid, not missing)
        assert block[8:11] == bytes([0x02, 0x02, 0x02])
        # phased=0, B=8
        assert block[11] == 0
        assert block[12] == 8
        # 6 prob bytes:
        # hom-ref: P(00)=0xFF, P(01)=0x00
        # het:     P(00)=0x00, P(01)=0xFF
        # hom-alt: P(00)=0x00, P(01)=0x00
        assert block[13:19] == bytes([0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00])

    def test_unphased_het_reverse_order(self):
        G = np.array([[1, 0]], dtype=np.int8)
        block = _single_variant_geno_block(G)
        # (1,0) is also het: P(00)=0, P(01)=0xFF.
        assert block[-2:] == bytes([0x00, 0xFF])

    def test_missing_genotype(self):
        G = np.array([[-1, -1], [0, -1], [0, 1]], dtype=np.int8)
        block = _single_variant_geno_block(G)
        # Ploidy bytes: 0x82 (missing), 0x82 (any neg → missing), 0x02.
        assert block[8:11] == bytes([0x82, 0x82, 0x02])
        # Probability bytes for missing samples are zeroed; het stays.
        assert block[13:19] == bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0xFF])

    def test_haploid_padding_emits_ploidy_one(self):
        # Single haploid sample (allele 0): one ploidy byte (0x01),
        # one probability byte (0xFF for ref). Total block length:
        # 8 header + 1 ploidy + 2 flags + 1 prob = 12.
        G = np.array([[0, -2]], dtype=np.int8)
        block = _single_variant_geno_block(G)
        assert len(block) == 12
        (n, k, p_min, p_max) = struct.unpack_from("<IHBB", block, 0)
        assert (n, k, p_min, p_max) == (1, 2, 1, 1)
        assert block[8] == 0x01
        assert block[9] == 0
        assert block[10] == 8
        assert block[11] == 0xFF

    def test_haploid_padding_missing_allele(self):
        # Haploid sample with missing allele: ploidy = 0x81, one zero
        # probability byte.
        G = np.array([[-1, -2]], dtype=np.int8)
        block = _single_variant_geno_block(G)
        assert len(block) == 12
        (n, k, p_min, p_max) = struct.unpack_from("<IHBB", block, 0)
        assert (n, k, p_min, p_max) == (1, 2, 1, 1)
        assert block[8] == 0x81
        assert block[11] == 0x00

    def test_mixed_ploidy_within_variant(self):
        # Sample 0 diploid (0, 0), sample 1 haploid (1, -2).
        # Block length: 8 + 2 ploidy + 2 flags + (2 + 1) prob = 15.
        # Pmin = 1, Pmax = 2.
        G = np.array([[0, 0], [1, -2]], dtype=np.int8)
        block = _single_variant_geno_block(G)
        assert len(block) == 15
        (n, k, p_min, p_max) = struct.unpack_from("<IHBB", block, 0)
        assert (n, k, p_min, p_max) == (2, 2, 1, 2)
        assert block[8:10] == bytes([0x02, 0x01])
        # 2 prob bytes for diploid (0,0)=homref then 1 byte for haploid
        # allele 1.
        assert block[12:15] == bytes([0xFF, 0x00, 0x00])

    def test_haploid_shape_one(self):
        # Shape (V, S, 1) input is promoted by _prepare_chunk to the
        # -2-padded form before reaching the C kernel.
        G = np.array([[0], [1], [-1]], dtype=np.int8)
        block = _single_variant_geno_block(G)
        # 3 haploid samples: 8 + 3 ploidy + 2 flags + 3 prob = 16.
        assert len(block) == 16
        (n, k, p_min, p_max) = struct.unpack_from("<IHBB", block, 0)
        assert (n, k, p_min, p_max) == (3, 2, 1, 1)
        assert block[8:11] == bytes([0x01, 0x01, 0x81])
        assert block[13:16] == bytes([0xFF, 0x00, 0x00])

    def test_zero_ploidy_raises(self):
        # -2 in slot 0 has no BGEN representation; surface as an error.
        G = np.array([[-2, 0]], dtype=np.int8)
        with pytest.raises(ValueError, match="zero-ploidy"):
            _single_variant_geno_block(G)

    def test_phased(self):
        # Phased het variants distinguish (0,1) from (1,0).
        G = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int8)
        block = _single_variant_geno_block(G, phased=True)
        # Header (8 bytes) + 4 ploidy bytes -> phased flag at offset 12,
        # B at 13, prob bytes start at 14.
        assert block[12] == 1
        assert block[13] == 8
        # Per haplotype, B0=P(allele 0|h1), B1=P(allele 0|h2):
        # (0,0) -> [0xFF, 0xFF]
        # (0,1) -> [0xFF, 0x00]
        # (1,0) -> [0x00, 0xFF]
        # (1,1) -> [0x00, 0x00]
        assert block[14:22] == bytes([0xFF, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00])

    def test_zero_samples(self):
        G = np.zeros((0, 2), dtype=np.int8)
        block = _single_variant_geno_block(G)
        # Header (8 bytes) + 0 ploidy bytes + phased + B (2 bytes) + 0 probs
        assert len(block) == 10
        (n, k, p_min, p_max) = struct.unpack_from("<IHBB", block, 0)
        assert (n, k, p_min, p_max) == (0, 2, 2, 2)


class TestBuildGenoBlocksAgainstReference:
    """Sweep the C kernel ``_vcztools.encode_bgen_geno_blocks`` against
    the pure-Python ``_build_geno_blocks_reference`` over a range of
    awkward inputs; the C output must match byte-for-byte. Inputs may
    include any of: diploid calls, haploid (``b == -2``), missing
    diploid, missing haploid, half-missing diploid. ``a == -2`` is the
    zero-ploidy error case and is exercised separately."""

    def _assert_matches(self, G, phased):
        c_buf, c_lens = _vcztools.encode_bgen_geno_blocks(G, phased)
        ref_blocks = _build_geno_blocks_reference(G, phased)
        assert c_buf.dtype == np.uint8
        assert c_lens.dtype == np.uint32
        assert c_buf.shape == (G.shape[0], 10 + 3 * G.shape[1])
        assert len(ref_blocks) == G.shape[0]
        for v in range(G.shape[0]):
            block_len = int(c_lens[v])
            assert block_len == len(ref_blocks[v])
            assert bytes(c_buf[v, :block_len]) == ref_blocks[v]

    @pytest.mark.parametrize("phased_value", [False, True])
    def test_single_sample_all_allele_pairs(self, phased_value):
        # Sweep every valid (a, b) from {-1, 0, 1} x {-2, -1, 0, 1}
        # as separate variants. ``a == -2`` is the zero-ploidy error
        # and out-of-range values are invalid-allele errors; see the
        # dedicated test_*_raises tests below.
        pairs = [(a, b) for a in (-1, 0, 1) for b in (-2, -1, 0, 1)]
        G = np.array([[[a, b]] for a, b in pairs], dtype=np.int8)
        phased = np.full(len(pairs), phased_value, dtype=bool)
        self._assert_matches(G, phased)

    def test_small_unphased_handcrafted(self):
        G = np.array([[[0, 0], [0, 1], [1, 1]]], dtype=np.int8)
        self._assert_matches(G, np.array([False]))

    def test_small_phased_handcrafted(self):
        G = np.array([[[0, 0], [0, 1], [1, 0], [1, 1]]], dtype=np.int8)
        self._assert_matches(G, np.array([True]))

    def test_mixed_phase_across_variants(self):
        single = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.int8)
        G = np.broadcast_to(single, (4, 3, 2)).copy()
        phased = np.array([True, False, True, False])
        self._assert_matches(G, phased)

    def test_all_missing(self):
        G = np.full((5, 7, 2), -1, dtype=np.int8)
        self._assert_matches(G, np.zeros(5, dtype=bool))
        self._assert_matches(G, np.ones(5, dtype=bool))

    def test_all_haploid(self):
        # All-haploid block: every sample is (allele, -2). Both phased
        # and unphased produce identical single-byte probabilities for
        # K=1 biallelic.
        v, s = 4, 6
        G = np.empty((v, s, 2), dtype=np.int8)
        G[:, :, 0] = np.arange(v * s, dtype=np.int8).reshape(v, s) % 3 - 1
        G[:, :, 1] = -2
        self._assert_matches(G, np.zeros(v, dtype=bool))
        self._assert_matches(G, np.ones(v, dtype=bool))

    def test_partial_ploidy_patterns(self):
        # One variant covers every valid (a, b) shape: diploid, haploid,
        # missing diploid, missing haploid, half-missing diploid.
        samples = np.array(
            [
                [-1, -1],
                [0, -1],
                [-1, 0],
                [0, -2],
                [1, -2],
                [-1, -2],
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ],
            dtype=np.int8,
        )
        G = samples[np.newaxis, ...]
        self._assert_matches(G, np.array([False]))
        self._assert_matches(G, np.array([True]))

    @pytest.mark.parametrize(
        "num_samples", [0, 1, 2, 3, 5, 7, 8, 9, 16, 17, 255, 256, 1000]
    )
    def test_unusual_sample_counts(self, num_samples):
        rng = np.random.default_rng(0)
        # Draw only valid alleles: a in {-1, 0, 1}, b in {-2, -1, 0, 1}.
        a = rng.integers(-1, 2, size=(2, num_samples), dtype=np.int8)
        b = rng.integers(-2, 2, size=(2, num_samples), dtype=np.int8)
        G = np.stack([a, b], axis=-1).astype(np.int8)
        phased = np.array([False, True])
        self._assert_matches(G, phased)

    @pytest.mark.parametrize("num_variants", [0, 1, 2, 7, 16, 100])
    def test_unusual_variant_counts(self, num_variants):
        rng = np.random.default_rng(1)
        a = rng.integers(-1, 2, size=(num_variants, 5), dtype=np.int8)
        b = rng.integers(-2, 2, size=(num_variants, 5), dtype=np.int8)
        G = np.stack([a, b], axis=-1).astype(np.int8)
        phased = rng.integers(0, 2, size=num_variants, dtype=np.int8).astype(bool)
        self._assert_matches(G, phased)

    def test_random_uniform(self):
        rng = np.random.default_rng(42)
        a = rng.integers(-1, 2, size=(50, 50), dtype=np.int8)
        b = rng.integers(-2, 2, size=(50, 50), dtype=np.int8)
        G = np.stack([a, b], axis=-1).astype(np.int8)
        phased = rng.integers(0, 2, size=50, dtype=np.int8).astype(bool)
        self._assert_matches(G, phased)

    def test_random_skewed_missing(self):
        # 30% of samples missing on at least one allele.
        rng = np.random.default_rng(7)
        num_variants = 20
        num_samples = 40
        G = rng.integers(0, 2, size=(num_variants, num_samples, 2), dtype=np.int8)
        mask = rng.random((num_variants, num_samples)) < 0.3
        G[mask, 0] = -1
        side = rng.integers(0, 2, size=(num_variants, num_samples), dtype=np.int8)
        G[mask & (side == 1), 1] = -2
        phased = rng.integers(0, 2, size=num_variants, dtype=np.int8).astype(bool)
        self._assert_matches(G, phased)

    def test_large(self):
        num_variants = 500
        num_samples = 200
        v_idx = np.arange(num_variants)[:, None]
        s_idx = np.arange(num_samples)[None, :]
        # Mix valid values: a in {-1, 0, 1}, b in {-2, -1, 0, 1}.
        a = (v_idx + s_idx) % 3 - 1
        b = (v_idx + 2 * s_idx) % 4 - 2
        G = np.stack([a, b], axis=-1).astype(np.int8)
        phased = (np.arange(num_variants) % 2).astype(bool)
        self._assert_matches(G, phased)

    def test_phased_from_all_reduction(self):
        # Real-world: phased flag comes from a 2D bool array's all(axis=1)
        # in _prepare_chunk. Confirm that source path stays compatible.
        per_sample_phased = np.array(
            [[True, True, True], [True, False, True], [True, True, True]]
        )
        phased = per_sample_phased.all(axis=1)
        assert phased.dtype == bool
        G = np.zeros((3, 3, 2), dtype=np.int8)
        self._assert_matches(G, phased)

    @pytest.mark.parametrize("phased_value", [False, True])
    @pytest.mark.parametrize("b", [-2, -1, 0, 1])
    def test_zero_ploidy_raises(self, phased_value, b):
        # -2 in slot 0 is zero-ploidy and not representable in BGEN;
        # surface as ValueError at write time.
        G = np.array([[[-2, b]]], dtype=np.int8)
        phased = np.array([phased_value])
        with pytest.raises(ValueError, match="zero-ploidy"):
            _vcztools.encode_bgen_geno_blocks(G, phased)

    @pytest.mark.parametrize("phased_value", [False, True])
    @pytest.mark.parametrize(
        "g",
        [
            # Out-of-range in slot 0 (a).
            [2, 0],
            [127, 0],
            [3, 1],
            [-3, 0],
            [-128, -128],
            # Out-of-range in slot 1 (b).
            [0, 2],
            [1, 127],
            [-1, 5],
            [0, -3],
            # Out-of-range haploid call: a invalid, b == -2.
            [2, -2],
            [-3, -2],
        ],
    )
    def test_invalid_allele_raises(self, phased_value, g):
        # BGEN is biallelic; only {-2, -1, 0, 1} are accepted. Anything
        # else is a data-quality error.
        G = np.array([[g]], dtype=np.int8)
        phased = np.array([phased_value])
        with pytest.raises(ValueError, match="out of range"):
            _vcztools.encode_bgen_geno_blocks(G, phased)

    def test_invalid_allele_mid_chunk_surfaces(self):
        # A bad value on variant 2 of a 3-variant chunk surfaces from
        # the kernel even when earlier variants are clean.
        G = np.zeros((3, 4, 2), dtype=np.int8)
        G[2, 1, 0] = 2
        phased = np.zeros(3, dtype=bool)
        with pytest.raises(ValueError, match="out of range"):
            _vcztools.encode_bgen_geno_blocks(G, phased)


class TestPloidyNormalization:
    """Cover the Python-layer shape promotion that lets the encoder
    accept either ``(V, S, 1)`` or ``(V, S, 2)`` call_genotype arrays
    while the C kernel always sees the ``(V, S, 2)``-with-``-2``-pad
    form."""

    def test_check_ploidy_dim_accepts_haploid(self):
        G = np.zeros((2, 3, 1), dtype=np.int8)
        bgen._check_ploidy_dim(G)

    def test_check_ploidy_dim_accepts_diploid(self):
        G = np.zeros((2, 3, 2), dtype=np.int8)
        bgen._check_ploidy_dim(G)

    def test_check_ploidy_dim_rejects_polyploid(self):
        G = np.zeros((2, 3, 3), dtype=np.int8)
        with pytest.raises(ValueError, match="shape"):
            bgen._check_ploidy_dim(G)

    def test_check_ploidy_dim_rejects_2d(self):
        G = np.zeros((2, 3), dtype=np.int8)
        with pytest.raises(ValueError, match="shape"):
            bgen._check_ploidy_dim(G)

    def test_normalize_haploid_promotes_to_padded(self):
        G = np.array([[[0], [1], [-1]]], dtype=np.int8)
        normalised = bgen._normalize_genotype_ploidy(G)
        assert normalised.shape == (1, 3, 2)
        assert normalised.dtype == np.int8
        nt.assert_array_equal(normalised[..., 0], G[..., 0])
        nt.assert_array_equal(normalised[..., 1], [[-2, -2, -2]])

    def test_normalize_diploid_unchanged(self):
        G = np.array([[[0, 1], [1, -2], [-1, -1]]], dtype=np.int8)
        normalised = bgen._normalize_genotype_ploidy(G)
        assert normalised is G

    def test_prepare_chunk_haploid_matches_padded(self):
        # (V, S, 1) input must produce the same per-variant geno blocks
        # as the equivalent (V, S, 2) input with -2 in slot 1.
        G_haploid = np.array([[[0], [1], [-1], [0]]], dtype=np.int8)
        G_padded = np.array([[[0, -2], [1, -2], [-1, -2], [0, -2]]], dtype=np.int8)

        def prepare(G):
            chunk = {
                "call_genotype": G,
                "variant_allele": np.array([["A", "T"]]),
                "variant_contig": np.array([0]),
                "variant_position": np.array([100]),
            }
            chunk_strings = bgen._prepare_chunk_strings(
                chunk, contig_ids=np.array(["chr1"])
            )
            return bgen._prepare_chunk(chunk, chunk_strings, start=0, end=1)

        prep1 = prepare(G_haploid)
        prep2 = prepare(G_padded)
        # _prepare_chunk no longer materialises geno blocks; build them
        # here from the normalised genotypes + phased and compare.
        gb1, gbl1 = _vcztools.encode_bgen_geno_blocks(prep1.genotypes, prep1.phased)
        gb2, gbl2 = _vcztools.encode_bgen_geno_blocks(prep2.genotypes, prep2.phased)
        nt.assert_array_equal(gbl1, gbl2)
        block_len = int(gbl1[0])
        nt.assert_array_equal(gb1[0, :block_len], gb2[0, :block_len])


class TestEncodeVariantBlock:
    def test_round_trip(self, tmp_path):
        G = np.array([[0, 0], [0, 1]], dtype=np.int8)
        geno_block_bytes = _single_variant_geno_block(G)
        block = bgen._encode_variant_block(
            varid_bytes=b"rs1",
            rsid_bytes=b"rs1",
            chrom_bytes=b"chr1",
            position=12345,
            allele1_bytes=b"A",
            allele2_bytes=b"T",
            geno_block_bytes=geno_block_bytes,
            compression_level=-1,
        )
        sample_block = bgen._build_sample_id_block(["s0", "s1"])
        header = bgen._build_header(1, 2, sample_block)
        path = tmp_path / "x.bgen"
        path.write_bytes(header + block)
        with br.open_bgen(path, verbose=False) as bgen_file:
            assert list(bgen_file.samples) == ["s0", "s1"]
            assert bgen_file.nvariants == 1
            assert bgen_file.chromosomes[0] == "chr1"
            assert int(bgen_file.positions[0]) == 12345
            assert str(bgen_file.allele_ids[0]).split(",") == ["A", "T"]
            assert not bool(bgen_file.phased[0])
            probs = bgen_file.read()
        # G=[[0,0],[0,1]] unphased: hom-ref → P(00,01,11)=[1,0,0],
        # het → [0,1,0].
        nt.assert_array_equal(probs[0, 0], [1.0, 0.0, 0.0])
        nt.assert_array_equal(probs[1, 0], [0.0, 1.0, 0.0])


class TestChecks:
    def test_multi_allelic_raises(self):
        alleles = np.array([["A", "T", "G"]], dtype="<U2")
        with pytest.raises(ValueError, match="Multi-allelic"):
            bgen._check_biallelic(alleles)

    def test_two_alleles_with_padding_passes(self):
        # alleles can have a 3rd column that is empty padding.
        alleles = np.array([["A", "T", ""]], dtype="<U2")
        bgen._check_biallelic(alleles)

    def test_polyploid_raises(self):
        G = np.zeros((2, 3, 3), dtype=np.int8)
        with pytest.raises(ValueError, match="shape"):
            bgen._check_ploidy_dim(G)

    def test_diploid_ok(self):
        G = np.zeros((2, 3, 2), dtype=np.int8)
        bgen._check_ploidy_dim(G)

    def test_haploid_ok(self):
        G = np.zeros((2, 3, 1), dtype=np.int8)
        bgen._check_ploidy_dim(G)


class TestPhaseDetection:
    def test_all_true(self):
        assert bgen._detect_variant_phase(np.array([True, True, True])) is True

    def test_all_false(self):
        assert bgen._detect_variant_phase(np.array([False, False])) is False

    def test_mixed(self):
        assert bgen._detect_variant_phase(np.array([True, False, True])) is False

    def test_empty(self):
        # Empty axis: ``.all()`` returns True for empty arrays. We accept
        # that — a zero-sample variant has no observable phase, so the
        # value doesn't matter.
        assert bgen._detect_variant_phase(np.array([], dtype=bool)) is True


# ---------------------------------------------------------------------------
# .bgi index
# ---------------------------------------------------------------------------


class TestWriteBgi:
    """``write_bgi(reader, output, variant_offsets)`` — the single seam
    for ``.bgen.bgi`` generation, used by both :func:`write_bgen`
    (variable-size, cumulative-block offsets) and :class:`BgenEncoder`
    (fixed-size, formula-derived offsets via
    :attr:`BgenEncoder.variant_offsets`)."""

    def test_basic_variant_table(self, tmp_path):
        # Two variants on contig "chr1" at positions 100, 101. Synthetic
        # offsets carve out 128-byte and 140-byte blocks after a 24-byte
        # prefix.
        reader = _build_reader(num_variants=2, num_samples=2)
        variant_offsets = np.array([24, 24 + 128, 24 + 128 + 140], dtype=np.int64)
        bgi_path = tmp_path / "out.bgen.bgi"
        bgen.write_bgi(reader, bgi_path, variant_offsets)
        conn = sqlite3.connect(str(bgi_path))
        try:
            rows = conn.execute(
                "SELECT chromosome, position, rsid, number_of_alleles, "
                "allele1, allele2, file_start_position, size_in_bytes "
                "FROM Variant ORDER BY file_start_position"
            ).fetchall()
            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
        finally:
            conn.close()
        assert rows == [
            ("chr1", 100, ".", 2, "A", "T", 24, 128),
            ("chr1", 101, ".", 2, "A", "T", 24 + 128, 140),
        ]
        assert "Variant" in tables
        assert "Metadata" not in tables

    def test_index_overwritten(self, tmp_path):
        reader = _build_reader(num_variants=2, num_samples=2)
        variant_offsets = np.array([0, 100, 200], dtype=np.int64)
        bgi_path = tmp_path / "out.bgen.bgi"
        bgen.write_bgi(reader, bgi_path, variant_offsets)
        # Second write must not raise (PK conflict would if not unlinked).
        bgen.write_bgi(reader, bgi_path, variant_offsets)

    def test_offsets_wrong_length_raises(self, tmp_path):
        reader = _build_reader(num_variants=2, num_samples=2)
        bad = np.array([0, 100, 200, 300], dtype=np.int64)
        with pytest.raises(ValueError, match="must have shape"):
            bgen.write_bgi(reader, tmp_path / "x.bgi", bad)

    def test_offsets_non_integer_raises(self, tmp_path):
        reader = _build_reader(num_variants=2, num_samples=2)
        bad = np.array([0.0, 100.0, 200.0])
        with pytest.raises(ValueError, match="integer-typed"):
            bgen.write_bgi(reader, tmp_path / "x.bgi", bad)

    def test_variant_id_propagates_to_rsid(self, tmp_path):
        reader = _build_reader(num_variants=2, num_samples=2, variant_id=["rsA", "rsB"])
        variant_offsets = np.array([0, 100, 200], dtype=np.int64)
        bgi_path = tmp_path / "out.bgen.bgi"
        bgen.write_bgi(reader, bgi_path, variant_offsets)
        conn = sqlite3.connect(str(bgi_path))
        try:
            rsids = [
                r[0]
                for r in conn.execute(
                    "SELECT rsid FROM Variant ORDER BY file_start_position"
                ).fetchall()
            ]
        finally:
            conn.close()
        assert rsids == ["rsA", "rsB"]

    def test_rejects_multiallelic(self, tmp_path):
        reader = _build_reader(num_variants=1, num_samples=1, alleles=[("A", "T", "G")])
        variant_offsets = np.array([0, 100], dtype=np.int64)
        with pytest.raises(ValueError, match="Multi-allelic"):
            bgen.write_bgi(reader, tmp_path / "out.bgi", variant_offsets)

    def test_encoder_offsets_round_trip(self, tmp_path):
        # Encoder formula offsets carve up the encoder's byte stream
        # into contiguous, equal-size blocks ending at total_size.
        with _build_encoder(num_variants=5, num_samples=4) as enc:
            reader = enc._reader
            bgi_path = tmp_path / "x.bgen.bgi"
            bgen.write_bgi(reader, bgi_path, enc.variant_offsets)
            conn = sqlite3.connect(str(bgi_path))
            try:
                rows = conn.execute(
                    "SELECT file_start_position, size_in_bytes FROM Variant "
                    "ORDER BY file_start_position"
                ).fetchall()
            finally:
                conn.close()
            assert rows[0][0] == enc.header_size
            for (start, size), (next_start, _) in zip(rows, rows[1:]):
                assert next_start == start + size
            last_start, last_size = rows[-1]
            assert last_start + last_size == enc.bgen_size

    def test_write_bgen_and_encoder_paths_agree_on_metadata(self, tmp_path):
        # The Variant-metadata columns (chrom, pos, rsid, alleles)
        # produced by the two call sites must match for the same reader.
        # Offsets/sizes differ (variable- vs fixed-size encoding) and
        # are excluded from the comparison.
        kwargs = dict(num_variants=4, num_samples=3)

        wb_bgen = tmp_path / "wb.bgen"
        wb_bgi = tmp_path / "wb.bgen.bgi"
        bgen.write_bgen(_build_reader(**kwargs), wb_bgen, bgi_path=wb_bgi)
        conn = sqlite3.connect(str(wb_bgi))
        try:
            wb_rows = conn.execute(
                "SELECT chromosome, position, rsid, number_of_alleles, "
                "allele1, allele2 FROM Variant ORDER BY file_start_position"
            ).fetchall()
        finally:
            conn.close()

        with _build_encoder(**kwargs) as enc:
            enc_bgi = tmp_path / "enc.bgen.bgi"
            bgen.write_bgi(enc._reader, enc_bgi, enc.variant_offsets)
            conn = sqlite3.connect(str(enc_bgi))
            try:
                enc_rows = conn.execute(
                    "SELECT chromosome, position, rsid, number_of_alleles, "
                    "allele1, allele2 FROM Variant ORDER BY file_start_position"
                ).fetchall()
            finally:
                conn.close()
        assert wb_rows == enc_rows


# ---------------------------------------------------------------------------
# End-to-end write_bgen
# ---------------------------------------------------------------------------


class TestWriteBgenEndToEnd:
    """``write_bgen``-specific end-to-end behaviour: sidecar files
    (``.sample``, ``.bgen.bgi``) and edge cases that don't translate to
    the byte-stream encoder. Format-correctness round-trips that apply
    to both encoder code paths live in
    :class:`TestBgenRoundTripViaBgenReader`.
    """

    def test_sidecar_files_written(self, tmp_path):
        reader = _build_reader(num_variants=1, num_samples=2)
        bgen_path = tmp_path / "out.bgen"
        sample_path = tmp_path / "out.sample"
        bgi_path = tmp_path / "out.bgen.bgi"
        bgen.write_bgen(reader, bgen_path, sample_path=sample_path, bgi_path=bgi_path)
        assert bgen_path.exists()
        assert sample_path.exists()
        assert bgi_path.exists()

    def test_sample_text_matches(self, tmp_path):
        reader = _build_reader(num_variants=1, num_samples=2)
        bgen_path = tmp_path / "x.bgen"
        sample_path = tmp_path / "x.sample"
        bgen.write_bgen(reader, bgen_path, sample_path=sample_path)
        text = sample_path.read_text()
        assert text.startswith("ID_1 ID_2 missing\n0 0 0\n")
        assert "sample_0 sample_0 0\n" in text
        assert "sample_1 sample_1 0\n" in text

    def test_bgi_offsets_consistent_with_bgen(self, tmp_path):
        reader = _build_reader(num_variants=3, num_samples=2)
        bgen_path = tmp_path / "x.bgen"
        bgi_path = tmp_path / "x.bgen.bgi"
        bgen.write_bgen(reader, bgen_path, bgi_path=bgi_path)

        # Header offset: first 4 bytes of the BGEN file give the byte
        # offset (relative to byte 4) of the first variant block.
        with open(bgen_path, "rb") as f:
            (offset,) = struct.unpack("<I", f.read(4))
        first_variant_pos = 4 + offset

        conn = sqlite3.connect(str(bgi_path))
        try:
            rows = conn.execute(
                "SELECT file_start_position, size_in_bytes "
                "FROM Variant ORDER BY position"
            ).fetchall()
        finally:
            conn.close()

        assert len(rows) == 3
        assert rows[0][0] == first_variant_pos
        for (start, size), (next_start, _) in zip(rows, rows[1:]):
            assert next_start == start + size
        last_start, last_size = rows[-1]
        assert last_start + last_size == bgen_path.stat().st_size

    def test_bytesio_output_round_trips(self, tmp_path):
        # File-like (no .name) output: write to an in-memory buffer,
        # spill to disk so bgen-reader can parse it back.
        reader = _build_reader(num_variants=2, num_samples=2)
        buf = io.BytesIO()
        bgen.write_bgen(reader, buf)
        bgen_path = tmp_path / "out.bgen"
        bgen_path.write_bytes(buf.getvalue())
        with br.open_bgen(bgen_path, verbose=False) as bg:
            assert bg.nvariants == 2
            assert bg.nsamples == 2

    def test_no_sidecars_when_paths_none(self, tmp_path):
        # sample_path/bgi_path default to None: only the .bgen file is
        # written; no sidecars appear in the same directory.
        reader = _build_reader(num_variants=1, num_samples=2)
        bgen_path = tmp_path / "out.bgen"
        bgen.write_bgen(reader, bgen_path)
        assert bgen_path.exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["out.bgen"]

    def test_warn_when_no_sample_ids_anywhere(self, tmp_path, caplog):
        # embed_header_samples=False and sample_path=None means downstream
        # tools won't be able to associate genotypes with sample IDs.
        reader = _build_reader(num_variants=1, num_samples=2)
        bgen_path = tmp_path / "out.bgen"
        with caplog.at_level(logging.WARNING, logger="vcztools.bgen"):
            bgen.write_bgen(reader, bgen_path, embed_header_samples=False)
        assert any("sample IDs nowhere" in rec.message for rec in caplog.records)

    def test_sample_subset_to_empty_via_filter(self, tmp_path):
        # An empty axis isn't a configuration vcz_builder supports
        # directly; produce a reader-side empty axis by selecting no
        # variants via set_variants. With zero variants the BGEN file
        # is just the header + sample-id block. bgen-reader can't open
        # a 0-variant file, so check the header counts directly.
        reader = _build_reader(num_variants=2, num_samples=2)
        empty_plan = regions.chunk_plan_from_indexes(
            np.array([], dtype=np.int64),
            min_chunk=reader.variants_chunk_size,
        )
        reader.set_variants(empty_plan)
        bgen_path = tmp_path / "x.bgen"
        bgen.write_bgen(reader, bgen_path)
        data = bgen_path.read_bytes()
        (offset,) = struct.unpack_from("<I", data, 0)
        (_, n_variants, n_samples) = struct.unpack_from("<III", data, 4)
        assert n_variants == 0
        assert n_samples == 2
        # File ends after the sample-id block.
        assert bgen_path.stat().st_size == 4 + offset


class TestCompressionLevel:
    @pytest.mark.parametrize("level", [0, 1, 6, 9])
    def test_round_trip_at_level(self, tmp_path, level):
        # G shape is (num_variants=2, num_samples=3, ploidy=2). Variant 0
        # has hom-ref / het / hom-alt across the three samples.
        G = np.array(
            [[[0, 0], [0, 1], [1, 1]], [[1, 1], [0, 1], [0, 0]]], dtype=np.int8
        )
        reader = _build_reader(num_variants=2, num_samples=3, call_genotype=G)
        bgen_path = tmp_path / f"out_l{level}.bgen"
        bgen.write_bgen(reader, bgen_path, compression_level=level)
        with br.open_bgen(bgen_path, verbose=False) as bgen_file:
            probs = bgen_file.read()
        # bgen-reader returns shape (n_samples, n_variants, 3).
        assert probs.shape == (3, 2, 3)
        nt.assert_array_equal(probs[0, 0], [1.0, 0.0, 0.0])
        nt.assert_array_equal(probs[1, 0], [0.0, 1.0, 0.0])
        nt.assert_array_equal(probs[2, 0], [0.0, 0.0, 1.0])

    def test_level_9_not_larger_than_level_0(self, tmp_path):
        # All-zero genotypes give a highly compressible payload, so the
        # difference between level 9 and level 0 dominates the per-variant
        # framing overhead. write_bgen materialises the variant filter in
        # place (bgen.py: ``materialise_variant_filter``), so each write
        # gets its own freshly-built reader.
        num_variants = 64
        num_samples = 32
        reader_kwargs = dict(
            num_variants=num_variants,
            num_samples=num_samples,
            call_genotype=np.zeros((num_variants, num_samples, 2), dtype=np.int8),
        )
        out0 = tmp_path / "l0.bgen"
        bgen.write_bgen(_build_reader(**reader_kwargs), out0, compression_level=0)
        out9 = tmp_path / "l9.bgen"
        bgen.write_bgen(_build_reader(**reader_kwargs), out9, compression_level=9)
        size_l0 = out0.stat().st_size
        size_l9 = out9.stat().st_size
        assert size_l9 < size_l0


# ---------------------------------------------------------------------------
# Round-trip via bgen-reader, parametrized over both encoder code paths.
# ---------------------------------------------------------------------------


@pytest.fixture(params=["write_bgen", "BgenEncoder"])
def write_to_bgen(request, tmp_path):
    """Write a reader to a ``.bgen`` file via one of the two encoder
    code paths and return the resulting path.

    Parametrized over the two BGEN code paths in ``vcztools.bgen``:

    * ``write_bgen`` — variable-size, configurably-compressed variant
      blocks; also writes ``.sample`` and ``.bgen.bgi`` sidecars.
    * ``BgenEncoder`` — fixed-size, zlib-stored variant blocks; emits
      only the ``.bgen`` byte stream (no sidecars).

    A single test exercising this fixture covers both paths.
    """
    interface = request.param

    def write(reader):
        bgen_path = tmp_path / "out.bgen"
        if interface == "write_bgen":
            bgen.write_bgen(reader, bgen_path)
        else:
            with bgen.BgenEncoder(reader) as enc:
                buf = _drain(enc)
            bgen_path.write_bytes(buf)
        return bgen_path

    return write


class TestBgenRoundTripViaBgenReader:
    """End-to-end correctness: write a ``.bgen`` file through one of the
    two encoder code paths and verify metadata and probabilities round-
    trip via the ``bgen-reader`` reference reader.

    Parametrized over the writer interface so each case exercises both
    code paths from a single definition.
    """

    def test_minimal_round_trip(self, write_to_bgen):
        # 3 variants, 4 samples; covers hom-ref / het / hom-alt /
        # missing on variant 0 and reversed-order het (1,0) on variant 2
        # (which the unphased path encodes identically to (0,1)).
        G = np.array(
            [
                [[0, 0], [0, 1], [1, 1], [-1, -1]],
                [[0, 0], [0, 0], [0, 0], [0, 0]],
                [[1, 1], [1, 0], [0, 1], [0, -1]],
            ],
            dtype=np.int8,
        )
        reader = _build_reader(
            num_variants=3,
            num_samples=4,
            variant_position=[100, 200, 300],
            alleles=[("A", "T"), ("C", "G"), ("G", "A")],
            call_genotype=G,
        )
        path = write_to_bgen(reader)
        with br.open_bgen(path, verbose=False) as bg:
            assert bg.nvariants == 3
            assert bg.nsamples == 4
            nt.assert_array_equal(
                bg.samples, ["sample_0", "sample_1", "sample_2", "sample_3"]
            )
            assert bg.chromosomes[0] == "chr1"
            nt.assert_array_equal(bg.positions, [100, 200, 300])
            nt.assert_array_equal(bg.allele_ids, ["A,T", "C,G", "G,A"])
            probs, missing = bg.read(return_missings=True)

        # Variant 0: hom-ref, het, hom-alt, missing.
        nt.assert_array_equal(missing[:, 0], [False, False, False, True])
        nt.assert_array_equal(probs[0, 0], [1.0, 0.0, 0.0])
        nt.assert_array_equal(probs[1, 0], [0.0, 1.0, 0.0])
        nt.assert_array_equal(probs[2, 0], [0.0, 0.0, 1.0])
        assert np.isnan(probs[3, 0]).all()

        # Variant 2: hom-alt, reversed het (1,0), het (0,1),
        # half-missing (treated as fully missing).
        nt.assert_array_equal(missing[:, 2], [False, False, False, True])
        nt.assert_array_equal(probs[0, 2], [0.0, 0.0, 1.0])
        nt.assert_array_equal(probs[1, 2], [0.0, 1.0, 0.0])
        nt.assert_array_equal(probs[2, 2], [0.0, 1.0, 0.0])
        assert np.isnan(probs[3, 2]).all()

    def test_phased_round_trip(self, write_to_bgen):
        # Phased: (0,1) and (1,0) must distinguish.
        G = np.array([[[0, 1], [1, 0], [0, 0], [1, 1]]], dtype=np.int8)
        phased = np.ones((1, 4), dtype=bool)
        reader = _build_reader(
            num_variants=1,
            num_samples=4,
            call_genotype=G,
            call_fields={"genotype_phased": phased},
        )
        path = write_to_bgen(reader)
        with br.open_bgen(path, verbose=False) as bg:
            assert bool(bg.phased[0])
            probs = bg.read()
        # Phased biallelic diploid: bgen-reader returns shape
        # (n_samples, n_variants, 4) with per-haplotype probabilities
        # [P(0|h1), P(1|h1), P(0|h2), P(1|h2)].
        assert probs.shape == (4, 1, 4)
        hap1 = np.argmax(probs[..., 0:2], axis=-1).T
        hap2 = np.argmax(probs[..., 2:4], axis=-1).T
        nt.assert_array_equal(hap1, G[..., 0])
        nt.assert_array_equal(hap2, G[..., 1])

    def test_unphased_when_phased_field_absent(self, write_to_bgen):
        reader = _build_reader(num_variants=1, num_samples=2)
        path = write_to_bgen(reader)
        with br.open_bgen(path, verbose=False) as bg:
            assert not bool(bg.phased[0])

    def test_unphased_flag_overrides_phased_field(self, request, tmp_path):
        # Reader has call_genotype_phased=True for every sample/variant,
        # but unphased=True must force the BGEN payload to phased=0.
        G = np.array([[[0, 1], [1, 0]]], dtype=np.int8)
        phased = np.ones((1, 2), dtype=bool)
        reader = _build_reader(
            num_variants=1,
            num_samples=2,
            call_genotype=G,
            call_fields={"genotype_phased": phased},
        )
        bgen_path = tmp_path / "out.bgen"
        bgen.write_bgen(reader, bgen_path, unphased=True)
        with br.open_bgen(bgen_path, verbose=False) as bg:
            assert not bool(bg.phased[0])

        # BgenEncoder path: same input, unphased=True.
        bgen_path2 = tmp_path / "out2.bgen"
        with bgen.BgenEncoder(reader, unphased=True) as enc:
            buf = _drain(enc)
        bgen_path2.write_bytes(buf)
        with br.open_bgen(bgen_path2, verbose=False) as bg:
            assert not bool(bg.phased[0])

    def test_mixed_phase_degrades_to_unphased(self, write_to_bgen, caplog):
        G = np.array([[[0, 1], [1, 0]]], dtype=np.int8)
        phased = np.array([[True, False]], dtype=bool)
        reader = _build_reader(
            num_variants=1,
            num_samples=2,
            call_genotype=G,
            call_fields={"genotype_phased": phased},
        )
        with caplog.at_level(logging.WARNING, logger="vcztools.bgen"):
            path = write_to_bgen(reader)
        with br.open_bgen(path, verbose=False) as bg:
            assert not bool(bg.phased[0])
        assert any("mixed phase" in r.getMessage() for r in caplog.records)

    def test_variant_id_field_used(self, write_to_bgen):
        reader = _build_reader(
            num_variants=2,
            num_samples=1,
            variant_id=["rsX", "rsY"],
        )
        path = write_to_bgen(reader)
        with br.open_bgen(path, verbose=False) as bg:
            nt.assert_array_equal(bg.rsids, ["rsX", "rsY"])

    def test_empty_variant_id_normalised_to_dot(self, write_to_bgen):
        # An empty-string variant_id (VCF "." → empty in VCZ) is
        # emitted as "." in BGEN, matching write_bim's convention.
        reader = _build_reader(
            num_variants=2,
            num_samples=1,
            variant_id=["", "rsY"],
        )
        path = write_to_bgen(reader)
        with br.open_bgen(path, verbose=False) as bg:
            nt.assert_array_equal(bg.rsids, [".", "rsY"])

    def test_monomorphic_alt_normalised_to_dot(self, write_to_bgen):
        # An empty ALT slot becomes "." in the BGEN output.
        reader = _build_reader(
            num_variants=1,
            num_samples=1,
            alleles=[("A",)],
        )
        path = write_to_bgen(reader)
        with br.open_bgen(path, verbose=False) as bg:
            assert str(bg.allele_ids[0]) == "A,."

    def test_multi_allelic_raises(self, write_to_bgen):
        reader = _build_reader(
            num_variants=1,
            num_samples=2,
            alleles=[("A", "T", "G")],
        )
        with pytest.raises(ValueError, match="Multi-allelic"):
            write_to_bgen(reader)

    def test_sample_subset(self, write_to_bgen):
        reader = _build_reader(num_variants=2, num_samples=4)
        reader.set_samples([0, 2])
        path = write_to_bgen(reader)
        with br.open_bgen(path, verbose=False) as bg:
            nt.assert_array_equal(bg.samples, ["sample_0", "sample_2"])
            assert bg.nsamples == 2
            assert bg.nvariants == 2
            assert bg.read().shape == (2, 2, 3)

    def test_multi_chunk_round_trip(self, write_to_bgen):
        # 10 variants split across 3 plan chunks (chunk size 4) drives
        # the chunk-boundary path through both encoders. Genotypes are
        # set so per-variant content is unique, locking ordering across
        # chunks.
        num_variants = 10
        num_samples = 3
        G = np.zeros((num_variants, num_samples, 2), dtype=np.int8)
        # Variant v: sample 0 is hom-alt, samples 1..end hom-ref. Lets
        # us assert per-variant probability content positionally.
        G[:, 0, 0] = 1
        G[:, 0, 1] = 1
        reader = _build_reader(
            num_variants=num_variants,
            num_samples=num_samples,
            variants_chunk_size=4,
            call_genotype=G,
        )
        path = write_to_bgen(reader)
        with br.open_bgen(path, verbose=False) as bg:
            assert bg.nvariants == num_variants
            probs = bg.read()
        # Sample 0 is hom-alt on every variant; samples 1 and 2 are hom-ref.
        for v in range(num_variants):
            nt.assert_array_equal(probs[0, v], [0.0, 0.0, 1.0])
            nt.assert_array_equal(probs[1, v], [1.0, 0.0, 0.0])
            nt.assert_array_equal(probs[2, v], [1.0, 0.0, 0.0])

    @pytest.mark.parametrize(
        "rsids",
        [
            pytest.param(["rs1", "rs2", "rs3"], id="short"),
            pytest.param(
                ["rs" + "0" * 60, "rs" + "1" * 60, "rs" + "2" * 60],
                id="long-62-byte",
            ),
            pytest.param(["1:100:A:T", "1:200:G:C", "."], id="mixed-and-dot"),
            pytest.param(["x" * 64, "y", "z" * 32], id="at-default-cap"),
        ],
    )
    def test_rsids_round_trip_exactly(self, write_to_bgen, rsids):
        # rsid values round-trip byte-for-byte through bgen-reader for
        # a range of short / long / boundary inputs. Locks in NUL-
        # padding behaviour for BgenEncoder (bgen-reader strips trailing
        # NULs) and direct length-prefixed encoding for write_bgen.
        #
        # Multi-byte UTF-8 rsids are valid per the BGEN spec but
        # bgen-reader assumes ASCII internally; UTF-8 byte-counting is
        # covered by TestPadToFixedLength.test_unicode_multibyte_counts_bytes.
        n = len(rsids)
        reader = _build_reader(
            num_variants=n,
            num_samples=2,
            variant_position=list(range(100, 100 + n)),
            alleles=[("A", "T")] * n,
            variant_id=rsids,
            call_genotype=np.zeros((n, 2, 2), dtype=np.int8),
        )
        path = write_to_bgen(reader)
        with br.open_bgen(path, verbose=False) as bg:
            expected = [r if r != "" else "." for r in rsids]
            nt.assert_array_equal(bg.rsids, expected)

    def test_rsids_dot_when_variant_id_absent(self, write_to_bgen):
        # A VCZ without a ``variant_id`` field emits all rsids as ".".
        # For BgenEncoder this triggers the 1-byte rsid_max default;
        # for write_bgen the rsid literal is "." (1 byte).
        reader = _build_reader(num_variants=3, num_samples=2)
        assert "variant_id" not in reader.field_names
        path = write_to_bgen(reader)
        with br.open_bgen(path, verbose=False) as bg:
            nt.assert_array_equal(bg.rsids, [".", ".", "."])

    def test_long_contig_name_round_trips(self, write_to_bgen):
        # BgenEncoder now derives chrom_max from reader.contig_ids, so
        # a long contig name should pass through without manual tuning.
        long_contig = "chr19_KI270939v1_alt"
        reader = _build_reader(
            num_variants=1,
            num_samples=2,
            contigs=(long_contig,),
            variant_contig=[0],
        )
        path = write_to_bgen(reader)
        with br.open_bgen(path, verbose=False) as bg:
            assert bg.chromosomes[0] == long_contig

    def test_uniform_haploid_round_trip(self, write_to_bgen):
        # All-haploid VCZ stored as shape (V, S, 1). Both encoders
        # support uniform haploid; BgenEncoder picks up the haploid
        # geno_size = 10 + 2*S from reader.call_genotype.shape[2].
        G = np.array(
            [
                [[0], [1], [-1], [0]],
                [[1], [0], [1], [1]],
            ],
            dtype=np.int8,
        )
        reader = _build_reader(
            num_variants=2,
            num_samples=4,
            variant_position=[100, 200],
            alleles=[("A", "T"), ("C", "G")],
            call_genotype=G,
            ploidy=1,
        )
        path = write_to_bgen(reader)
        with br.open_bgen(path, verbose=False, allow_complex=True) as bg:
            assert bg.nvariants == 2
            assert bg.nsamples == 4
            # K=1 biallelic: ncombinations is 2 (allele 0 or 1).
            nt.assert_array_equal(bg.ncombinations, [2, 2])
            probs, missing = bg.read(return_missings=True)
        # Variant 0 by sample: hom-ref haploid -> [1, 0]; alt -> [0, 1];
        # missing -> NaN; hom-ref -> [1, 0].
        assert probs.shape == (4, 2, 2)
        nt.assert_array_equal(probs[0, 0, :2], [1.0, 0.0])
        nt.assert_array_equal(probs[1, 0, :2], [0.0, 1.0])
        assert np.isnan(probs[2, 0]).all()
        nt.assert_array_equal(probs[3, 0, :2], [1.0, 0.0])
        nt.assert_array_equal(missing[:, 0], [False, False, True, False])


class TestBgenMixedPloidyRoundTrip:
    """Mixed-ploidy stores: some samples haploid, some diploid, within
    the same variant. Only ``write_bgen`` (variable-size variant blocks)
    supports this; ``BgenEncoder`` (fixed-size) raises
    ``NotImplementedError`` and is covered in
    :class:`TestBgenEncoderUniformPloidy`."""

    def test_mixed_ploidy_round_trip(self, tmp_path):
        # Synthetic X-chromosome style: sample 0 diploid (female-like),
        # sample 1 haploid (male-like), sample 2 missing-haploid.
        G = np.array(
            [
                [[0, 1], [1, -2], [-1, -2]],
                [[1, 1], [0, -2], [1, -2]],
            ],
            dtype=np.int8,
        )
        reader = _build_reader(
            num_variants=2,
            num_samples=3,
            variant_position=[100, 200],
            alleles=[("A", "T"), ("C", "G")],
            call_genotype=G,
            ploidy=2,
        )
        bgen_path = tmp_path / "mp.bgen"
        bgen.write_bgen(reader, bgen_path)
        with br.open_bgen(bgen_path, verbose=False, allow_complex=True) as bg:
            assert bg.nvariants == 2
            assert bg.nsamples == 3
            # Each variant has both diploid (K=2 -> 3 combos) and haploid
            # (K=1 -> 2 combos) samples; bgen-reader pads to the max.
            nt.assert_array_equal(bg.ncombinations, [3, 3])
            probs, missing = bg.read(return_missings=True)
        # probs.shape == (n_samples, n_variants, max_combinations).
        assert probs.shape == (3, 2, 3)
        # Variant 0:
        # sample 0 diploid het (0, 1) -> [0, 1, 0]
        nt.assert_array_equal(probs[0, 0], [0.0, 1.0, 0.0])
        # sample 1 haploid alt -> [0, 1, NaN]
        nt.assert_array_equal(probs[1, 0, :2], [0.0, 1.0])
        assert np.isnan(probs[1, 0, 2])
        # sample 2 missing haploid -> [NaN, NaN, NaN]
        assert np.isnan(probs[2, 0]).all()
        nt.assert_array_equal(missing[:, 0], [False, False, True])

    def test_haploid_shape_one_round_trip(self, tmp_path):
        # Source (V, S, 1) all-haploid VCZ through write_bgen.
        G = np.array([[[0], [1], [-1]], [[1], [0], [1]]], dtype=np.int8)
        reader = _build_reader(
            num_variants=2,
            num_samples=3,
            variant_position=[100, 200],
            alleles=[("A", "T"), ("C", "G")],
            call_genotype=G,
            ploidy=1,
        )
        bgen_path = tmp_path / "h.bgen"
        bgen.write_bgen(reader, bgen_path)
        with br.open_bgen(bgen_path, verbose=False, allow_complex=True) as bg:
            nt.assert_array_equal(bg.ncombinations, [2, 2])
            probs, missing = bg.read(return_missings=True)
        nt.assert_array_equal(probs[0, 0, :2], [1.0, 0.0])
        nt.assert_array_equal(probs[1, 0, :2], [0.0, 1.0])
        assert np.isnan(probs[2, 0]).all()

    def test_haploid_missing_round_trip(self, tmp_path):
        # All-haploid with some missing alleles via -1 in slot 0.
        G = np.full((1, 5, 2), -2, dtype=np.int8)
        G[0, :, 0] = [0, 1, -1, 1, 0]
        reader = _build_reader(
            num_variants=1,
            num_samples=5,
            call_genotype=G,
            ploidy=2,
        )
        bgen_path = tmp_path / "hm.bgen"
        bgen.write_bgen(reader, bgen_path)
        with br.open_bgen(bgen_path, verbose=False, allow_complex=True) as bg:
            probs, missing = bg.read(return_missings=True)
        nt.assert_array_equal(missing[:, 0], [False, False, True, False, False])
        nt.assert_array_equal(probs[0, 0, :2], [1.0, 0.0])
        nt.assert_array_equal(probs[1, 0, :2], [0.0, 1.0])
        nt.assert_array_equal(probs[3, 0, :2], [0.0, 1.0])
        nt.assert_array_equal(probs[4, 0, :2], [1.0, 0.0])

    def test_zero_ploidy_in_chunk_raises(self, tmp_path):
        # A variant with -2 in slot 0 is zero-ploidy and unrepresentable
        # in BGEN; write_bgen surfaces this as a ValueError from the C
        # kernel via _prepare_chunk.
        G = np.array([[[-2, 0]]], dtype=np.int8)
        reader = _build_reader(
            num_variants=1,
            num_samples=1,
            call_genotype=G,
            ploidy=2,
        )
        out = tmp_path / "z.bgen"
        with pytest.raises(ValueError, match="zero-ploidy"):
            bgen.write_bgen(reader, out)

    def test_invalid_allele_in_chunk_raises(self, tmp_path):
        # An out-of-range allele (e.g. 2 in a biallelic store) is a
        # data-quality error; write_bgen surfaces it from the C kernel
        # via _prepare_chunk.
        G = np.array([[[0, 2]]], dtype=np.int8)
        reader = _build_reader(
            num_variants=1,
            num_samples=1,
            call_genotype=G,
            ploidy=2,
        )
        out = tmp_path / "bad.bgen"
        with pytest.raises(ValueError, match="out of range"):
            bgen.write_bgen(reader, out)


# ---------------------------------------------------------------------------
# BgenEncoder: fixed-size random-access encoder.
# ---------------------------------------------------------------------------


def _varied_genotypes(num_variants, num_samples, seed=42):
    """Deterministic genotypes covering hom-ref/het/hom-alt/missing."""
    rng = np.random.default_rng(seed)
    g = rng.integers(low=-1, high=2, size=(num_variants, num_samples, 2)).astype(
        np.int8
    )
    # A sample is missing iff both alleles are missing; symmetrise so the
    # half-missing case (which BGEN doesn't represent) never arises.
    missing_mask = (g[:, :, 0] == -1) | (g[:, :, 1] == -1)
    g[missing_mask, 0] = -1
    g[missing_mask, 1] = -1
    return g


def _build_encoder(*, num_variants=2, num_samples=3, encoder_kwargs=None, **overrides):
    """Return a BgenEncoder over an in-memory VCZ group."""
    reader = _build_reader(
        num_variants=num_variants, num_samples=num_samples, **overrides
    )
    return bgen.BgenEncoder(reader, **(encoder_kwargs or {}))


def _drain(encoder, step=1 << 17):
    out = bytearray()
    off = 0
    while off < encoder.bgen_size:
        chunk = encoder.read(off, step)
        assert len(chunk) > 0
        out += chunk
        off += len(chunk)
    return bytes(out)


class TestEncodeFieldChunk:
    def test_variable_mode_raw_utf8(self):
        out = bgen._encode_field_chunk(np.array(["rs1", "rs22"]))
        assert bytes(out[0]) == b"rs1"
        assert bytes(out[1]) == b"rs22"
        assert len(out[0]) == 3
        assert len(out[1]) == 4

    def test_fixed_mode_nul_padded(self):
        out = bgen._encode_field_chunk(
            np.array(["rs1", "rs22"]), max_len=8, field_name="rsid"
        )
        assert out.shape == (2, 8)
        assert bytes(out[0]) == b"rs1\x00\x00\x00\x00\x00"
        assert bytes(out[1]) == b"rs22\x00\x00\x00\x00"
        # The length prefix the block encoder writes is len(out[i]); in
        # fixed mode that must equal max_len regardless of content.
        assert len(out[0]) == 8
        assert len(out[1]) == 8

    def test_fixed_mode_exact_length(self):
        out = bgen._encode_field_chunk(np.array(["ABCD"]), max_len=4, field_name="x")
        assert bytes(out[0]) == b"ABCD"
        assert len(out[0]) == 4

    def test_fixed_mode_overflow_raises(self):
        with pytest.raises(ValueError, match="rsid 'too_long'"):
            bgen._encode_field_chunk(
                np.array(["too_long"]), max_len=4, field_name="rsid"
            )

    def test_fixed_mode_overflow_message_names_field_and_max(self):
        with pytest.raises(ValueError, match="exceeds configured max of 3"):
            bgen._encode_field_chunk(np.array(["abcd"]), max_len=3, field_name="varid")

    def test_unicode_multibyte_counts_bytes(self):
        # "é" = 2 bytes in UTF-8.
        out = bgen._encode_field_chunk(np.array(["é"]), max_len=4, field_name="x")
        assert bytes(out[0]) == b"\xc3\xa9\x00\x00"
        with pytest.raises(ValueError, match="exceeds configured max"):
            bgen._encode_field_chunk(np.array(["é"]), max_len=1, field_name="x")

    def test_fixed_mode_empty_chunk(self):
        out = bgen._encode_field_chunk(
            np.array([], dtype="<U8"), max_len=4, field_name="x"
        )
        assert out.shape == (0, 4)


def _make_chunk_for_prep(
    *,
    alleles=None,
    contigs=None,
    positions=None,
    variant_id=None,
    phased=None,
    num_samples=2,
):
    if alleles is None:
        alleles = np.array([["A", "T"], ["C", "G"]])
    n = alleles.shape[0]
    if contigs is None:
        contigs = np.zeros(n, dtype=np.int32)
    if positions is None:
        positions = np.array([100 + i for i in range(n)], dtype=np.int32)
    G = np.zeros((n, num_samples, 2), dtype=np.int8)
    chunk = {
        "call_genotype": G,
        "variant_allele": alleles,
        "variant_contig": contigs,
        "variant_position": positions,
    }
    if variant_id is not None:
        chunk["variant_id"] = np.asarray(variant_id)
    if phased is not None:
        chunk["call_genotype_phased"] = np.asarray(phased, dtype=bool)
    return chunk


class TestPrepareChunkStrings:
    """:func:`_prepare_chunk_strings` does the full-chunk utf-8 +
    NUL-pad pass for every variant string field. Hoisted out of
    :func:`_prepare_chunk` so it runs once per chunk on the main
    thread rather than once per worker slice."""

    def test_variable_mode_basic(self):
        chunk = _make_chunk_for_prep()
        strings = bgen._prepare_chunk_strings(chunk, contig_ids=np.array(["chr1"]))
        assert [bytes(b) for b in strings.varid] == [b".", b"."]
        assert [bytes(b) for b in strings.rsid] == [b".", b"."]
        assert [bytes(b) for b in strings.chrom] == [b"chr1", b"chr1"]
        assert [bytes(b) for b in strings.allele1] == [b"A", b"C"]
        assert [bytes(b) for b in strings.allele2] == [b"T", b"G"]

    def test_fixed_mode_pads_to_max_len(self):
        chunk = _make_chunk_for_prep(variant_id=["rsX", "rsY"])
        strings = bgen._prepare_chunk_strings(
            chunk,
            contig_ids=np.array(["chr1"]),
            varid_max_len=8,
            rsid_max_len=8,
            chrom_max_len=4,
            allele_max_len=2,
        )
        # Fixed mode returns (N, max_len) uint8 rows.
        assert strings.varid.shape == (2, 8)
        assert bytes(strings.varid[0]) == b"rsX\x00\x00\x00\x00\x00"
        assert bytes(strings.chrom[0]) == b"chr1"
        assert bytes(strings.allele1[0]) == b"A\x00"
        # rsid is propagated from variant_id; varid mirrors rsid.
        assert bytes(strings.rsid[0]) == b"rsX\x00\x00\x00\x00\x00"

    def test_missing_rsid_normalised_to_dot(self):
        chunk = _make_chunk_for_prep(variant_id=["", "rsY"])
        strings = bgen._prepare_chunk_strings(chunk, contig_ids=np.array(["chr1"]))
        assert [bytes(b) for b in strings.rsid] == [b".", b"rsY"]

    def test_monomorphic_allele2_normalised_to_dot(self):
        chunk = _make_chunk_for_prep(alleles=np.array([["A", ""], ["C", "G"]]))
        strings = bgen._prepare_chunk_strings(chunk, contig_ids=np.array(["chr1"]))
        assert [bytes(b) for b in strings.allele2] == [b".", b"G"]

    def test_varid_overflow_checked_before_rsid(self):
        # Both rsid_max_len and varid_max_len would overflow, but the
        # varid check fires first.
        chunk = _make_chunk_for_prep(variant_id=["x" * 100, "x" * 100])
        with pytest.raises(ValueError, match="varid"):
            bgen._prepare_chunk_strings(
                chunk,
                contig_ids=np.array(["chr1"]),
                varid_max_len=8,
                rsid_max_len=64,
            )

    def test_rejects_multiallelic(self):
        # _check_biallelic now lives in the strings step (it runs on
        # the full chunk, not a slice).
        alleles = np.array([["A", "T", "G"]])
        chunk = _make_chunk_for_prep(alleles=alleles)
        with pytest.raises(ValueError, match="Multi-allelic"):
            bgen._prepare_chunk_strings(chunk, contig_ids=np.array(["chr1"]))


class TestPrepareChunk:
    """Slice-level :func:`_prepare_chunk` work: genotype encoding,
    phase reduction, and row-slicing the pre-built
    :class:`_ChunkStrings`."""

    def _strings(self, chunk, **kwargs):
        return bgen._prepare_chunk_strings(
            chunk, contig_ids=np.array(["chr1"]), **kwargs
        )

    def test_slices_strings_and_positions(self):
        chunk = _make_chunk_for_prep()
        strings = self._strings(chunk)
        prep = bgen._prepare_chunk(chunk, strings, start=0, end=2)
        assert [bytes(b) for b in prep.varid] == [b".", b"."]
        assert [bytes(b) for b in prep.allele1] == [b"A", b"C"]
        assert [bytes(b) for b in prep.allele2] == [b"T", b"G"]
        np.testing.assert_array_equal(prep.position, [100, 101])
        # Without a call_genotype_phased field every variant is unphased.
        nt.assert_array_equal(prep.phased, [False, False])
        assert prep.mixed_phase_count == 0
        # Slice-level shape: (2 variants, 2 samples, 2 ploidy slots).
        assert prep.genotypes.shape == (2, 2, 2)

    def test_partial_slice_indexes_into_full_chunk_strings(self):
        # Build a 3-variant chunk; ask for the middle row only. The
        # returned _ChunkBytes must reflect just that row.
        chunk = _make_chunk_for_prep(
            alleles=np.array([["A", "T"], ["C", "G"], ["G", "A"]])
        )
        strings = self._strings(chunk)
        prep = bgen._prepare_chunk(chunk, strings, start=1, end=2)
        assert prep.varid.shape[0] == 1
        assert bytes(prep.allele1[0]) == b"C"
        assert bytes(prep.allele2[0]) == b"G"
        np.testing.assert_array_equal(prep.position, [101])
        assert prep.genotypes.shape[0] == 1

    def test_phase_detection_and_mixed_count(self):
        # Variant 0: all samples phased → flag=1
        # Variant 1: some samples phased → mixed → flag=0, mixed++
        # Variant 2: no samples phased → flag=0
        phased = np.array(
            [
                [True, True],
                [True, False],
                [False, False],
            ]
        )
        alleles = np.array([["A", "T"], ["C", "G"], ["G", "A"]])
        chunk = _make_chunk_for_prep(alleles=alleles, phased=phased, num_samples=2)
        strings = self._strings(chunk)
        prep = bgen._prepare_chunk(chunk, strings, start=0, end=3)
        # The per-variant phased flag now lives directly on prep.phased
        # rather than inside materialised geno blocks.
        nt.assert_array_equal(prep.phased, [True, False, False])
        assert prep.mixed_phase_count == 1


class TestBgenEncoderMetadata:
    def test_num_variants_and_samples(self):
        with _build_encoder(num_variants=4, num_samples=5) as enc:
            assert enc.num_variants == 4
            assert enc.num_samples == 5

    def test_bytes_per_variant_formula(self):
        # bpv comes from _vcztools.bgen_variant_block_size; the check
        # asserts the encoder routes the right per-field maxes through
        # it. Default _build_reader has no variant_id, so
        # rsid_max=varid_max=1; default contig "chr1" → chrom_max=4;
        # allele_max=1.
        with _build_encoder(num_variants=2, num_samples=4) as enc:
            expected = _vcztools.bgen_variant_block_size(
                num_samples=4,
                uniform_ploidy=2,
                varid_max=1,
                rsid_max=1,
                chrom_max=4,
                allele_max=1,
            )
            assert enc.bytes_per_variant == expected

    def test_header_size_matches_serialised_header(self):
        reader = _build_reader(num_variants=2, num_samples=3)
        with bgen.BgenEncoder(reader) as enc:
            sib = bgen._build_sample_id_block(reader.sample_ids)
            expected = len(bgen._build_header(2, 3, sib))
            assert enc.header_size == expected

    def test_bgen_size_is_header_plus_variants(self):
        with _build_encoder(num_variants=7, num_samples=5) as enc:
            assert enc.bgen_size == enc.header_size + 7 * enc.bytes_per_variant

    def test_custom_maxes_change_bytes_per_variant(self):
        # Default contig "chr1" → chrom_max=4 (derived, no longer a kwarg).
        with _build_encoder(
            num_variants=1,
            num_samples=2,
            encoder_kwargs=dict(varid_max=16, rsid_max=16, allele_max=2),
        ) as enc:
            expected = _vcztools.bgen_variant_block_size(
                num_samples=2,
                uniform_ploidy=2,
                varid_max=16,
                rsid_max=16,
                chrom_max=4,
                allele_max=2,
            )
            assert enc.bytes_per_variant == expected

    def test_default_rsid_varid_max_when_variant_id_absent(self):
        # Defaults shrink to 1 byte when the source has no variant_id.
        with _build_encoder(num_variants=2, num_samples=4) as enc:
            assert "variant_id" not in enc._reader.field_names
            assert enc._rsid_max == 1
            assert enc._varid_max == 1

    def test_default_rsid_varid_max_when_variant_id_present(self):
        # Defaults stay at 64 when variant_id is present.
        with _build_encoder(
            num_variants=2, num_samples=4, variant_id=["rsA", "rsB"]
        ) as enc:
            assert enc._rsid_max == 64
            assert enc._varid_max == 64

    def test_varid_max_default_follows_rsid_max(self):
        # Explicit rsid_max propagates to varid_max when varid_max is
        # not explicitly set (they share fate in _encode_chunk).
        with _build_encoder(
            num_variants=1,
            num_samples=2,
            encoder_kwargs=dict(rsid_max=32),
        ) as enc:
            assert enc._rsid_max == 32
            assert enc._varid_max == 32

    def test_chrom_max_derived_from_contig_ids(self):
        # chrom_max is derived from the longest contig name in
        # reader.contig_ids rather than a manual knob.
        long_contig = "chr19_KI270939v1_alt"  # 20 bytes
        with _build_encoder(
            num_variants=1,
            num_samples=2,
            contigs=(long_contig,),
            variant_contig=[0],
        ) as enc:
            assert enc._chrom_max == len(long_contig)

    def test_embed_header_samples_false(self, tmp_path):
        # With embed_header_samples=False the SAMPLE_IDS_PRESENT flag is
        # cleared, the sample-id block is absent, and header_size shrinks
        # to the bare 4-byte offset + 20-byte header.
        reader = _build_reader(num_variants=2, num_samples=3)
        with bgen.BgenEncoder(reader, embed_header_samples=False) as enc:
            assert enc.header_size == 4 + bgen.HEADER_LENGTH
            head = enc.read(0, enc.header_size)
            (offset,) = struct.unpack_from("<I", head, 0)
            assert offset == bgen.HEADER_LENGTH
            (flags,) = struct.unpack_from("<I", head, 20)
            assert (flags & bgen.SAMPLE_IDS_PRESENT) == 0


class TestBgenEncoderSequential:
    """Concatenated read(off, step) reassembles the full byte stream
    and parses cleanly with bgen-reader."""

    @pytest.mark.parametrize("step", [1, 7, 17, 4096, 1 << 17])
    def test_step_sizes_reassemble_full_stream(self, step):
        gt = _varied_genotypes(num_variants=11, num_samples=4)
        with _build_encoder(
            num_variants=11,
            num_samples=4,
            call_genotype=gt,
            variants_chunk_size=4,
        ) as enc:
            reference = _drain(enc)
        # Re-drain at the given step and compare.
        with _build_encoder(
            num_variants=11,
            num_samples=4,
            call_genotype=gt,
            variants_chunk_size=4,
        ) as enc:
            out = bytearray()
            off = 0
            while off < enc.bgen_size:
                chunk = enc.read(off, step)
                assert len(chunk) > 0
                out += chunk
                off += len(chunk)
            assert bytes(out) == reference

    def test_full_stream_parses_with_bgen_reader(self, tmp_path):
        gt = _varied_genotypes(num_variants=5, num_samples=3)
        with _build_encoder(num_variants=5, num_samples=3, call_genotype=gt) as enc:
            buf = _drain(enc)
        path = tmp_path / "out.bgen"
        path.write_bytes(buf)
        with br.open_bgen(path, verbose=False) as bg:
            assert bg.nvariants == 5
            assert bg.nsamples == 3


class TestBgenEncoderRestart:
    """Non-sequential reads restart the iterator at the chunk
    containing the requested offset."""

    def test_jump_back_then_forward(self):
        gt = _varied_genotypes(num_variants=12, num_samples=4)
        kwargs = dict(
            num_variants=12,
            num_samples=4,
            call_genotype=gt,
            variants_chunk_size=4,
        )
        with _build_encoder(**kwargs) as enc:
            reference = _drain(enc)
        with _build_encoder(**kwargs) as enc:
            bpv = enc.bytes_per_variant
            hs = enc.header_size
            a = enc.read(hs, 4 * bpv)
            b = enc.read(hs + 4 * bpv, 4 * bpv)
            c = enc.read(hs, 4 * bpv)  # backward jump
            d = enc.read(hs + 8 * bpv, 4 * bpv)  # forward skip
        assert a == reference[hs : hs + 4 * bpv]
        assert b == reference[hs + 4 * bpv : hs + 8 * bpv]
        assert c == reference[hs : hs + 4 * bpv]
        assert d == reference[hs + 8 * bpv : hs + 12 * bpv]

    def test_restart_into_partial_last_chunk(self):
        gt = _varied_genotypes(num_variants=11, num_samples=4)
        kwargs = dict(
            num_variants=11,
            num_samples=4,
            call_genotype=gt,
            variants_chunk_size=4,
        )
        with _build_encoder(**kwargs) as enc:
            reference = _drain(enc)
        with _build_encoder(**kwargs) as enc:
            bpv = enc.bytes_per_variant
            target = enc.header_size + 9 * bpv + 1
            actual = enc.read(target, 5)
        assert actual == reference[target : target + 5]


class TestBgenEncoderEdges:
    def test_header_only_read(self):
        with _build_encoder(num_variants=2, num_samples=3) as enc:
            hdr = enc.read(0, enc.header_size)
            assert len(hdr) == enc.header_size
            # First 4 bytes are the offset prefix; bytes 4-7 are header
            # length (20); bytes 16-19 are BGEN magic.
            assert hdr[16:20] == bgen.BGEN_MAGIC

    def test_partial_header(self):
        with _build_encoder(num_variants=2, num_samples=3) as enc:
            full = enc.read(0, enc.header_size)
        with _build_encoder(num_variants=2, num_samples=3) as enc:
            a = enc.read(0, 1)
            b = enc.read(1, enc.header_size - 1)
        assert a + b == full

    def test_cross_header_into_data(self):
        with _build_encoder(num_variants=2, num_samples=3) as enc:
            reference = _drain(enc)
            hs = enc.header_size
            cross = enc.read(hs - 2, 4)
        assert cross == reference[hs - 2 : hs + 2]

    def test_tail_one_byte(self):
        with _build_encoder(num_variants=4, num_samples=3) as enc:
            reference = _drain(enc)
            tail = enc.read(enc.bgen_size - 1, 1)
        assert tail == reference[-1:]

    def test_read_past_eof_returns_empty(self):
        with _build_encoder(num_variants=2, num_samples=3) as enc:
            assert enc.read(enc.bgen_size, 10) == b""
            assert enc.read(enc.bgen_size + 100, 10) == b""

    def test_zero_size_returns_empty(self):
        with _build_encoder(num_variants=2, num_samples=3) as enc:
            assert enc.read(0, 0) == b""
            assert enc.read(10, 0) == b""

    def test_size_clamped_to_end(self):
        with _build_encoder(num_variants=2, num_samples=3) as enc:
            tail_len = 7
            target = enc.bgen_size - tail_len
            assert len(enc.read(target, 1024)) == tail_len

    def test_negative_off_raises(self):
        with _build_encoder(num_variants=2, num_samples=3) as enc:
            with pytest.raises(ValueError, match="off must be >= 0"):
                enc.read(-1, 5)

    def test_negative_size_raises(self):
        with _build_encoder(num_variants=2, num_samples=3) as enc:
            with pytest.raises(ValueError, match="size must be >= 0"):
                enc.read(0, -1)


class TestBgenEncoderWireFormat:
    """Walk the raw .bgen bytes from a BgenEncoder run and confirm every
    variant block parses without garbage between blocks. Catches
    bytes_per_variant / chunk-stride mismatches that downstream tools
    (plink2, bgen-reader) see as ``Length-0 rsID`` at the first chunk
    boundary."""

    def _parse_variant_blocks(self, payload, num_variants):
        """Parse ``num_variants`` consecutive layout-2 variant blocks
        from ``payload`` (no header). Returns the list of parsed
        records and the byte offset after the last variant."""
        records = []
        p = 0
        for v in range(num_variants):
            start = p
            (L_id,) = struct.unpack_from("<H", payload, p)
            p += 2 + L_id
            (L_rsid,) = struct.unpack_from("<H", payload, p)
            p += 2
            rsid_b = bytes(payload[p : p + L_rsid])
            p += L_rsid
            (L_chr,) = struct.unpack_from("<H", payload, p)
            p += 2
            chr_b = bytes(payload[p : p + L_chr])
            p += L_chr
            (pos,) = struct.unpack_from("<I", payload, p)
            p += 4
            (K,) = struct.unpack_from("<H", payload, p)
            p += 2
            assert K == 2, f"variant {v}: K={K} expected 2"
            (LA,) = struct.unpack_from("<I", payload, p)
            p += 4
            allele1_b = bytes(payload[p : p + LA])
            p += LA
            (LB,) = struct.unpack_from("<I", payload, p)
            p += 4
            allele2_b = bytes(payload[p : p + LB])
            p += LB
            (C,) = struct.unpack_from("<I", payload, p)
            (D,) = struct.unpack_from("<I", payload, p + 4)
            geno_payload = bytes(payload[p + 8 : p + 4 + C])
            p += 4 + C
            records.append(
                dict(
                    L_id=L_id,
                    rsid=rsid_b,
                    L_rsid=L_rsid,
                    chrom=chr_b,
                    pos=pos,
                    allele1=allele1_b,
                    allele2=allele2_b,
                    C=C,
                    D=D,
                    payload=geno_payload,
                    size=p - start,
                )
            )
        return records, p

    def test_multi_chunk_diploid_22k_samples_parses_contiguously(self):
        # geno_size = 10 + 3*21841 = 65533: Python's zlib.compress emits
        # one extra stored block for source lengths in [65532, 65535]
        # vs vcz_compress_bound's prediction, so a Python-side bpv
        # computed from len(zlib.compress(...)) over-allocates by 5
        # bytes per variant. With variants_chunk_size=4 < num_variants
        # the encoder iterates more than one chunk; the over-allocation
        # leaves zero-byte slack at the end of every encoded chunk
        # and the next chunk's first variant is no longer where the
        # spec-walker expects it.
        num_samples = 21841
        num_variants = 8
        reader = _build_reader(
            num_variants=num_variants,
            num_samples=num_samples,
            variants_chunk_size=4,
        )
        with bgen.BgenEncoder(reader) as enc:
            data = _drain(enc)[enc.header_size :]
        records, end = self._parse_variant_blocks(data, num_variants)
        assert end == len(data), (
            f"after parsing {num_variants} variants, walked {end} bytes "
            f"but payload is {len(data)} bytes (trailing slack)"
        )
        for v, rec in enumerate(records):
            assert rec["L_rsid"] >= 1, (
                f"variant {v}: L_rsid={rec['L_rsid']} (BGEN spec allows 0 "
                f"but plink2 rejects it as a missing-ID marker)"
            )
            # Every variant block must have the same size for the
            # fixed-size encoder; slack between blocks would make this
            # diverge.
            assert rec["size"] == records[0]["size"], (
                f"variant {v}: size={rec['size']} != variant 0 size "
                f"{records[0]['size']}; chunk-boundary misalignment"
            )

    def test_single_chunk_small_samples_parses_contiguously(self):
        # Sanity guard for the path where Python and C agree on
        # compressed_geno_size (small num_samples => geno_size < 65532).
        # Lock down the wire-format walker so a future regression in
        # the small-sample path also surfaces.
        num_samples = 4
        num_variants = 3
        reader = _build_reader(
            num_variants=num_variants,
            num_samples=num_samples,
        )
        with bgen.BgenEncoder(reader) as enc:
            data = _drain(enc)[enc.header_size :]
        records, end = self._parse_variant_blocks(data, num_variants)
        assert end == len(data)
        for rec in records:
            assert rec["L_rsid"] >= 1


class TestBgenEncoderBoundarySizeDecompression:
    """End-to-end decompression of BgenEncoder output at the
    ``geno_size`` values where Python's ``zlib.compress`` previously
    drifted from the C kernel's ``vcz_compress_bound`` (and where the
    user-reported ``Length-0 rsID`` failure originated). For every
    variant in the encoded stream, decompress the C-declared payload
    and assert it matches the reference geno-block builder's output
    for the same inputs — proves the wire format is internally
    consistent at the boundary sizes, not just structurally walkable.
    """

    def _parse_variant_blocks(self, payload, num_variants):
        # Local copy of the parser in TestBgenEncoderWireFormat so
        # these tests stay self-contained.
        wf = TestBgenEncoderWireFormat()
        return wf._parse_variant_blocks(payload, num_variants)

    def _check_decompression(self, *, num_samples, num_variants, ploidy, chunk_size):
        """Build a varied-genotype reader, encode through BgenEncoder,
        decompress each variant's payload, and assert it round-trips
        to the geno-block builder's output."""
        G = _varied_genotypes(num_variants=num_variants, num_samples=num_samples)
        if ploidy == 1:
            G = G[:, :, :1]
        reader = _build_reader(
            num_variants=num_variants,
            num_samples=num_samples,
            call_genotype=G,
            ploidy=ploidy,
            variants_chunk_size=chunk_size,
        )
        with bgen.BgenEncoder(reader) as enc:
            data = _drain(enc)[enc.header_size :]
            uniform_geno_size = enc._uniform_geno_size

        records, end = self._parse_variant_blocks(data, num_variants)
        assert end == len(data)

        # Reference geno-blocks: same input the kernel saw, run
        # through the same kernel-side builder so missingness /
        # phasing / ploidy normalisation lines up exactly.
        phased = np.zeros(num_variants, dtype=bool)
        G_padded = np.full((num_variants, num_samples, 2), -2, dtype=np.int8)
        G_padded[:, :, : G.shape[2]] = G
        ref_blocks, ref_lens = _vcztools.encode_bgen_geno_blocks(G_padded, phased)
        assert (ref_lens == uniform_geno_size).all()

        for v, rec in enumerate(records):
            assert rec["D"] == uniform_geno_size, (
                f"variant {v}: D={rec['D']} != geno_size {uniform_geno_size}"
            )
            decompressed = zlib.decompress(rec["payload"])
            assert len(decompressed) == uniform_geno_size
            assert decompressed == bytes(ref_blocks[v, :uniform_geno_size]), (
                f"variant {v}: decompressed geno-block does not match reference"
            )
            # The adler32 trailer must verify too — zlib.decompress
            # would have raised on a mismatch, but assert explicitly
            # so a regression that strips the trailer is caught here.
            adler = int.from_bytes(rec["payload"][-4:], "big")
            assert adler == zlib.adler32(decompressed)

    def test_diploid_boundary_single_chunk(self):
        # ns=21841 → geno_size=65533 (boundary band, diploid). Single
        # chunk: confirms the boundary-size payload itself decompresses.
        self._check_decompression(
            num_samples=21841, num_variants=3, ploidy=2, chunk_size=3
        )

    def test_diploid_boundary_multi_chunk(self):
        # Same geno_size, multiple chunks: confirms each chunk's
        # payloads decompress and inter-chunk alignment holds.
        self._check_decompression(
            num_samples=21841, num_variants=6, ploidy=2, chunk_size=2
        )

    def test_haploid_boundary_65532(self):
        # ns=32761, ploidy=1 → geno_size=65532 (boundary band, haploid).
        self._check_decompression(
            num_samples=32761, num_variants=4, ploidy=1, chunk_size=2
        )

    def test_haploid_boundary_65534(self):
        # ns=32762, ploidy=1 → geno_size=65534 (boundary band, haploid).
        self._check_decompression(
            num_samples=32762, num_variants=4, ploidy=1, chunk_size=2
        )

    def test_diploid_just_past_boundary(self):
        # ns=21844 → geno_size=65542 (one byte past the boundary; one
        # full 65535-byte stored block + a small trailing block).
        # Python and C agree on the size here, but exercising the
        # two-stored-block path keeps this side of the boundary
        # covered alongside the band itself.
        self._check_decompression(
            num_samples=21844, num_variants=4, ploidy=2, chunk_size=2
        )


class TestBgenEncoderEmptyStore:
    """num_variants == 0 ⇒ stream is just the header bytes."""

    def test_zero_variants_emits_header_only(self):
        # vcz_builder.make_vcz infers region_index shape from row count,
        # so for 0 variants we pass an explicit (0, 6) array.
        root = vcz_builder.make_vcz(
            variant_contig=np.zeros(0, dtype=np.int32),
            variant_position=np.zeros(0, dtype=np.int32),
            alleles=np.zeros((0, 2), dtype="<U16"),
            num_samples=3,
            sample_id=["s0", "s1", "s2"],
            call_genotype=np.zeros((0, 3, 2), dtype=np.int8),
            region_index=np.zeros((0, 6), dtype=np.int32),
        )
        with bgen.BgenEncoder(retrieval.VczReader(root)) as enc:
            assert enc.num_variants == 0
            assert enc.bgen_size == enc.header_size
            full = enc.read(0, 4096)
            assert len(full) == enc.header_size
            assert enc.read(enc.header_size, 1) == b""


class TestBgenEncoderLifecycle:
    def test_context_manager(self):
        reader = _build_reader(num_variants=2, num_samples=3)
        with bgen.BgenEncoder(reader) as enc:
            _drain(enc)
        with pytest.raises(RuntimeError, match="encoder closed"):
            enc.read(0, 1)

    def test_close_is_idempotent(self):
        reader = _build_reader(num_variants=2, num_samples=3)
        enc = bgen.BgenEncoder(reader)
        enc.close()
        enc.close()  # no error

    def test_close_does_not_close_reader(self):
        reader = _build_reader(num_variants=2, num_samples=3)
        with bgen.BgenEncoder(reader):
            pass
        # Reader still usable after encoder close.
        assert reader.sample_ids.size == 3

    def test_constructor_rejects_variant_filter(self):
        reader = _build_reader(num_variants=3, num_samples=3)
        reader.set_variant_filter(
            bcftools_filter.BcftoolsFilter(field_names=set(), include="N_ALT <= 1")
        )
        with pytest.raises(NotImplementedError, match="set_variant_filter"):
            bgen.BgenEncoder(reader)

    def test_constructor_validates_args(self):
        reader = _build_reader(num_variants=1, num_samples=1)
        with pytest.raises(ValueError, match="encode_threads"):
            bgen.BgenEncoder(reader, encode_threads=0)
        with pytest.raises(ValueError, match="encode_block_bytes"):
            bgen.BgenEncoder(reader, encode_block_bytes=0)
        with pytest.raises(ValueError, match="varid_max"):
            bgen.BgenEncoder(reader, varid_max=0)
        with pytest.raises(ValueError, match="rsid_max"):
            bgen.BgenEncoder(reader, rsid_max=0)
        with pytest.raises(ValueError, match="allele_max"):
            bgen.BgenEncoder(reader, allele_max=0)

    def test_constructor_rejects_oversized_maxes(self):
        reader = _build_reader(num_variants=1, num_samples=1)
        with pytest.raises(ValueError, match="varid_max=65536"):
            bgen.BgenEncoder(reader, varid_max=0x10000)
        with pytest.raises(ValueError, match="allele_max"):
            bgen.BgenEncoder(reader, allele_max=0x100000000)


class TestBgenEncoderSharedReader:
    def test_two_encoders_share_one_reader(self):
        gt = _varied_genotypes(num_variants=5, num_samples=3)
        reader = _build_reader(num_variants=5, num_samples=3, call_genotype=gt)
        with bgen.BgenEncoder(reader) as a, bgen.BgenEncoder(reader) as b:
            buf_a = _drain(a)
            buf_b = _drain(b)
        assert buf_a == buf_b


class TestBgenEncoderParallelEncoding:
    """Parallel encoding produces byte-identical output regardless of
    thread count or sub-block size."""

    def test_parallel_matches_sequential(self):
        gt = _varied_genotypes(num_variants=16, num_samples=8)
        kwargs = dict(
            num_variants=16,
            num_samples=8,
            call_genotype=gt,
            variants_chunk_size=8,
        )
        with _build_encoder(**kwargs, encoder_kwargs=dict(encode_threads=1)) as enc:
            seq = _drain(enc)
        for threads in (2, 4):
            with _build_encoder(
                **kwargs,
                encoder_kwargs=dict(encode_threads=threads),
            ) as enc:
                par = _drain(enc)
            assert par == seq

    def test_tiny_block_bytes_forces_subblocking(self):
        gt = _varied_genotypes(num_variants=12, num_samples=6)
        kwargs = dict(num_variants=12, num_samples=6, call_genotype=gt)
        with _build_encoder(**kwargs, encoder_kwargs=dict(encode_threads=1)) as enc:
            seq = _drain(enc)
        with _build_encoder(
            **kwargs,
            encoder_kwargs=dict(encode_threads=4, encode_block_bytes=1),
        ) as enc:
            par = _drain(enc)
        assert par == seq


class TestBgenEncoderWithSetVariants:
    def test_variant_subset_reflected_in_output(self, tmp_path):
        gt = _varied_genotypes(num_variants=10, num_samples=4)
        reader = _build_reader(num_variants=10, num_samples=4, call_genotype=gt)
        indexes = np.array([1, 3, 5], dtype=np.int64)
        reader.set_variants(indexes)
        with bgen.BgenEncoder(reader) as enc:
            assert enc.num_variants == 3
            buf = _drain(enc)
        path = tmp_path / "vs.bgen"
        path.write_bytes(buf)
        with br.open_bgen(path, verbose=False) as bg:
            assert bg.nvariants == 3
            # Default reader builds variant_position = [100..109]; subset
            # picks indices 1, 3, 5 → positions 101, 103, 105.
            nt.assert_array_equal(bg.positions, [101, 103, 105])

    def test_sample_subset_reflected_in_output(self, tmp_path):
        gt = _varied_genotypes(num_variants=3, num_samples=6)
        reader = _build_reader(num_variants=3, num_samples=6, call_genotype=gt)
        reader.set_samples(np.array([0, 2, 4], dtype=np.int64))
        with bgen.BgenEncoder(reader) as enc:
            assert enc.num_samples == 3
            buf = _drain(enc)
        path = tmp_path / "out.bgen"
        path.write_bytes(buf)
        with br.open_bgen(path, verbose=False) as bg:
            assert bg.nsamples == 3
            nt.assert_array_equal(bg.samples, ["sample_0", "sample_2", "sample_4"])


class TestBgenEncoderOverflowRaises:
    def test_varid_too_long(self):
        reader = _build_reader(num_variants=1, num_samples=1, variant_id=["x" * 100])
        with bgen.BgenEncoder(reader, varid_max=8) as enc:
            with pytest.raises(ValueError, match="varid"):
                enc.read(0, enc.bgen_size)

    def test_contig_exceeds_bgen_length_prefix(self):
        # chrom_max is now derived from reader.contig_ids, so an
        # arbitrarily long contig name produces a wider variant block
        # rather than failing. A name that exceeds the uint16 length
        # prefix is the only remaining error path; caught at
        # construction time, not lazily.
        #
        # vcz_builder stores contig_id with dtype "<U32" which truncates
        # 70000-char strings, so inject directly into the reader's
        # cached_property dict.
        reader = _build_reader(num_variants=1, num_samples=1)
        reader.__dict__["contig_ids"] = np.array(["x" * 70000])
        with pytest.raises(ValueError, match="longest contig is 70000 bytes"):
            bgen.BgenEncoder(reader)

    def test_allele_too_long(self):
        reader = _build_reader(
            num_variants=1,
            num_samples=1,
            alleles=[("AT", "G")],
        )
        with bgen.BgenEncoder(reader) as enc:
            with pytest.raises(ValueError, match="allele1"):
                enc.read(0, enc.bgen_size)


class TestBgenEncoderUniformPloidy:
    """BgenEncoder requires uniform ploidy across the store. It auto-
    detects from ``reader.call_genotype.shape[2]``: 1 -> haploid, 2 ->
    diploid. Mixed-ploidy (a ``-2`` sentinel under declared diploid)
    raises ``NotImplementedError`` lazily on read with a message
    pointing the user at :func:`write_bgen`."""

    def test_haploid_geno_size_formula(self):
        # Haploid: geno_size = 10 + 2*S.
        num_samples = 4
        G = np.zeros((2, num_samples, 1), dtype=np.int8)
        reader = _build_reader(
            num_variants=2, num_samples=num_samples, call_genotype=G, ploidy=1
        )
        with bgen.BgenEncoder(reader) as enc:
            geno_size = 10 + 2 * num_samples
            expected = _vcztools.bgen_variant_block_size(
                num_samples=num_samples,
                uniform_ploidy=1,
                varid_max=1,
                rsid_max=1,
                chrom_max=4,
                allele_max=1,
            )
            assert enc.bytes_per_variant == expected
            assert enc._uniform_ploidy == 1
            assert enc._uniform_geno_size == geno_size

    def test_diploid_geno_size_formula(self):
        # Default diploid path: geno_size = 10 + 3*S.
        with _build_encoder(num_variants=2, num_samples=4) as enc:
            assert enc._uniform_ploidy == 2
            assert enc._uniform_geno_size == 10 + 3 * 4

    def test_haploid_round_trip(self, tmp_path):
        # All-haploid (V, S, 1) reader; bgen-reader recovers per-sample
        # allele probabilities at K=1.
        G = np.array([[[0], [1], [-1], [0]], [[1], [0], [1], [1]]], dtype=np.int8)
        reader = _build_reader(
            num_variants=2,
            num_samples=4,
            variant_position=[100, 200],
            alleles=[("A", "T"), ("C", "G")],
            call_genotype=G,
            ploidy=1,
        )
        bgen_path = tmp_path / "h.bgen"
        with bgen.BgenEncoder(reader) as enc:
            bgen_path.write_bytes(_drain(enc))
        with br.open_bgen(bgen_path, verbose=False, allow_complex=True) as bg:
            assert bg.nvariants == 2
            assert bg.nsamples == 4
            nt.assert_array_equal(bg.ncombinations, [2, 2])
            probs, missing = bg.read(return_missings=True)
        nt.assert_array_equal(probs[0, 0, :2], [1.0, 0.0])
        nt.assert_array_equal(probs[1, 0, :2], [0.0, 1.0])
        nt.assert_array_equal(missing[:, 0], [False, False, True, False])

    def test_mixed_ploidy_raises_not_implemented(self):
        # Declared ploidy=2 (shape[2]==2) but the chunk has a -2
        # sentinel -> per-variant length 12 != worst-case 13 for one
        # sample. BgenEncoder rejects the chunk on read.
        G = np.array([[[0, -2]], [[1, 1]]], dtype=np.int8)
        reader = _build_reader(
            num_variants=2,
            num_samples=1,
            call_genotype=G,
            ploidy=2,
        )
        with bgen.BgenEncoder(reader) as enc:
            with pytest.raises(NotImplementedError, match="write_bgen"):
                enc.read(0, enc.bgen_size)

    def test_invalid_ploidy_dim_raises(self):
        # Polyploid (shape[2] > 2) is rejected at construction time;
        # the encoder doesn't carry a meaningful geno_size for it.
        G = np.zeros((1, 1, 3), dtype=np.int8)
        # vcz_builder requires the ploidy kwarg to match call_genotype's
        # last dim; bypass by building the reader with a shape-3 array.
        reader = _build_reader(num_variants=1, num_samples=1, call_genotype=G, ploidy=3)
        with pytest.raises(ValueError, match=r"shape\[2\] in"):
            bgen.BgenEncoder(reader)


# ---------------------------------------------------------------------------
# C kernel: encode_bgen_chunk_slice_level0.
# Drives BgenEncoder._encode_variant_range; tests target it via the encoder
# in addition to direct unit tests on _vcztools.encode_bgen_chunk_slice_level0.
# ---------------------------------------------------------------------------


def _python_encode_variant_range(prep, uniform_len):
    """Reference Python implementation matching the old loop body."""
    n = prep.varid.shape[0]
    geno_blocks, _ = _vcztools.encode_bgen_geno_blocks(prep.genotypes, prep.phased)
    out = bytearray()
    for j in range(n):
        block = bgen._encode_variant_block(
            varid_bytes=bytes(prep.varid[j]),
            rsid_bytes=bytes(prep.rsid[j]),
            chrom_bytes=bytes(prep.chrom[j]),
            position=int(prep.position[j]),
            allele1_bytes=bytes(prep.allele1[j]),
            allele2_bytes=bytes(prep.allele2[j]),
            geno_block_bytes=bytes(geno_blocks[j, :uniform_len]),
            compression_level=0,
        )
        out.extend(block)
    return bytes(out)


class TestBgenChunkSliceLevel0Kernel:
    """Direct unit tests for ``_vcztools.encode_bgen_chunk_slice_level0``."""

    @pytest.mark.parametrize(
        ("num_samples", "uniform_ploidy"),
        [(1, 2), (1024, 1), (1024, 2)],
    )
    @pytest.mark.parametrize(
        ("varid_max", "rsid_max", "chrom_max", "allele_max"),
        [(1, 1, 4, 1), (32, 32, 8, 4)],
    )
    def test_byte_for_byte_parity_with_python(
        self, num_samples, uniform_ploidy, varid_max, rsid_max, chrom_max, allele_max
    ):
        # rsid_max=1 is paired with no variant_id field (encoder emits "."
        # for every rsid in that case); the >1 case carries real ids.
        num_variants = 6
        rng = np.random.default_rng(0)
        G = rng.integers(
            low=-1, high=2, size=(num_variants, num_samples, uniform_ploidy)
        ).astype(np.int8)
        if uniform_ploidy == 2:
            missing = (G[:, :, 0] == -1) | (G[:, :, 1] == -1)
            G[missing, 0] = -1
            G[missing, 1] = -1
        if rsid_max == 1:
            variant_id = None
        else:
            variant_id = [f"rs{i}" for i in range(num_variants)]
        contig_name = "chr" + "X" * (chrom_max - 3)
        reader_kwargs = dict(
            num_variants=num_variants,
            num_samples=num_samples,
            variant_position=list(range(100, 100 + num_variants)),
            alleles=[("A" * allele_max, "T" * allele_max)] * num_variants,
            contigs=(contig_name,),
            variant_contig=[0] * num_variants,
            call_genotype=G,
            ploidy=uniform_ploidy,
        )
        if variant_id is not None:
            reader_kwargs["variant_id"] = variant_id
        reader = _build_reader(**reader_kwargs)
        with bgen.BgenEncoder(
            reader,
            varid_max=varid_max,
            rsid_max=rsid_max,
            allele_max=allele_max,
        ) as enc:
            c_out = _drain(enc)[enc.header_size :]
        # Build the reference by replaying _prepare_chunk + Python loop.
        chunk = {
            "call_genotype": G,
            "variant_allele": np.array(
                [("A" * allele_max, "T" * allele_max) for _ in range(num_variants)]
            ),
            "variant_contig": np.zeros(num_variants, dtype=np.int32),
            "variant_position": np.arange(100, 100 + num_variants, dtype=np.int32),
        }
        if variant_id is not None:
            chunk["variant_id"] = np.asarray(variant_id)
        chunk_strings = bgen._prepare_chunk_strings(
            chunk,
            contig_ids=np.array([contig_name]),
            varid_max_len=varid_max,
            rsid_max_len=rsid_max,
            chrom_max_len=chrom_max,
            allele_max_len=allele_max,
        )
        prep = bgen._prepare_chunk(chunk, chunk_strings, start=0, end=num_variants)
        uniform_len = 10 + (uniform_ploidy + 1) * num_samples
        py_out = _python_encode_variant_range(prep, uniform_len)
        assert c_out == py_out

    def test_d_greater_than_65535_two_stored_blocks(self):
        # num_samples=22000 diploid -> geno_size = 10 + 3*22000 = 66010,
        # which spans two stored DEFLATE blocks (65535 + 475). Drive the
        # C kernel directly to avoid the overhead of materialising a
        # 22000-sample VCZ store.
        num_samples = 22000
        num_variants = 2
        varid_max = rsid_max = chrom_max = 4
        allele_max = 1
        uniform_ploidy = 2
        geno_size = 10 + 3 * num_samples
        G = np.zeros((num_variants, num_samples, 2), dtype=np.int8)
        phased = np.zeros(num_variants, dtype=bool)
        varid = np.zeros((num_variants, varid_max), dtype=np.uint8)
        rsid = np.zeros((num_variants, rsid_max), dtype=np.uint8)
        chrom = np.zeros((num_variants, chrom_max), dtype=np.uint8)
        allele1 = np.zeros((num_variants, allele_max), dtype=np.uint8)
        allele2 = np.zeros((num_variants, allele_max), dtype=np.uint8)
        position = np.array([100, 200], dtype=np.int32)
        bpv = _vcztools.bgen_variant_block_size(
            num_samples=num_samples,
            uniform_ploidy=uniform_ploidy,
            varid_max=varid_max,
            rsid_max=rsid_max,
            chrom_max=chrom_max,
            allele_max=allele_max,
        )
        out = np.empty(num_variants * bpv, dtype=np.uint8)
        _vcztools.encode_bgen_chunk_slice_level0(
            varid,
            rsid,
            chrom,
            allele1,
            allele2,
            position,
            G,
            phased,
            out,
            uniform_ploidy,
        )

        header_overhead = (
            2
            + varid_max
            + 2
            + rsid_max
            + 2
            + chrom_max
            + 4
            + 2
            + 4
            + allele_max
            + 4
            + allele_max
        )
        v0 = bytes(out[:bpv])
        (C,) = struct.unpack_from("<I", v0, header_overhead)
        (D,) = struct.unpack_from("<I", v0, header_overhead + 4)
        assert D == geno_size
        payload = v0[header_overhead + 8 : header_overhead + 4 + C]
        # zlib stream decodes back to the uncompressed geno block built
        # from the same G/phased the kernel saw.
        expected_geno, _ = _vcztools.encode_bgen_geno_blocks(G, phased)
        assert zlib.decompress(payload) == bytes(expected_geno[0, :geno_size])
        # First block has BFINAL=0; second has BFINAL=1.
        first_block_header = payload[2]
        second_block_offset = 2 + 5 + 65535
        second_block_header = payload[second_block_offset]
        assert first_block_header & 0x01 == 0
        assert second_block_header & 0x01 == 1

    def test_adler32_matches_python(self):
        num_samples = 3
        num_variants = 4
        gt = _varied_genotypes(num_variants=num_variants, num_samples=num_samples)
        with _build_encoder(
            num_variants=num_variants, num_samples=num_samples, call_genotype=gt
        ) as enc:
            buf = _drain(enc)[enc.header_size :]
            bpv = enc.bytes_per_variant
            uniform_len = enc._uniform_geno_size
        header_overhead = (
            2
            + enc._varid_max
            + 2
            + enc._rsid_max
            + 2
            + enc._chrom_max
            + 4
            + 2
            + 4
            + enc._allele_max
            + 4
            + enc._allele_max
        )
        # Cross-check adler32 against the actual uncompressed block built
        # by the geno-block C kernel — same input the kernel saw.
        G2 = np.zeros((num_variants, num_samples, 2), dtype=np.int8)
        # Match _build_reader default (zeros) — but use the same gt the
        # encoder ran on to keep results consistent.
        G2[:] = gt
        phased = np.zeros(num_variants, dtype=bool)
        geno_blocks, geno_lens = _vcztools.encode_bgen_geno_blocks(G2, phased)
        assert (geno_lens == uniform_len).all()
        for v in range(num_variants):
            v_block = buf[v * bpv : (v + 1) * bpv]
            (C,) = struct.unpack_from("<I", v_block, header_overhead)
            payload = v_block[header_overhead + 8 : header_overhead + 4 + C]
            uncompressed = bytes(geno_blocks[v, :uniform_len])
            assert int.from_bytes(payload[-4:], "big") == zlib.adler32(uncompressed)

    def test_buffer_too_small_raises(self):
        # Build a valid inputs set with bytes_per_variant > short buffer.
        num_variants = 2
        num_samples = 2
        varid_max = rsid_max = chrom_max = 4
        allele_max = 1
        G = np.zeros((num_variants, num_samples, 2), dtype=np.int8)
        phased = np.zeros(num_variants, dtype=bool)
        varid = np.zeros((num_variants, varid_max), dtype=np.uint8)
        rsid = np.zeros((num_variants, rsid_max), dtype=np.uint8)
        chrom = np.zeros((num_variants, chrom_max), dtype=np.uint8)
        allele1 = np.zeros((num_variants, allele_max), dtype=np.uint8)
        allele2 = np.zeros((num_variants, allele_max), dtype=np.uint8)
        position = np.zeros(num_variants, dtype=np.int32)
        out_buf = np.zeros(8, dtype=np.uint8)  # deliberately too small
        with pytest.raises(_vcztools.VczBufferTooSmall):
            _vcztools.encode_bgen_chunk_slice_level0(
                varid,
                rsid,
                chrom,
                allele1,
                allele2,
                position,
                G,
                phased,
                out_buf,
                2,  # uniform_ploidy
            )

    def test_round_trip_via_bgen_reader(self, tmp_path):
        # Encoder path uses the C kernel by default; this drains the
        # full byte stream and parses it with bgen-reader.
        gt = _varied_genotypes(num_variants=5, num_samples=3)
        with _build_encoder(num_variants=5, num_samples=3, call_genotype=gt) as enc:
            buf = _drain(enc)
        path = tmp_path / "k.bgen"
        path.write_bytes(buf)
        with br.open_bgen(path, verbose=False) as bg:
            assert bg.nvariants == 5
            assert bg.nsamples == 3
