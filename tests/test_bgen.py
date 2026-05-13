"""
Unit tests for vcztools.bgen.

These exercise the encoder layers (header, sample-id block, genotype
block, variant block, .sample text, .bgi index, end-to-end write_bgen)
directly against in-memory VCZ groups built with
:func:`tests.vcz_builder.make_vcz`.

Round-trip checks parse the bytes produced by ``write_bgen`` with the
``bgen-reader`` reference reader.
"""

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

_REFERENCE_PLOIDY_DIPLOID = 0x02
_REFERENCE_PLOIDY_MISSING = 0x82


def _build_geno_blocks_reference(G, phased):
    """Pure-NumPy reference implementation of the BGEN Layout-2
    genotype-block builder, kept verbatim from the original
    ``vcztools.bgen._build_geno_blocks`` so the C kernel can be
    validated against it.

    Layout per row (Layout 2 / 8-bit / biallelic / diploid):

        uint32 N            n_samples
        uint16 K = 2        n_alleles
        uint8  P_min = 2
        uint8  P_max = 2
        N bytes             ploidy/missing per sample
        uint8  phased       per-variant flag
        uint8  B = 8        bits per probability
        2*N bytes           per-sample probability bytes (B0, B1 interleaved)
    """
    v, s, _ = G.shape
    a = G[..., 0]
    b = G[..., 1]
    missing = (a < 0) | (b < 0)

    a_zero = a == 0
    b_zero = b == 0
    homref = a_zero & b_zero
    het = (a_zero & (b == 1)) | ((a == 1) & b_zero)
    phased_row = phased[:, None]
    b0_bit = np.where(phased_row, a_zero, homref)
    b1_bit = np.where(phased_row, b_zero, het)
    valid = ~missing
    B0 = (b0_bit & valid).view(np.uint8) * np.uint8(0xFF)
    B1 = (b1_bit & valid).view(np.uint8) * np.uint8(0xFF)

    ploidy = np.where(
        missing,
        np.uint8(_REFERENCE_PLOIDY_MISSING),
        np.uint8(_REFERENCE_PLOIDY_DIPLOID),
    )

    geno_blocks = np.empty((v, 10 + 3 * s), dtype=np.uint8)
    header = struct.pack("<IHBB", s, 2, 2, 2)
    geno_blocks[:, 0:8] = np.frombuffer(header, dtype=np.uint8)
    geno_blocks[:, 8 : 8 + s] = ploidy
    geno_blocks[:, 8 + s] = phased.astype(np.uint8)
    geno_blocks[:, 8 + s + 1] = bgen.BITS_PER_PROB
    geno_blocks[:, 8 + s + 2 :: 2] = B0
    geno_blocks[:, 8 + s + 3 :: 2] = B1
    return geno_blocks


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


class TestGenerateSample:
    def test_minimal(self):
        reader = _build_reader(num_samples=3)
        text = bgen.generate_sample(reader)
        lines = text.splitlines()
        assert lines[0] == "ID_1 ID_2 missing"
        assert lines[1] == "0 0 0"
        assert lines[2] == "sample_0 sample_0 0"
        assert lines[3] == "sample_1 sample_1 0"
        assert lines[4] == "sample_2 sample_2 0"
        assert text.endswith("\n")

    def test_zero_samples(self):
        reader = _build_reader(num_samples=0)
        text = bgen.generate_sample(reader)
        assert text == "ID_1 ID_2 missing\n0 0 0\n"

    def test_whitespace_rejected(self):
        reader = _build_reader(
            num_samples=2,
            sample_id=np.array(["ok", "has space"], dtype="<U16"),
        )
        with pytest.raises(ValueError, match="whitespace"):
            bgen.generate_sample(reader)

    def test_sample_subset_round_trips(self):
        reader = _build_reader(num_samples=4)
        reader.set_samples([0, 2])
        text = bgen.generate_sample(reader)
        lines = text.splitlines()
        assert lines[2] == "sample_0 sample_0 0"
        assert lines[3] == "sample_2 sample_2 0"
        assert len(lines) == 4


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


# ---------------------------------------------------------------------------
# Genotype block / variant block byte-level encoding
# ---------------------------------------------------------------------------


def _single_variant_geno_block(G_single, *, phased=False):
    """Build a one-variant chunk, run `_prepare_chunk`, and return the
    uncompressed genotype-block bytes for that variant. ``G_single``
    has shape ``(num_samples, 2)``."""
    G = G_single[None, ...]  # (1, S, 2)
    n = G_single.shape[0]
    chunk = {
        "call_genotype": G,
        "variant_allele": np.array([["A", "T"]]),
        "variant_contig": np.array([0]),
        "variant_position": np.array([100]),
    }
    if phased:
        chunk["call_genotype_phased"] = np.ones((1, n), dtype=bool)
    prep = bgen._prepare_chunk(chunk, contig_ids=np.array(["chr1"]), start=0, end=1)
    return bytes(prep.geno_blocks[0])


class TestPrepareChunkGenoBlocks:
    """Spec-level byte assertions on the uncompressed genotype-block
    bytes produced by :func:`_prepare_chunk` (`prep.geno_blocks[i]`).
    Layout 2 / 8-bit / biallelic / diploid; matches the per-variant
    byte form the BGEN spec defines."""

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

    def test_haploid_padding_treated_as_missing(self):
        G = np.array([[0, -2]], dtype=np.int8)
        block = _single_variant_geno_block(G)
        assert block[8] == 0x82
        assert block[-2:] == bytes([0x00, 0x00])

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
    the pure-NumPy ``_build_geno_blocks_reference`` over a range of
    awkward inputs; the C output must match byte-for-byte."""

    def _assert_matches(self, G, phased):
        c_out = _vcztools.encode_bgen_geno_blocks(G, phased)
        ref_out = _build_geno_blocks_reference(G, phased)
        assert c_out.dtype == np.uint8
        assert c_out.shape == (G.shape[0], 10 + 3 * G.shape[1])
        nt.assert_array_equal(c_out, ref_out)

    @pytest.mark.parametrize("phased_value", [False, True])
    def test_single_sample_all_allele_pairs(self, phased_value):
        # Sweep every (a, b) from {-2,-1,0,1,2}^2 as separate variants.
        pairs = [(a, b) for a in range(-2, 3) for b in range(-2, 3)]
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

    def test_all_haploid_padded(self):
        G = np.full((4, 6, 2), -2, dtype=np.int8)
        self._assert_matches(G, np.zeros(4, dtype=bool))

    def test_partial_missing_patterns(self):
        # One variant covers every kind of missing/half-call/normal pair.
        samples = np.array(
            [
                [-1, -1],
                [0, -1],
                [-1, 0],
                [-2, 0],
                [0, -2],
                [1, -2],
                [-2, 1],
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
        G = rng.integers(-2, 3, size=(2, num_samples, 2), dtype=np.int8)
        phased = np.array([False, True])
        self._assert_matches(G, phased)

    @pytest.mark.parametrize("num_variants", [0, 1, 2, 7, 16, 100])
    def test_unusual_variant_counts(self, num_variants):
        rng = np.random.default_rng(1)
        G = rng.integers(-2, 3, size=(num_variants, 5, 2), dtype=np.int8)
        phased = rng.integers(0, 2, size=num_variants, dtype=np.int8).astype(bool)
        self._assert_matches(G, phased)

    def test_random_uniform(self):
        rng = np.random.default_rng(42)
        G = rng.integers(-2, 3, size=(50, 50, 2), dtype=np.int8)
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
        v_idx = np.arange(num_variants)[:, None, None]
        s_idx = np.arange(num_samples)[None, :, None]
        # Mix of all five values across the (V, S) grid.
        G = ((v_idx + s_idx + np.array([[[0, 1]]])) % 5 - 2).astype(np.int8)
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

    def test_non_diploid_raises(self):
        G = np.zeros((2, 3, 3), dtype=np.int8)
        with pytest.raises(ValueError, match="diploid"):
            bgen._check_diploid(G)

    def test_diploid_ok(self):
        G = np.zeros((2, 3, 2), dtype=np.int8)
        bgen._check_diploid(G)


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


class TestWriteBgenIndex:
    """``write_bgen_index(reader, bgi_path, variant_offsets)`` — the
    single seam for ``.bgen.bgi`` generation, used by both
    :func:`write_bgen` (variable-size, cumulative-block offsets) and
    :class:`BgenEncoder` (fixed-size, formula-derived offsets via
    :attr:`BgenEncoder.variant_offsets`)."""

    def test_basic_variant_table(self, tmp_path):
        # Two variants on contig "chr1" at positions 100, 101. Synthetic
        # offsets carve out 128-byte and 140-byte blocks after a 24-byte
        # prefix.
        reader = _build_reader(num_variants=2, num_samples=2)
        variant_offsets = np.array([24, 24 + 128, 24 + 128 + 140], dtype=np.int64)
        bgi_path = tmp_path / "out.bgen.bgi"
        bgen.write_bgen_index(reader, bgi_path, variant_offsets)
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
        bgen.write_bgen_index(reader, bgi_path, variant_offsets)
        # Second write must not raise (PK conflict would if not unlinked).
        bgen.write_bgen_index(reader, bgi_path, variant_offsets)

    def test_offsets_wrong_length_raises(self, tmp_path):
        reader = _build_reader(num_variants=2, num_samples=2)
        bad = np.array([0, 100, 200, 300], dtype=np.int64)
        with pytest.raises(ValueError, match="must have shape"):
            bgen.write_bgen_index(reader, tmp_path / "x.bgi", bad)

    def test_offsets_non_integer_raises(self, tmp_path):
        reader = _build_reader(num_variants=2, num_samples=2)
        bad = np.array([0.0, 100.0, 200.0])
        with pytest.raises(ValueError, match="integer-typed"):
            bgen.write_bgen_index(reader, tmp_path / "x.bgi", bad)

    def test_variant_id_propagates_to_rsid(self, tmp_path):
        reader = _build_reader(num_variants=2, num_samples=2, variant_id=["rsA", "rsB"])
        variant_offsets = np.array([0, 100, 200], dtype=np.int64)
        bgi_path = tmp_path / "out.bgen.bgi"
        bgen.write_bgen_index(reader, bgi_path, variant_offsets)
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
            bgen.write_bgen_index(reader, tmp_path / "out.bgi", variant_offsets)

    def test_encoder_offsets_round_trip(self, tmp_path):
        # Encoder formula offsets carve up the encoder's byte stream
        # into contiguous, equal-size blocks ending at total_size.
        with _build_encoder(num_variants=5, num_samples=4) as enc:
            reader = enc._reader
            bgi_path = tmp_path / "x.bgen.bgi"
            bgen.write_bgen_index(reader, bgi_path, enc.variant_offsets)
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

        out = tmp_path / "wb"
        bgen.write_bgen(_build_reader(**kwargs), out)
        wb_bgi = out.with_suffix(".bgen").with_suffix(".bgen.bgi")
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
            bgen.write_bgen_index(enc._reader, enc_bgi, enc.variant_offsets)
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
        out = tmp_path / "out"
        bgen.write_bgen(reader, out)
        assert out.with_suffix(".bgen").exists()
        assert out.with_suffix(".sample").exists()
        assert out.with_suffix(".bgen").with_suffix(".bgen.bgi").exists()

    def test_sample_text_matches(self, tmp_path):
        reader = _build_reader(num_variants=1, num_samples=2)
        out = tmp_path / "x"
        bgen.write_bgen(reader, out)
        text = (out.with_suffix(".sample")).read_text()
        assert text.startswith("ID_1 ID_2 missing\n0 0 0\n")
        assert "sample_0 sample_0 0\n" in text
        assert "sample_1 sample_1 0\n" in text

    def test_bgi_offsets_consistent_with_bgen(self, tmp_path):
        reader = _build_reader(num_variants=3, num_samples=2)
        out = tmp_path / "x"
        bgen.write_bgen(reader, out)
        bgen_path = out.with_suffix(".bgen")
        bgi_path = bgen_path.with_suffix(".bgen.bgi")

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
        out = tmp_path / "x"
        bgen.write_bgen(reader, out)
        bgen_path = out.with_suffix(".bgen")
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
        out = tmp_path / f"out_l{level}"
        bgen.write_bgen(reader, out, compression_level=level)
        with br.open_bgen(out.with_suffix(".bgen"), verbose=False) as bgen_file:
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
        out0 = tmp_path / "l0"
        bgen.write_bgen(_build_reader(**reader_kwargs), out0, compression_level=0)
        out9 = tmp_path / "l9"
        bgen.write_bgen(_build_reader(**reader_kwargs), out9, compression_level=9)
        size_l0 = out0.with_suffix(".bgen").stat().st_size
        size_l9 = out9.with_suffix(".bgen").stat().st_size
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
            bgen.write_bgen(reader, tmp_path / "out")
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
        # emitted as "." in BGEN, matching generate_bim's convention.
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


class TestPrepareChunk:
    def _make_chunk(
        self,
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

    def test_variable_mode_basic(self):
        chunk = self._make_chunk()
        prep = bgen._prepare_chunk(chunk, contig_ids=np.array(["chr1"]), start=0, end=2)
        assert [bytes(b) for b in prep.varid] == [b".", b"."]
        assert [bytes(b) for b in prep.rsid] == [b".", b"."]
        assert [bytes(b) for b in prep.chrom] == [b"chr1", b"chr1"]
        assert [bytes(b) for b in prep.allele1] == [b"A", b"C"]
        assert [bytes(b) for b in prep.allele2] == [b"T", b"G"]
        np.testing.assert_array_equal(prep.position, [100, 101])
        # 2 samples → geno-block layout has phased flag at byte 8 + 2 = 10.
        # Without a call_genotype_phased field every variant is unphased.
        s = 2
        assert prep.geno_blocks[0, 8 + s] == 0
        assert prep.geno_blocks[1, 8 + s] == 0
        assert prep.mixed_phase_count == 0

    def test_fixed_mode_pads_to_max_len(self):
        chunk = self._make_chunk(variant_id=["rsX", "rsY"])
        prep = bgen._prepare_chunk(
            chunk,
            contig_ids=np.array(["chr1"]),
            start=0,
            end=2,
            varid_max_len=8,
            rsid_max_len=8,
            chrom_max_len=4,
            allele_max_len=2,
        )
        # Fixed mode returns (N, max_len) uint8 rows.
        assert prep.varid.shape == (2, 8)
        assert bytes(prep.varid[0]) == b"rsX\x00\x00\x00\x00\x00"
        assert bytes(prep.chrom[0]) == b"chr1"
        assert bytes(prep.allele1[0]) == b"A\x00"
        # rsid is propagated from variant_id; varid mirrors rsid.
        assert bytes(prep.rsid[0]) == b"rsX\x00\x00\x00\x00\x00"

    def test_missing_rsid_normalised_to_dot(self):
        chunk = self._make_chunk(variant_id=["", "rsY"])
        prep = bgen._prepare_chunk(chunk, contig_ids=np.array(["chr1"]), start=0, end=2)
        assert [bytes(b) for b in prep.rsid] == [b".", b"rsY"]

    def test_monomorphic_allele2_normalised_to_dot(self):
        chunk = self._make_chunk(alleles=np.array([["A", ""], ["C", "G"]]))
        prep = bgen._prepare_chunk(chunk, contig_ids=np.array(["chr1"]), start=0, end=2)
        assert [bytes(b) for b in prep.allele2] == [b".", b"G"]

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
        chunk = self._make_chunk(alleles=alleles, phased=phased, num_samples=2)
        prep = bgen._prepare_chunk(chunk, contig_ids=np.array(["chr1"]), start=0, end=3)
        # Phased flag lives at byte 8 + num_samples in each geno-block row.
        flags = prep.geno_blocks[:, 8 + 2]
        np.testing.assert_array_equal(flags, [1, 0, 0])
        assert prep.mixed_phase_count == 1

    def test_varid_overflow_checked_before_rsid(self):
        # Both rsid_max_len and varid_max_len would overflow, but the
        # varid check fires first.
        chunk = self._make_chunk(variant_id=["x" * 100, "x" * 100])
        with pytest.raises(ValueError, match="varid"):
            bgen._prepare_chunk(
                chunk,
                contig_ids=np.array(["chr1"]),
                start=0,
                end=2,
                varid_max_len=8,
                rsid_max_len=64,
            )

    def test_rejects_multiallelic(self):
        alleles = np.array([["A", "T", "G"]])
        chunk = self._make_chunk(alleles=alleles)
        with pytest.raises(ValueError, match="Multi-allelic"):
            bgen._prepare_chunk(chunk, contig_ids=np.array(["chr1"]), start=0, end=1)


class TestBgenEncoderMetadata:
    def test_num_variants_and_samples(self):
        with _build_encoder(num_variants=4, num_samples=5) as enc:
            assert enc.num_variants == 4
            assert enc.num_samples == 5

    def test_bytes_per_variant_formula(self):
        # bpv = 28 + vmax + rmax + cmax + 2*amax + zlib_stored(10 + 3*N).
        # Default _build_reader has no variant_id, so rsid_max=varid_max=1;
        # default contig "chr1" → chrom_max=4; allele_max=1.
        with _build_encoder(num_variants=2, num_samples=4) as enc:
            geno_size = 10 + 3 * 4
            expected = 28 + 1 + 1 + 4 + 2 + len(zlib.compress(b"\x00" * geno_size, 0))
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
            geno_size = 10 + 3 * 2
            expected = 28 + 16 + 16 + 4 + 4 + len(zlib.compress(b"\x00" * geno_size, 0))
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
