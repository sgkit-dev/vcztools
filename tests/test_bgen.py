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
import pytest

from tests import vcz_builder
from vcztools import bcftools_filter, bgen, regions, retrieval


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


class TestEncodeGenotypeBlock:
    def test_unphased_basic(self):
        # Three samples: hom-ref, het, hom-alt.
        G = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.int8)
        block = bgen._encode_genotype_block(G, phased=False)
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
        block = bgen._encode_genotype_block(G, phased=False)
        # (1,0) is also het: P(00)=0, P(01)=0xFF.
        assert block[-2:] == bytes([0x00, 0xFF])

    def test_missing_genotype(self):
        G = np.array([[-1, -1], [0, -1], [0, 1]], dtype=np.int8)
        block = bgen._encode_genotype_block(G, phased=False)
        # Ploidy bytes: 0x82 (missing), 0x82 (any neg → missing), 0x02.
        assert block[8:11] == bytes([0x82, 0x82, 0x02])
        # Probability bytes for missing samples are zeroed; het stays.
        assert block[13:19] == bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0xFF])

    def test_haploid_padding_treated_as_missing(self):
        G = np.array([[0, -2]], dtype=np.int8)
        block = bgen._encode_genotype_block(G, phased=False)
        assert block[8] == 0x82
        assert block[-2:] == bytes([0x00, 0x00])

    def test_phased(self):
        # Phased het variants distinguish (0,1) from (1,0).
        G = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int8)
        block = bgen._encode_genotype_block(G, phased=True)
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
        block = bgen._encode_genotype_block(G, phased=False)
        # Header (8 bytes) + 0 ploidy bytes + phased + B (2 bytes) + 0 probs
        assert len(block) == 10
        (n, k, p_min, p_max) = struct.unpack_from("<IHBB", block, 0)
        assert (n, k, p_min, p_max) == (0, 2, 2, 2)


class TestEncodeVariantBlock:
    def test_round_trip(self, tmp_path):
        G = np.array([[0, 0], [0, 1]], dtype=np.int8)
        block = bgen._encode_variant_block(
            varid="rs1",
            rsid="rs1",
            chrom="chr1",
            position=12345,
            allele1="A",
            allele2="T",
            num_samples=2,
            genotypes=G,
            phased=False,
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
        np.testing.assert_array_equal(probs[0, 0], [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(probs[1, 0], [0.0, 1.0, 0.0])


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


class TestBgiIndex:
    def test_basic(self, tmp_path):
        bgen_path = tmp_path / "out.bgen"
        bgen_path.write_bytes(b"\x00" * 1500)
        entries = [
            bgen._BgiEntry(
                chromosome="chr1",
                position=100,
                rsid="rs1",
                number_of_alleles=2,
                allele1="A",
                allele2="T",
                file_start_position=24,
                size_in_bytes=128,
            ),
            bgen._BgiEntry(
                chromosome="chr1",
                position=200,
                rsid="rs2",
                number_of_alleles=2,
                allele1="C",
                allele2="G",
                file_start_position=24 + 128,
                size_in_bytes=140,
            ),
        ]
        bgi_path = tmp_path / "out.bgen.bgi"
        bgen._write_bgi_index(bgi_path, bgen_path, entries)
        conn = sqlite3.connect(str(bgi_path))
        try:
            rows = conn.execute(
                "SELECT chromosome, position, rsid, number_of_alleles, "
                "allele1, allele2, file_start_position, size_in_bytes "
                "FROM Variant ORDER BY position"
            ).fetchall()
            assert rows == [
                ("chr1", 100, "rs1", 2, "A", "T", 24, 128),
                ("chr1", 200, "rs2", 2, "C", "G", 152, 140),
            ]
            metadata = conn.execute(
                "SELECT filename, file_size, length(first_1000_bytes) FROM Metadata"
            ).fetchone()
            assert metadata == ("out.bgen", 1500, 1000)
        finally:
            conn.close()

    def test_index_overwritten(self, tmp_path):
        bgen_path = tmp_path / "out.bgen"
        bgen_path.write_bytes(b"\x00" * 100)
        bgi_path = tmp_path / "out.bgen.bgi"
        # First write
        bgen._write_bgi_index(bgi_path, bgen_path, [])
        # Second write should not raise (PK conflict would if not unlinked).
        bgen._write_bgi_index(bgi_path, bgen_path, [])


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
        np.testing.assert_array_equal(probs[0, 0], [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(probs[1, 0], [0.0, 1.0, 0.0])
        np.testing.assert_array_equal(probs[2, 0], [0.0, 0.0, 1.0])

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
            assert list(bg.samples) == [
                "sample_0",
                "sample_1",
                "sample_2",
                "sample_3",
            ]
            assert bg.chromosomes[0] == "chr1"
            assert list(bg.positions) == [100, 200, 300]
            assert [str(a).split(",") for a in bg.allele_ids] == [
                ["A", "T"],
                ["C", "G"],
                ["G", "A"],
            ]
            probs, missing = bg.read(return_missings=True)

        # Variant 0: hom-ref, het, hom-alt, missing.
        np.testing.assert_array_equal(missing[:, 0], [False, False, False, True])
        np.testing.assert_array_equal(probs[0, 0], [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(probs[1, 0], [0.0, 1.0, 0.0])
        np.testing.assert_array_equal(probs[2, 0], [0.0, 0.0, 1.0])
        assert np.isnan(probs[3, 0]).all()

        # Variant 2: hom-alt, reversed het (1,0), het (0,1),
        # half-missing (treated as fully missing).
        np.testing.assert_array_equal(missing[:, 2], [False, False, False, True])
        np.testing.assert_array_equal(probs[0, 2], [0.0, 0.0, 1.0])
        np.testing.assert_array_equal(probs[1, 2], [0.0, 1.0, 0.0])
        np.testing.assert_array_equal(probs[2, 2], [0.0, 1.0, 0.0])
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
        np.testing.assert_array_equal(hap1, G[..., 0])
        np.testing.assert_array_equal(hap2, G[..., 1])

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
            assert [str(r) for r in bg.rsids] == ["rsX", "rsY"]

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
            assert [str(r) for r in bg.rsids] == [".", "rsY"]

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
            assert list(bg.samples) == ["sample_0", "sample_2"]
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
            np.testing.assert_array_equal(probs[0, v], [0.0, 0.0, 1.0])
            np.testing.assert_array_equal(probs[1, v], [1.0, 0.0, 0.0])
            np.testing.assert_array_equal(probs[2, v], [1.0, 0.0, 0.0])

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
            assert [str(r) for r in bg.rsids] == expected

    def test_rsids_dot_when_variant_id_absent(self, write_to_bgen):
        # A VCZ without a ``variant_id`` field emits all rsids as ".".
        # For BgenEncoder this triggers the 1-byte rsid_max default;
        # for write_bgen the rsid literal is "." (1 byte).
        reader = _build_reader(num_variants=3, num_samples=2)
        assert "variant_id" not in reader.field_names
        path = write_to_bgen(reader)
        with br.open_bgen(path, verbose=False) as bg:
            assert [str(r) for r in bg.rsids] == [".", ".", "."]

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


class TestPadToFixedLength:
    def test_pads_with_nul(self):
        assert (
            bgen._pad_to_fixed_length("rs1", 8, field_name="rsid")
            == b"rs1\x00" * 1 + b"\x00" * 4
        )

    def test_exact_length(self):
        assert bgen._pad_to_fixed_length("ABCD", 4, field_name="x") == b"ABCD"

    def test_overflow_raises(self):
        with pytest.raises(ValueError, match="rsid 'too_long'"):
            bgen._pad_to_fixed_length("too_long", 4, field_name="rsid")

    def test_overflow_message_names_field_and_max(self):
        with pytest.raises(ValueError, match="exceeds configured max of 3"):
            bgen._pad_to_fixed_length("abcd", 3, field_name="varid")

    def test_unicode_multibyte_counts_bytes(self):
        # "é" = 2 bytes in UTF-8; fits in 4 but not in 1.
        assert bgen._pad_to_fixed_length("é", 4, field_name="x") == b"\xc3\xa9\x00\x00"
        with pytest.raises(ValueError, match="exceeds configured max"):
            bgen._pad_to_fixed_length("é", 1, field_name="x")


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
            assert list(bg.positions) == [101, 103, 105]

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
            assert list(bg.samples) == ["sample_0", "sample_2", "sample_4"]


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
