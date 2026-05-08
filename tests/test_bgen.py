"""
Unit tests for vcztools.bgen.

These exercise the encoder layers (header, sample-id block, genotype
block, variant block, .sample text, .bgi index, end-to-end write_bgen)
directly against in-memory VCZ groups built with
:func:`tests.vcz_builder.make_vcz`. No external BGEN tooling.

Round-trip checks parse the bytes produced by ``write_bgen`` with a
small in-test reader (see :func:`_read_bgen_back`) so the tests do not
depend on a third-party BGEN library.
"""

import logging
import sqlite3
import struct
import tempfile
import zlib

import numpy as np
import pytest

from tests import vcz_builder
from vcztools import bgen, regions, retrieval


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
# Tiny in-test BGEN reader for round-trip verification.
# Sufficient for layout-2/zlib/8-bit/biallelic/diploid output; not a
# general-purpose BGEN parser.
# ---------------------------------------------------------------------------


def _read_bgen_back(path):
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    (offset,) = struct.unpack_from("<I", data, pos)
    pos += 4
    header_start = pos
    (header_length, n_variants, n_samples) = struct.unpack_from("<III", data, pos)
    pos += 12
    magic = data[pos : pos + 4]
    assert magic == b"bgen"
    pos += 4
    (flags,) = struct.unpack_from("<I", data, pos)
    pos += 4
    assert pos - header_start == header_length

    sample_ids = []
    if flags & bgen.SAMPLE_IDS_PRESENT:
        (block_length, n_samples_check) = struct.unpack_from("<II", data, pos)
        block_end = pos + block_length
        pos += 8
        assert n_samples_check == n_samples
        for _ in range(n_samples):
            (id_len,) = struct.unpack_from("<H", data, pos)
            pos += 2
            sample_ids.append(data[pos : pos + id_len].decode("utf-8"))
            pos += id_len
        assert pos == block_end

    variants = []
    first_variant_pos = 4 + offset
    assert pos == first_variant_pos

    for _ in range(n_variants):
        block_start = pos
        (id_len,) = struct.unpack_from("<H", data, pos)
        pos += 2
        varid = data[pos : pos + id_len].decode("utf-8")
        pos += id_len
        (rsid_len,) = struct.unpack_from("<H", data, pos)
        pos += 2
        rsid = data[pos : pos + rsid_len].decode("utf-8")
        pos += rsid_len
        (chr_len,) = struct.unpack_from("<H", data, pos)
        pos += 2
        chrom = data[pos : pos + chr_len].decode("utf-8")
        pos += chr_len
        (position,) = struct.unpack_from("<I", data, pos)
        pos += 4
        (k,) = struct.unpack_from("<H", data, pos)
        pos += 2
        alleles = []
        for _ in range(k):
            (a_len,) = struct.unpack_from("<I", data, pos)
            pos += 4
            alleles.append(data[pos : pos + a_len].decode("utf-8"))
            pos += a_len
        (c_size, d_size) = struct.unpack_from("<II", data, pos)
        pos += 8
        compressed = data[pos : pos + (c_size - 4)]
        pos += c_size - 4
        decompressed = zlib.decompress(compressed)
        assert len(decompressed) == d_size

        gpos = 0
        (n_geno_samples,) = struct.unpack_from("<I", decompressed, gpos)
        gpos += 4
        (n_alleles,) = struct.unpack_from("<H", decompressed, gpos)
        gpos += 2
        p_min = decompressed[gpos]
        gpos += 1
        p_max = decompressed[gpos]
        gpos += 1
        ploidy_bytes = decompressed[gpos : gpos + n_geno_samples]
        gpos += n_geno_samples
        phased = decompressed[gpos]
        gpos += 1
        bits = decompressed[gpos]
        gpos += 1
        prob_bytes = decompressed[gpos : gpos + 2 * n_geno_samples]

        variants.append(
            {
                "varid": varid,
                "rsid": rsid,
                "chrom": chrom,
                "position": position,
                "alleles": alleles,
                "n_samples": n_geno_samples,
                "n_alleles": n_alleles,
                "p_min": p_min,
                "p_max": p_max,
                "ploidy": ploidy_bytes,
                "phased": phased,
                "bits": bits,
                "probs": prob_bytes,
                "block_start": block_start,
                "block_size": pos - block_start,
            }
        )

    return {
        "n_variants": n_variants,
        "n_samples": n_samples,
        "flags": flags,
        "sample_ids": sample_ids,
        "variants": variants,
    }


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
    def test_round_trip(self):
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
        )
        # Parse it back via the in-test reader, by faking a one-variant
        # file: header + this block.
        sample_block = bgen._build_sample_id_block(["s0", "s1"])
        header = bgen._build_header(1, 2, sample_block)
        with tempfile.NamedTemporaryFile(suffix=".bgen", delete=False) as f:
            f.write(header)
            f.write(block)
            path = f.name
        parsed = _read_bgen_back(path)
        v = parsed["variants"][0]
        assert v["varid"] == "rs1"
        assert v["rsid"] == "rs1"
        assert v["chrom"] == "chr1"
        assert v["position"] == 12345
        assert v["alleles"] == ["A", "T"]
        assert v["n_samples"] == 2
        assert v["n_alleles"] == 2
        assert v["phased"] == 0
        assert v["bits"] == 8


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


def _decode_unphased_hard_call(b0, b1):
    """Inverse of the unphased hard-call mapping: returns (a, b) for
    the called genotype, or (-1, -1) if both bytes are zero (which we
    treat as "ambiguous"; missing samples are detected via the ploidy
    byte, not the probability bytes)."""
    if b0 == 0xFF and b1 == 0x00:
        return (0, 0)
    if b0 == 0x00 and b1 == 0xFF:
        return (0, 1)
    if b0 == 0x00 and b1 == 0x00:
        return (1, 1)
    raise ValueError(f"unexpected probability bytes: {b0!r} {b1!r}")


class TestWriteBgenEndToEnd:
    def test_minimal_round_trip(self, tmp_path):
        # 3 variants, 4 samples, biallelic, diploid; varied genotypes.
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
        out = tmp_path / "out"
        bgen.write_bgen(reader, out)
        bgen_path = out.with_suffix(".bgen")
        sample_path = out.with_suffix(".sample")
        bgi_path = bgen_path.with_suffix(".bgen.bgi")
        assert bgen_path.exists()
        assert sample_path.exists()
        assert bgi_path.exists()

        parsed = _read_bgen_back(bgen_path)
        assert parsed["n_variants"] == 3
        assert parsed["n_samples"] == 4
        assert parsed["sample_ids"] == ["sample_0", "sample_1", "sample_2", "sample_3"]
        assert len(parsed["variants"]) == 3

        v0 = parsed["variants"][0]
        assert v0["chrom"] == "chr1"
        assert v0["position"] == 100
        assert v0["alleles"] == ["A", "T"]
        # Ploidy: sample 3 was missing.
        assert list(v0["ploidy"]) == [0x02, 0x02, 0x02, 0x82]
        probs = list(v0["probs"])
        assert _decode_unphased_hard_call(probs[0], probs[1]) == (0, 0)
        assert _decode_unphased_hard_call(probs[2], probs[3]) == (0, 1)
        assert _decode_unphased_hard_call(probs[4], probs[5]) == (1, 1)
        # Missing sample probs zeroed.
        assert (probs[6], probs[7]) == (0, 0)

        # Variant 2 has reversed-order het (1,0) which is encoded
        # identically to (0,1) in the unphased path. Last sample is
        # half-missing and therefore fully missing.
        v2 = parsed["variants"][2]
        assert list(v2["ploidy"]) == [0x02, 0x02, 0x02, 0x82]
        probs = list(v2["probs"])
        assert _decode_unphased_hard_call(probs[0], probs[1]) == (1, 1)
        assert _decode_unphased_hard_call(probs[2], probs[3]) == (0, 1)
        assert _decode_unphased_hard_call(probs[4], probs[5]) == (0, 1)

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
        parsed = _read_bgen_back(bgen_path)
        conn = sqlite3.connect(str(bgi_path))
        try:
            rows = conn.execute(
                "SELECT file_start_position, size_in_bytes "
                "FROM Variant ORDER BY position"
            ).fetchall()
        finally:
            conn.close()
        for v, (start, size) in zip(parsed["variants"], rows):
            assert v["block_start"] == start
            assert v["block_size"] == size

    def test_phased_round_trip(self, tmp_path):
        # Phased: (0,1) and (1,0) must distinguish.
        G = np.array(
            [
                [[0, 1], [1, 0], [0, 0], [1, 1]],
            ],
            dtype=np.int8,
        )
        phased = np.ones((1, 4), dtype=bool)
        reader = _build_reader(
            num_variants=1,
            num_samples=4,
            call_genotype=G,
            call_fields={"genotype_phased": phased},
        )
        out = tmp_path / "p"
        bgen.write_bgen(reader, out)
        parsed = _read_bgen_back(out.with_suffix(".bgen"))
        v = parsed["variants"][0]
        assert v["phased"] == 1
        # Per-haplotype P(allele 0):
        # (0,1) -> [0xFF, 0x00]; (1,0) -> [0x00, 0xFF];
        # (0,0) -> [0xFF, 0xFF]; (1,1) -> [0x00, 0x00]
        assert list(v["probs"]) == [
            0xFF,
            0x00,
            0x00,
            0xFF,
            0xFF,
            0xFF,
            0x00,
            0x00,
        ]

    def test_unphased_when_phased_field_absent(self, tmp_path):
        reader = _build_reader(num_variants=1, num_samples=2)
        out = tmp_path / "x"
        bgen.write_bgen(reader, out)
        parsed = _read_bgen_back(out.with_suffix(".bgen"))
        assert parsed["variants"][0]["phased"] == 0

    def test_mixed_phase_degrades_to_unphased(self, tmp_path, caplog):
        G = np.array([[[0, 1], [1, 0]]], dtype=np.int8)
        phased = np.array([[True, False]], dtype=bool)
        reader = _build_reader(
            num_variants=1,
            num_samples=2,
            call_genotype=G,
            call_fields={"genotype_phased": phased},
        )
        out = tmp_path / "x"
        with caplog.at_level(logging.WARNING, logger="vcztools.bgen"):
            bgen.write_bgen(reader, out)
        parsed = _read_bgen_back(out.with_suffix(".bgen"))
        assert parsed["variants"][0]["phased"] == 0
        assert any("mixed phase" in record.getMessage() for record in caplog.records)

    def test_multi_allelic_raises(self, tmp_path):
        reader = _build_reader(
            num_variants=1,
            num_samples=2,
            alleles=[("A", "T", "G")],
        )
        with pytest.raises(ValueError, match="Multi-allelic"):
            bgen.write_bgen(reader, tmp_path / "x")

    def test_variant_id_field_used(self, tmp_path):
        reader = _build_reader(
            num_variants=2,
            num_samples=1,
            variant_id=["rsX", "rsY"],
        )
        out = tmp_path / "x"
        bgen.write_bgen(reader, out)
        parsed = _read_bgen_back(out.with_suffix(".bgen"))
        rsids = [v["rsid"] for v in parsed["variants"]]
        assert rsids == ["rsX", "rsY"]

    def test_empty_variant_id_normalised_to_dot(self, tmp_path):
        # An empty-string variant_id (VCF "." → empty in VCZ) should
        # be emitted as "." in BGEN, matching generate_bim.
        reader = _build_reader(
            num_variants=2,
            num_samples=1,
            variant_id=["", "rsY"],
        )
        out = tmp_path / "x"
        bgen.write_bgen(reader, out)
        parsed = _read_bgen_back(out.with_suffix(".bgen"))
        rsids = [v["rsid"] for v in parsed["variants"]]
        assert rsids == [".", "rsY"]

    def test_monomorphic_alt_normalised_to_dot(self, tmp_path):
        # A monomorphic-REF variant has an empty ALT slot in VCZ; emit
        # "." in BGEN to match generate_bim's convention.
        reader = _build_reader(
            num_variants=1,
            num_samples=1,
            alleles=[("A",)],
        )
        out = tmp_path / "x"
        bgen.write_bgen(reader, out)
        parsed = _read_bgen_back(out.with_suffix(".bgen"))
        assert parsed["variants"][0]["alleles"] == ["A", "."]

    def test_sample_subset(self, tmp_path):
        reader = _build_reader(num_variants=2, num_samples=4)
        reader.set_samples([0, 2])
        out = tmp_path / "x"
        bgen.write_bgen(reader, out)
        parsed = _read_bgen_back(out.with_suffix(".bgen"))
        assert parsed["sample_ids"] == ["sample_0", "sample_2"]
        assert parsed["n_samples"] == 2
        for v in parsed["variants"]:
            assert v["n_samples"] == 2
            assert len(v["ploidy"]) == 2
            assert len(v["probs"]) == 4

    def test_sample_subset_to_empty_via_filter(self, tmp_path):
        # An empty axis isn't a configuration vcz_builder supports
        # directly; but we can produce a reader-side empty axis by
        # selecting no variants via set_variants. With zero variants
        # the BGEN file should still be valid (header only).
        reader = _build_reader(num_variants=2, num_samples=2)
        empty_plan = regions.chunk_plan_from_indexes(
            np.array([], dtype=np.int64),
            variants_chunk_size=reader.variants_chunk_size,
        )
        reader.set_variants(empty_plan)
        out = tmp_path / "x"
        bgen.write_bgen(reader, out)
        parsed = _read_bgen_back(out.with_suffix(".bgen"))
        assert parsed["n_variants"] == 0
        assert parsed["n_samples"] == 2
        assert parsed["variants"] == []
