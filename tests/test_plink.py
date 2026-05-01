"""
Unit tests for vcztools.plink.

These exercise the four layers of the writer (encode_genotypes wrapper,
_compute_alleles, generate_fam, generate_bim, end-to-end Writer) directly
against in-memory VCZ groups built with :func:`tests.vcz_builder.make_vcz`.
No PLINK binary, no roundtripping.
"""

import math
import pathlib

import numpy as np
import pandas as pd
import pytest

from tests import vcz_builder
from vcztools import bcftools_filter, plink, retrieval

# ---------------------------------------------------------------------------
# Python reference encoder used to cross-check the C-level encoder.
# ---------------------------------------------------------------------------


def _encode_genotypes_row(g, allele_1, allele_2):
    # Missing genotype: 01 in PLINK format
    # Homozygous allele 1: 00 in PLINK format
    # Homozygous allele 2: 11 in PLINK format
    # Heterozygous: 10 in PLINK format
    HOM_A1 = 0b00
    HOM_A2 = 0b11
    HET = 0b10
    MISSING = 0b01

    num_samples = g.shape[0]
    assert g.shape[1] == 2
    bytes_per_variant = (num_samples + 3) // 4
    buff = bytearray(bytes_per_variant)
    for j in range(num_samples):
        byte_idx = j // 4
        bit_pos = (j % 4) * 2
        code = MISSING
        a, b = g[j]
        if b == -2:
            # Treated as a haploid call by plink
            if a == allele_1:
                code = HOM_A1
            elif a == allele_2:
                code = HOM_A2
        else:
            if a == allele_1:
                if b == allele_1:
                    code = HOM_A1
                elif b == allele_2:
                    code = HET
            elif a == allele_2:
                if b == allele_2:
                    code = HOM_A2
                elif b == allele_1:
                    code = HET
            if allele_1 == -1 and (code == HOM_A1 or code == HET):
                code = MISSING
        mask = ~(0b11 << bit_pos)
        buff[byte_idx] = (buff[byte_idx] & mask) | (code << bit_pos)
    return buff


def encode_genotypes_reference(G, a12_allele=None):
    G = np.array(G, dtype=np.int8)
    if a12_allele is None:
        a12_allele = np.zeros((G.shape[0], 2), dtype=G.dtype)
        a12_allele[:, 0] = 1
    assert G.shape[0] == a12_allele.shape[0]
    assert G.shape[2] == 2
    buff = bytearray()
    for j in range(len(G)):
        buff.extend(_encode_genotypes_row(G[j], *a12_allele[j]))
    return bytes(buff)


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _build_reader(*, num_variants=2, num_samples=3, **overrides):
    """Return a VczReader over an in-memory VCZ group with sensible defaults."""
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


def _parse_fam(text):
    if text == "":
        return pd.DataFrame()
    return pd.read_csv(
        pd.io.common.StringIO(text),
        sep="\t",
        header=None,
        names=["FID", "IID", "Father", "Mother", "Sex", "Pheno"],
        dtype=str,
    )


def _parse_bim(text):
    if text == "":
        return pd.DataFrame()
    return pd.read_csv(
        pd.io.common.StringIO(text),
        sep="\t",
        header=None,
        names=["Chrom", "ID", "CM", "Pos", "A1", "A2"],
        dtype={"Chrom": str, "ID": str, "CM": int, "Pos": int, "A1": str, "A2": str},
    )


# ---------------------------------------------------------------------------
# Low-level encoder: parity with Python reference.
# ---------------------------------------------------------------------------


class TestEncodeGenotypesPython:
    @pytest.mark.parametrize(
        "genotypes",
        [
            [
                [[0, 0], [0, 1], [0, 0]],
            ],
            [
                [[0, 0], [0, 1], [0, 0]],
                [[1, 0], [1, 1], [0, -2]],
                [[1, 1], [0, 1], [-1, -1]],
            ],
            [
                [[0, 0], [0, 1], [0, 0], [0, 1]],
                [[0, 0], [0, 1], [0, 0], [0, 1]],
            ],
            [
                [[0, 0], [0, 1], [0, 0], [0, 1], [1, 1]],
                [[0, 0], [0, 1], [0, 0], [0, 1], [-1, -2]],
                [[1, 0], [-3, 1], [0, 0], [0, 1], [-1, -2]],
                [[0, 1], [0, 1], [1, 2], [0, 1], [1, 1]],
                [[0, 0], [0, -2], [0, 3], [-2, 1], [-1, -2]],
            ],
        ],
    )
    def test_examples_default_a12(self, genotypes):
        b1 = encode_genotypes_reference(genotypes)
        b2 = plink.encode_genotypes(genotypes)
        assert b1 == b2

    @pytest.mark.parametrize(
        ("num_variants", "num_samples"),
        [
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (1, 4),
            (1, 7),
            (1, 16),
            (1, 100),
            (1, 101),
            (10, 1),
            (10, 7),
            (10, 9),
        ],
    )
    @pytest.mark.parametrize("value", [-1, 0, 1, 2])
    def test_shapes_default_a12(self, value, num_variants, num_samples):
        g = np.zeros((num_variants, num_samples, 2), dtype=np.int8) + value
        b1 = encode_genotypes_reference(g)
        b2 = plink.encode_genotypes(g)
        assert b1 == b2

    def test_returns_bytes(self):
        g = np.zeros((1, 4, 2), dtype=np.int8)
        result = plink.encode_genotypes(g)
        assert isinstance(result, bytes)

    @pytest.mark.parametrize(
        ("num_variants", "num_samples"),
        [
            (0, 0),
            (1, 1),
            (1, 4),
            (1, 5),
            (3, 7),
            (10, 100),
        ],
    )
    def test_output_length(self, num_variants, num_samples):
        g = np.zeros((num_variants, num_samples, 2), dtype=np.int8)
        result = plink.encode_genotypes(g)
        expected = ((num_samples + 3) // 4) * num_variants
        assert len(result) == expected

    @pytest.mark.parametrize("num_samples", [1, 2, 3, 5, 6, 7, 9, 13])
    def test_trailing_pad_bits_are_hom_a1(self, num_samples):
        # All MISSING (0b01) genotypes: bytes have 01 in the live slots and
        # 00 in the trailing pad slots. We can read them back to confirm.
        g = -np.ones((1, num_samples, 2), dtype=np.int8)
        encoded = plink.encode_genotypes(g)
        bits_used = num_samples * 2
        bits_total = len(encoded) * 8
        if bits_used == bits_total:
            return  # no padding to inspect
        # Inspect the last byte's high (pad) bits.
        last = encoded[-1]
        live_in_last = num_samples - 4 * (len(encoded) - 1)
        pad_mask = 0xFF & ~((1 << (live_in_last * 2)) - 1)
        assert (last & pad_mask) == 0

    def test_dimension_mismatch_raises(self):
        g = np.zeros((2, 3, 2), dtype=np.int8)
        a12 = np.zeros((1, 2), dtype=np.int8)
        with pytest.raises(ValueError, match="same first dimension"):
            plink.encode_genotypes(g, a12)

    def test_non_diploid_raises(self):
        g = np.zeros((2, 3, 3), dtype=np.int8)
        with pytest.raises(ValueError, match="diploid"):
            plink.encode_genotypes(g)


class TestEncoderPythonReferenceParity:
    """Parity tests that go beyond the default a12 = (1, 0)."""

    @pytest.mark.parametrize(
        "a12",
        [
            np.array([[1, 0]], dtype=np.int8),
            np.array([[0, 1]], dtype=np.int8),
            np.array([[-1, 0]], dtype=np.int8),  # single-allele site
        ],
    )
    @pytest.mark.parametrize(
        "genotypes",
        [
            [[[0, 0], [1, 1], [0, 1], [-1, -1], [0, -2]]],
            [[[1, 0], [0, 1], [-2, -2], [1, -1], [-1, 0]]],
        ],
    )
    def test_vary_a12(self, genotypes, a12):
        b1 = encode_genotypes_reference(genotypes, a12)
        b2 = plink.encode_genotypes(genotypes, a12)
        assert b1 == b2

    @pytest.mark.parametrize(
        ("num_variants", "num_samples"),
        [(1, 33), (5, 100)],
    )
    def test_arange_data(self, num_variants, num_samples):
        g = np.arange((num_variants * num_samples * 2), dtype=np.int8).reshape(
            (num_variants, num_samples, 2)
        )
        a12 = np.arange(num_variants * 2, dtype=np.int8).reshape((num_variants, 2))
        b1 = encode_genotypes_reference(g, a12)
        b2 = plink.encode_genotypes(g, a12)
        assert b1 == b2


# ---------------------------------------------------------------------------
# Chromosome normalisation (matches plink 2 --make-bed output).
# ---------------------------------------------------------------------------


class TestPlink2NormaliseChrom:
    @pytest.mark.parametrize(
        ("input_chrom", "expected"),
        [
            # Standard human chromosomes: strip "chr" prefix.
            ("chr1", "1"),
            ("chr22", "22"),
            ("chrX", "X"),
            ("chrY", "Y"),
            ("chrMT", "MT"),
            # chrM is rewritten to MT (plink 2 normalises mitochondrial).
            ("chrM", "MT"),
            # Already-normalised: pass through.
            ("1", "1"),
            ("22", "22"),
            ("X", "X"),
            ("MT", "MT"),
            # Non-standard contigs: pass through (plink 2 keeps these
            # as-is under --allow-extra-chr).
            ("chrFoo", "chrFoo"),
            ("chr23", "chr23"),
            ("chrUn_random", "chrUn_random"),
            ("scaffold_42", "scaffold_42"),
        ],
    )
    def test_normalise_chrom(self, input_chrom, expected):
        assert plink._plink2_normalise_chrom(input_chrom) == expected


# ---------------------------------------------------------------------------
# _compute_alleles: A1/A2 selection logic.
# ---------------------------------------------------------------------------


class TestComputeAlleles:
    """A1/A2 derivation in plink 2 REF/ALT convention.

    All biallelic input maps to ``[[1, 0], …]`` (A1=ALT, A2=REF).
    Single-allele input maps to ``[[-1, 0]]`` (A1 = "missing"
    sentinel; .bim writer emits "."). Multi-allelic raises.
    """

    @staticmethod
    def _writer():
        return plink.Writer(reader=None, bed_path=None, fam_path=None, bim_path=None)

    @pytest.mark.parametrize(
        "alleles",
        [
            np.array([["A", "T"]], dtype="U1"),
            np.array([["G", "C"]], dtype="U1"),
            np.array([["A", "T"], ["G", "C"], ["T", "A"]], dtype="U1"),
        ],
    )
    def test_biallelic_returns_alt_then_ref(self, alleles):
        a12 = self._writer()._compute_alleles(alleles)
        expected = np.tile([1, 0], (alleles.shape[0], 1))
        np.testing.assert_array_equal(a12, expected)
        assert a12.dtype == np.int8

    def test_monomorphic_returns_minus_one(self):
        alleles = np.array([["A", ""]], dtype="U1")
        a12 = self._writer()._compute_alleles(alleles)
        np.testing.assert_array_equal(a12, [[-1, 0]])

    def test_mixed_biallelic_and_monomorphic(self):
        alleles = np.array([["A", "T"], ["G", ""], ["C", "G"]], dtype="U1")
        a12 = self._writer()._compute_alleles(alleles)
        np.testing.assert_array_equal(a12, [[1, 0], [-1, 0], [1, 0]])

    def test_multiallelic_raises(self):
        alleles = np.array([["A", "T", "C"]], dtype="U1")
        with pytest.raises(ValueError, match="Multi-allelic"):
            self._writer()._compute_alleles(alleles)

    def test_wide_store_with_only_biallelic_rows_passes(self):
        # variant_allele can have a wider second axis when the store's
        # max-allele count is >2; if no row actually uses columns past
        # index 1, _compute_alleles must accept it. This is the path
        # exercised by --max-alleles 2 against a multi-allelic store.
        alleles = np.array(
            [
                ["A", "T", "", ""],
                ["G", "C", "", ""],
                ["G", "", "", ""],
            ],
            dtype="U1",
        )
        a12 = self._writer()._compute_alleles(alleles)
        np.testing.assert_array_equal(a12, [[1, 0], [1, 0], [-1, 0]])


# ---------------------------------------------------------------------------
# generate_fam.
# ---------------------------------------------------------------------------


class TestGenerateFam:
    def test_plain_ascii_sample_ids(self):
        reader = _build_reader(num_samples=2, sample_id=["s1", "s2"])
        fam = plink.generate_fam(reader)
        # FID = "0" matches plink 2 default for `--vcf`.
        assert fam == "0\ts1\t0\t0\t0\t-9\n0\ts2\t0\t0\t0\t-9\n"

    def test_zero_samples(self):
        reader = _build_reader(num_samples=0, call_genotype=None)
        fam = plink.generate_fam(reader)
        assert fam == ""

    def test_null_samples_excluded_from_fam(self):
        # Null sample (sample_id == "") must not appear in the FAM, and the
        # FAM row count must equal the BED column count.
        reader = _build_reader(
            num_samples=3,
            sample_id=["s1", "", "s3"],
        )
        fam = plink.generate_fam(reader)
        df = _parse_fam(fam)
        assert list(df["IID"]) == ["s1", "s3"]
        # There must be no empty-IID rows.
        assert not (df["IID"] == "").any()
        assert list(df["FID"]) == ["0", "0"]

    @pytest.mark.parametrize(
        "bad_id",
        [
            "has space",
            "tab\there",
            "newline\nhere",
        ],
    )
    def test_whitespace_in_sample_id_raises(self, bad_id):
        reader = _build_reader(num_samples=2, sample_id=["ok", bad_id])
        with pytest.raises(ValueError, match="whitespace"):
            plink.generate_fam(reader)

    def test_long_sample_id_round_trips(self):
        long_id = "x" * 50
        reader = _build_reader(num_samples=2, sample_id=[long_id, "s2"])
        fam = plink.generate_fam(reader)
        df = _parse_fam(fam)
        assert df["IID"].iloc[0] == long_id
        assert df["FID"].iloc[0] == "0"

    def test_sample_subset_reflected_in_fam(self):
        reader = _build_reader(num_samples=4, sample_id=["s0", "s1", "s2", "s3"])
        reader.set_samples([2, 0])
        fam = plink.generate_fam(reader)
        df = _parse_fam(fam)
        assert list(df["IID"]) == ["s2", "s0"]
        assert list(df["FID"]) == ["0", "0"]


# ---------------------------------------------------------------------------
# generate_bim.
# ---------------------------------------------------------------------------


class TestGenerateBim:
    def test_standard_biallelic_no_variant_id(self):
        reader = _build_reader(
            num_variants=2,
            variant_position=[100, 200],
            alleles=[("A", "T"), ("G", "C")],
            num_samples=1,
            call_genotype=np.zeros((2, 1, 2), dtype=np.int8),
        )
        a12 = np.array([[1, 0], [1, 0]])
        bim = plink.generate_bim(reader, a12)
        df = _parse_bim(bim)
        # Default contig is "chr1"; plink 2 normalises this to "1".
        assert list(df["Chrom"]) == ["1", "1"]
        assert list(df["ID"]) == [".", "."]
        assert list(df["CM"]) == [0, 0]
        assert list(df["Pos"]) == [100, 200]
        assert list(df["A1"]) == ["T", "C"]
        assert list(df["A2"]) == ["A", "G"]

    def test_with_variant_id(self):
        reader = _build_reader(
            num_variants=3,
            num_samples=1,
            call_genotype=np.zeros((3, 1, 2), dtype=np.int8),
            variant_id=["rs1", ".", "rs3"],
        )
        a12 = np.array([[1, 0], [1, 0], [1, 0]])
        bim = plink.generate_bim(reader, a12)
        df = _parse_bim(bim)
        assert list(df["ID"]) == ["rs1", ".", "rs3"]

    def test_single_allele_site_emits_dot_for_a1(self):
        reader = _build_reader(
            num_variants=2,
            alleles=[("A", "T"), ("G", "")],
            num_samples=1,
            call_genotype=np.zeros((2, 1, 2), dtype=np.int8),
        )
        a12 = np.array([[1, 0], [-1, 0]])
        bim = plink.generate_bim(reader, a12)
        df = _parse_bim(bim)
        # Monomorphic A1 is "." (plink 2 missing-allele convention).
        assert list(df["A1"]) == ["T", "."]
        assert list(df["A2"]) == ["A", "G"]

    def test_multi_contig(self):
        reader = _build_reader(
            num_variants=3,
            variant_contig=[0, 1, 2],
            variant_position=[10, 20, 30],
            num_samples=1,
            call_genotype=np.zeros((3, 1, 2), dtype=np.int8),
            contigs=("chrA", "chrB", "chrC"),
        )
        a12 = np.array([[1, 0], [1, 0], [1, 0]])
        bim = plink.generate_bim(reader, a12)
        df = _parse_bim(bim)
        assert list(df["Chrom"]) == ["chrA", "chrB", "chrC"]

    def test_multi_chunk_offset_arithmetic(self):
        reader = _build_reader(
            num_variants=5,
            variant_position=[10, 20, 30, 40, 50],
            alleles=[("A", "T"), ("A", "G"), ("C", "G"), ("T", "A"), ("G", "C")],
            num_samples=1,
            call_genotype=np.zeros((5, 1, 2), dtype=np.int8),
            variants_chunk_size=2,
        )
        a12 = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [1, 0]])
        bim = plink.generate_bim(reader, a12)
        df = _parse_bim(bim)
        assert list(df["Pos"]) == [10, 20, 30, 40, 50]
        assert list(df["A1"]) == ["T", "G", "G", "A", "C"]
        assert list(df["A2"]) == ["A", "A", "C", "T", "G"]

    def test_no_yielded_chunks_returns_empty_string(self):
        # All variants filtered out via -i ⇒ variant_chunks yields nothing
        # ⇒ generate_bim returns "" (the empty-rows-list branch).
        reader = _build_reader(
            num_variants=2,
            variant_position=[100, 200],
            alleles=[("A", "T"), ("G", "C")],
            num_samples=1,
            call_genotype=np.zeros((2, 1, 2), dtype=np.int8),
            info_fields={"DP": np.array([1, 2], dtype=np.int32)},
        )
        vf = bcftools_filter.BcftoolsFilter(
            field_names=reader.field_names, include="DP>100"
        )
        reader.set_variant_filter(vf)
        a12 = np.zeros((0, 2), dtype=int)
        assert plink.generate_bim(reader, a12) == ""


# ---------------------------------------------------------------------------
# Writer / write_plink end-to-end.
# ---------------------------------------------------------------------------


class TestWriterEndToEnd:
    def test_keystone_known_bytes(self, tmp_path):
        # 3 variants, 3 samples — the same example used by
        # test_encode_plink_example in lib/tests.c.
        genotypes = np.array(
            [
                [[0, 0], [0, 1], [0, 0]],
                [[1, 0], [1, 1], [0, -2]],
                [[1, 1], [0, 1], [-1, -1]],
            ],
            dtype=np.int8,
        )
        reader = _build_reader(
            num_variants=3,
            variant_position=[100, 200, 300],
            alleles=[("A", "T"), ("C", "G"), ("G", "A")],
            num_samples=3,
            sample_id=["s1", "s2", "s3"],
            call_genotype=genotypes,
        )
        out = tmp_path / "out"
        plink.write_plink(reader, out)
        bed = out.with_suffix(".bed").read_bytes()
        assert bed[:3] == b"\x6c\x1b\x01"
        # Per-row encoded byte (computed independently from the spec).
        # See lib/tests.c::test_encode_plink_example.
        assert bed[3:] == bytes([59, 50, 24])

    def test_bed_magic_and_size(self, tmp_path):
        num_variants, num_samples = 5, 7
        genotypes = np.zeros((num_variants, num_samples, 2), dtype=np.int8)
        reader = _build_reader(
            num_variants=num_variants,
            variant_position=list(range(100, 100 + num_variants)),
            alleles=[("A", "T")] * num_variants,
            num_samples=num_samples,
            call_genotype=genotypes,
        )
        out = tmp_path / "p"
        plink.write_plink(reader, out)
        bed = out.with_suffix(".bed").read_bytes()
        assert bed[:3] == b"\x6c\x1b\x01"
        bytes_per_variant = math.ceil(num_samples / 4)
        assert len(bed) == 3 + bytes_per_variant * num_variants

    def test_three_outputs_consistent(self, tmp_path):
        num_variants, num_samples = 4, 5
        reader = _build_reader(
            num_variants=num_variants,
            variant_position=[100, 200, 300, 400],
            alleles=[("A", "T")] * num_variants,
            num_samples=num_samples,
            sample_id=["s0", "s1", "s2", "s3", "s4"],
            call_genotype=np.zeros((num_variants, num_samples, 2), dtype=np.int8),
        )
        out = tmp_path / "p"
        plink.write_plink(reader, out)
        bim = _parse_bim(out.with_suffix(".bim").read_text())
        fam = _parse_fam(out.with_suffix(".fam").read_text())
        bed = out.with_suffix(".bed").read_bytes()
        assert len(bim) == num_variants
        assert len(fam) == num_samples
        bytes_per_variant = math.ceil(num_samples / 4)
        assert (len(bed) - 3) == bytes_per_variant * num_variants

    def test_multi_chunk_variant_axis(self, tmp_path):
        num_variants, num_samples = 7, 4
        rng = np.random.default_rng(42)
        genotypes = rng.integers(0, 2, size=(num_variants, num_samples, 2)).astype(
            np.int8
        )
        reader = _build_reader(
            num_variants=num_variants,
            variant_position=list(range(100, 100 + num_variants)),
            alleles=[("A", "T")] * num_variants,
            num_samples=num_samples,
            call_genotype=genotypes,
            variants_chunk_size=2,
        )
        out = tmp_path / "p"
        plink.write_plink(reader, out)
        bim = _parse_bim(out.with_suffix(".bim").read_text())
        bed = out.with_suffix(".bed").read_bytes()
        assert len(bim) == num_variants
        bytes_per_variant = math.ceil(num_samples / 4)
        assert (len(bed) - 3) == bytes_per_variant * num_variants
        # In-order block writes: BIM positions are sorted ascending.
        assert list(bim["Pos"]) == list(range(100, 100 + num_variants))

    @pytest.mark.parametrize("num_samples", [1, 2, 3, 5, 6, 7, 9])
    def test_pad_bits_are_zero(self, tmp_path, num_samples):
        # Encode all-MISSING genotypes; trailing pad bits in the last byte
        # of each variant must be 0.
        num_variants = 1
        genotypes = -np.ones((num_variants, num_samples, 2), dtype=np.int8)
        reader = _build_reader(
            num_variants=num_variants,
            variant_position=[100],
            alleles=[("A", "T")],
            num_samples=num_samples,
            call_genotype=genotypes,
        )
        out = tmp_path / "p"
        plink.write_plink(reader, out)
        bed = out.with_suffix(".bed").read_bytes()
        bytes_per_variant = math.ceil(num_samples / 4)
        live_in_last = num_samples - 4 * (bytes_per_variant - 1)
        last = bed[3 + bytes_per_variant - 1]
        pad_mask = 0xFF & ~((1 << (live_in_last * 2)) - 1)
        assert (last & pad_mask) == 0

    def test_sample_subset(self, tmp_path):
        num_variants, num_samples = 2, 4
        # Distinct genotype patterns per sample so we can read the subset
        # back from the BED.
        genotypes = np.zeros((num_variants, num_samples, 2), dtype=np.int8)
        genotypes[:, 0] = [0, 0]  # HOM_A2
        genotypes[:, 1] = [0, 1]  # HET
        genotypes[:, 2] = [1, 1]  # HOM_A1
        genotypes[:, 3] = [-1, -1]  # MISSING
        reader = _build_reader(
            num_variants=num_variants,
            variant_position=[100, 200],
            alleles=[("A", "T")] * num_variants,
            num_samples=num_samples,
            sample_id=["s0", "s1", "s2", "s3"],
            call_genotype=genotypes,
        )
        reader.set_samples([3, 0])  # MISSING then HOM_A2
        out = tmp_path / "p"
        plink.write_plink(reader, out)
        fam = _parse_fam(out.with_suffix(".fam").read_text())
        assert list(fam["IID"]) == ["s3", "s0"]
        bed = out.with_suffix(".bed").read_bytes()
        # 2 samples -> 1 byte per variant; bits low->high are sample 0, 1.
        # Sample 0 (was s3, MISSING) = 0b01; sample 1 (was s0, HOM_A2) = 0b11.
        # Byte = (0b11 << 2) | 0b01 = 0b1101 = 0x0D = 13.
        assert bed[3] == 13
        assert bed[4] == 13

    def test_null_samples_excluded(self, tmp_path):
        # Null samples (sample_id == "") must not appear in BED columns
        # or FAM rows. Distinct genotype patterns per sample let us pin
        # down exactly which columns survived: any leaked null column
        # would change the BED byte.
        num_variants, num_samples = 2, 4
        genotypes = np.zeros((num_variants, num_samples, 2), dtype=np.int8)
        genotypes[:, 0] = [0, 0]  # s0   -> HOM_A2 = 0b11
        genotypes[:, 1] = [1, 1]  # null -> HOM_A1 (0b00) if leaked
        genotypes[:, 2] = [0, 1]  # s2   -> HET    = 0b10
        genotypes[:, 3] = [-1, -1]  # null -> MISSING (0b01) if leaked
        reader = _build_reader(
            num_variants=num_variants,
            variant_position=[100, 200],
            alleles=[("A", "T")] * num_variants,
            num_samples=num_samples,
            sample_id=["s0", "", "s2", ""],
            call_genotype=genotypes,
        )
        out = tmp_path / "p"
        plink.write_plink(reader, out)
        fam = _parse_fam(out.with_suffix(".fam").read_text())
        assert list(fam["IID"]) == ["s0", "s2"]
        bed = out.with_suffix(".bed").read_bytes()
        # 2 surviving samples -> 1 byte per variant. Bits low->high:
        # sample 0 (s0, HOM_A2 = 0b11), sample 1 (s2, HET = 0b10).
        # Byte = (0b10 << 2) | 0b11 = 0b1011 = 11.
        assert bed[3] == 11
        assert bed[4] == 11
        bytes_per_variant = math.ceil(2 / 4)
        assert (len(bed) - 3) == bytes_per_variant * num_variants

    def test_variant_filter(self, tmp_path):
        # Build a 5-variant store and drop two variants via a -i filter.
        num_variants, num_samples = 5, 1
        reader = _build_reader(
            num_variants=num_variants,
            variant_position=[100, 200, 300, 400, 500],
            alleles=[("A", "T")] * num_variants,
            num_samples=num_samples,
            call_genotype=np.zeros((num_variants, num_samples, 2), dtype=np.int8),
            info_fields={"DP": np.array([10, 5, 30, 5, 50], dtype=np.int32)},
            variants_chunk_size=2,
        )
        vf = bcftools_filter.BcftoolsFilter(
            field_names=reader.field_names, include="DP>=10"
        )
        reader.set_variant_filter(vf)
        out = tmp_path / "p"
        plink.write_plink(reader, out)
        bim = _parse_bim(out.with_suffix(".bim").read_text())
        bed = out.with_suffix(".bed").read_bytes()
        assert list(bim["Pos"]) == [100, 300, 500]
        bytes_per_variant = math.ceil(num_samples / 4)
        assert (len(bed) - 3) == bytes_per_variant * 3

    def test_filter_excludes_all(self, tmp_path):
        # No variants survive — BED gets only the magic header, BIM/FAM
        # are empty. Exercises the empty-a12 branch in _write_genotypes.
        num_variants, num_samples = 3, 2
        reader = _build_reader(
            num_variants=num_variants,
            variant_position=[100, 200, 300],
            alleles=[("A", "T")] * num_variants,
            num_samples=num_samples,
            sample_id=["s1", "s2"],
            call_genotype=np.zeros((num_variants, num_samples, 2), dtype=np.int8),
            info_fields={"DP": np.array([1, 2, 3], dtype=np.int32)},
        )
        vf = bcftools_filter.BcftoolsFilter(
            field_names=reader.field_names, include="DP>100"
        )
        reader.set_variant_filter(vf)
        out = tmp_path / "p"
        plink.write_plink(reader, out)
        assert out.with_suffix(".bed").read_bytes() == b"\x6c\x1b\x01"
        assert out.with_suffix(".bim").read_text() == ""
        # FAM still lists samples (samples are unaffected by variant filter).
        fam = _parse_fam(out.with_suffix(".fam").read_text())
        assert list(fam["IID"]) == ["s1", "s2"]

    def test_multiallelic_raises(self, tmp_path):
        reader = _build_reader(
            num_variants=1,
            alleles=[("A", "T", "C")],
            num_samples=1,
            call_genotype=np.zeros((1, 1, 2), dtype=np.int8),
        )
        out = tmp_path / "p"
        with pytest.raises(ValueError, match="Multi-allelic"):
            plink.write_plink(reader, out)

    def test_non_diploid_raises(self, tmp_path):
        reader = _build_reader(
            num_variants=1,
            num_samples=1,
            call_genotype=np.zeros((1, 1, 1), dtype=np.int8),
            ploidy=1,
        )
        out = tmp_path / "p"
        with pytest.raises(ValueError, match="diploid"):
            plink.write_plink(reader, out)

    @pytest.mark.parametrize(
        "out_arg",
        [
            "p",
            "p.bed",
            "p.something_else",
            pathlib.Path("p"),
        ],
    )
    def test_path_suffix_handling(self, tmp_path, out_arg):
        reader = _build_reader(
            num_variants=1,
            num_samples=1,
            call_genotype=np.zeros((1, 1, 2), dtype=np.int8),
        )
        out = tmp_path / out_arg
        plink.write_plink(reader, out)
        # All three paths share the stem of `out` (suffix gets replaced).
        stem = pathlib.Path(out).stem
        assert (tmp_path / f"{stem}.bed").exists()
        assert (tmp_path / f"{stem}.bim").exists()
        assert (tmp_path / f"{stem}.fam").exists()
