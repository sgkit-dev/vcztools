import numpy as np
import pytest

from vcztools import plink


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
        # print("\t", a, b, code)
        mask = ~(0b11 << bit_pos)
        buff[byte_idx] = (buff[byte_idx] & mask) | (code << bit_pos)
    return buff


def encode_genotypes(G, a12_allele=None):
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


class TestEncodeGenotypes:
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
                [[0, 0], [0, 1], [0, 0], [0, 1], [1, 1]],
                [[1, 0], [-3, 1], [0, 0], [0, 1], [-1, -2]],
                [[0, 1], [0, 1], [1, 2], [0, 1], [1, 1]],
                [[0, 0], [0, -2], [0, 3], [-2, 1], [-1, -2]],
            ],
        ],
    )
    def test_examples_01_alleles(self, genotypes):
        b1 = encode_genotypes(genotypes)
        b2 = plink.encode_genotypes(genotypes)
        assert b1 == b2

    @pytest.mark.parametrize(
        ("num_variants", "num_samples"),
        [
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (1, 10),
            (1, 4),
            (1, 16),
            (1, 100),
            (1, 101),
            (10, 1),
            (100, 1),
            (10, 2),
            (10, 3),
            (10, 4),
            (10, 5),
            (10, 6),
            (10, 7),
            (10, 8),
            (10, 9),
        ],
    )
    @pytest.mark.parametrize("value", [-1, 0, 1, 2])
    def test_shapes_01_alleles(self, value, num_variants, num_samples):
        g = np.zeros((num_variants, num_samples, 2), dtype=np.int8) + value
        b1 = encode_genotypes(g)
        b2 = plink.encode_genotypes(g)
        # assert len(b1) == len(b2)
        assert b1 == b2

    @pytest.mark.parametrize(
        ("num_variants", "num_samples"),
        [
            (1, 4),
            (1, 8),
            (1, 16),
            (1, 32),
            (1, 100),
            (33, 4),
            (33, 8),
            (33, 16),
            (33, 32),
            (33, 100),
        ],
    )
    def test_all_zeros_div_4(self, num_variants, num_samples):
        assert num_samples % 4 == 0
        g = np.zeros((num_variants, num_samples, 2), dtype=np.int8)
        b1 = encode_genotypes(g)
        b2 = plink.encode_genotypes(g)
        assert b1 == b2
        assert b1 == bytearray(0xFF for _ in range(num_variants * num_samples // 4))

    @pytest.mark.parametrize(
        ("num_variants", "num_samples"),
        [
            (1, 33),
            (10, 1000),
        ],
    )
    def test_nonsensical_data(self, num_variants, num_samples):
        g = np.arange((num_variants * num_samples * 2), dtype=np.int8).reshape(
            (num_variants, num_samples, 2)
        )
        a12 = np.arange(num_variants * 2, dtype=np.int8).reshape((num_variants, 2))
        b1 = encode_genotypes(g, a12)
        b2 = plink.encode_genotypes(g, a12)
        assert b1 == b2
