"""
Convert VCZ to plink 1 binary format.
"""

import numpy as np
import pandas as pd
import zarr


def encode_genotypes(g, allele_1, allele_2):
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


def genotype_to_plink_bed(genotypes, out_buffer):
    """
    Convert genotypes from Zarr format to PLINK BED format

    In Zarr:
      - Genotypes are stored as int8 with values 0 (REF), 1+ (ALT), -1 (missing)

    In PLINK BED format (2 bits per genotype):
      - 00 (0b00): Homozygous for first allele (0/0)
      - 01 (0b01): Missing genotype
      - 10 (0b10): Heterozygous (0/1 or 1/0)
      - 11 (0b11): Homozygous for second allele (1/1)

    Each byte in BED stores 4 genotypes (2 bits each), with sample ordering:
    - Lowest sample index in least significant bits
    - Highest sample index in most significant bits
    """
    n_variants, n_samples, ploidy = genotypes.shape
    bytes_per_variant = (n_samples + 3) // 4

    for var_idx in range(n_variants):
        for sam_idx in range(n_samples):
            # Determine which byte and bit position this genotype belongs to
            byte_idx = var_idx * bytes_per_variant + (sam_idx // 4)
            bit_pos = (sam_idx % 4) * 2

            a1 = genotypes[var_idx, sam_idx, 0]
            a2 = genotypes[var_idx, sam_idx, 1] if ploidy > 1 else a1

            if a1 == -1 or a2 == -1:
                # Missing genotype: 01 in PLINK format
                code = 0b01
            elif a1 == 0 and a2 == 0:
                # Homozygous REF: 00 in PLINK format
                code = 0b00
            elif a1 == 1 and a2 == 1:
                # Homozygous ALT: 11 in PLINK format
                code = 0b11
            else:
                # Heterozygous: 10 in PLINK format
                code = 0b10

            mask = ~(0b11 << bit_pos)
            out_buffer[byte_idx] = (out_buffer[byte_idx] & mask) | (code << bit_pos)


def generate_fam(root):
    # TODO generate an error if sample_id contains a space
    sample_id = root["sample_id"][:].astype(str)
    zeros = np.zeros(sample_id.shape, dtype=int)
    df = pd.DataFrame(
        {
            "FamilyID": sample_id,
            "IndividualID": sample_id,
            "FatherID": zeros,
            "MotherId": zeros,
            "Sex": zeros,
            "Phenotype": np.full_like(zeros, -9),
        }
    )
    return df.to_csv(sep="\t", header=False, index=False)


def generate_bim(root, a12_allele):
    select = a12_allele[:, 1] != -1
    contig_id = root["contig_id"][:].astype(str)
    alleles = root["variant_allele"][:].astype(str)[select]
    a12_allele = a12_allele[select]
    num_variants = np.sum(select)
    allele_1 = alleles[np.arange(num_variants), a12_allele[:, 0]]
    single_allele_sites = np.where(a12_allele[:, 0] == -1)
    allele_1[single_allele_sites] = "0"
    df = pd.DataFrame(
        {
            "Chrom": contig_id[root["variant_contig"][:][select]],
            "VariantId": root["variant_id"][:][select],
            "GeneticPosition": np.zeros(np.sum(select), dtype=int),
            "Position": root["variant_position"][:][select],
            "Allele1": allele_1,
            "Allele2": alleles[np.arange(num_variants), a12_allele[:, 1]],
        }
    )
    return df.to_csv(header=False, sep="\t", index=False)


class Writer:
    def __init__(self, vcz_path, bed_path, fam_path, bim_path):
        self.root = zarr.open(vcz_path, mode="r")
        self.bim_path = bim_path
        self.fam_path = fam_path
        self.bed_path = bed_path
        with open(fam_path, "w") as f:
            f.write(generate_fam(self.root))

    def run(self):
        max_alleles = self.root["variant_allele"].shape[1]
        G = self.root["call_genotype"]
        num_samples = G.shape[1]
        num_variants = G.shape[0]
        a12_allele = np.zeros((num_variants, 2), dtype=int) - 1

        with open(self.bed_path, "wb") as bed_file:
            bed_file.write(bytes([0x6C, 0x1B, 0x01]))
            for j, g in enumerate(self.root["call_genotype"]):
                g = g.reshape(num_samples * 2)
                assert np.all(g >= -2)
                count = np.bincount(g + 2, minlength=max_alleles + 2)
                # [dimension pad, missing data, reference, allele 1, ...]
                count = count[2:]
                argsort = np.argsort(count)
                a12_allele[j, 1] = 0
                if argsort[-1] == 0:
                    # print("Ref allele most frequent")
                    # Ref allele is most frequent - chose lowest allele from next most
                    # frequent class
                    f = count[argsort[-2]]
                else:
                    # print("Ref allele not most frequent")
                    f = count[argsort[-1]]
                a = 1
                while count[a] != f:
                    a += 1
                a12_allele[j, 0] = a
                assert a12_allele[j, 0] != a12_allele[j, 1]
                if count[a12_allele[j, 0]] == 0:
                    a12_allele[j, 0] = -1
                # print(
                #     self.root["variant_contig"][j],
                #     self.root["variant_position"][j],
                #     [j],
                #     alleles[j],
                #     count,
                #     argsort,
                #     a12_allele[j],
                # )
                encoded = encode_genotypes(g.reshape(num_samples, 2), *a12_allele[j])
                bed_file.write(encoded)

        with open(self.bim_path, "w") as f:
            f.write(generate_bim(self.root, a12_allele))
