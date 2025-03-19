"""
Convert VCZ to plink 1 binary format.
"""

import numpy as np
import pandas as pd
import zarr


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
        with open(self.bim_path, "w") as f:
            f.write(generate_bim(self.root, a12_allele))
