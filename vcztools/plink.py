"""
Convert VCZ to plink 1 binary format.
"""

import numpy as np
import pandas as pd
import zarr

from . import _vcztools, retrieval


def encode_genotypes(genotypes, a12_allele=None):
    G = np.asarray(genotypes, dtype=np.int8)
    if a12_allele is None:
        a12_allele = np.zeros((G.shape[0], 2), dtype=G.dtype)
        a12_allele[:, 0] = 1
    a12_allele = np.asarray(a12_allele, dtype=G.dtype)
    # TODO: not sure if this is taking a copy. See the point about
    # allocating a numpy array in the C code.
    return bytes(_vcztools.encode_plink(G, a12_allele).data)


def new_encode_genotypes(genotypes):
    G = np.asarray([genotypes], dtype=np.int8)
    a12_allele = np.asarray([[1, 0]], dtype=G.dtype)
    return bytes(_vcztools.encode_plink(G, a12_allele).data)


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

    num_variants = np.sum(select)
    if "variant_id" in root:
        variant_id = root["variant_id"][:][select]
    else:
        variant_id = np.array(["."] * num_variants, dtype="S")

    df = pd.DataFrame(
        {
            "Chrom": contig_id[root["variant_contig"][:][select]],
            "VariantId": variant_id,
            "GeneticPosition": np.zeros(np.sum(select), dtype=int),
            "Position": root["variant_position"][:][select],
            "Allele1": allele_1,
            "Allele2": alleles[np.arange(num_variants), a12_allele[:, 1]],
        }
    )
    return df.to_csv(header=False, sep="\t", index=False)


def translate(genotypes, alleles):
    copy = np.full_like(genotypes, -1)
    for new, old in enumerate(alleles):
        copy[genotypes == old] = new
    return copy


def reorder_alleles(genotypes):
    """
    Return a tuple (minor, major, reordered) for the specified numpy array of diploid
    genotypes for a given variant. The reordered genotypes will be coded such that
    0 is the major allele and 1 is the minor allele. The returned values of minor
    and major are the respective alleles in the *input* genotypes.
    """
    num_samples = genotypes.shape[0]
    assert genotypes.shape[1] == 2
    g = genotypes.reshape(num_samples * 2)
    assert np.all(g >= -2)
    max_alleles = np.max(g)
    count = np.bincount(g + 2, minlength=max_alleles + 2)
    # [dimension pad, missing data, alleles[0], alleles[1], ...]
    count = count[2:]
    if max_alleles == 1 and count[0] > count[1]:
        # Common case - exit early with nothing to do
        return 1, 0, genotypes
    # General case
    argsort = np.argsort(count)

    # a12_allele[j, 1] = 0
    major = 0
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
    # a12_allele[j, 0] = a
    minor = a

    # assert a12_allele[j, 0] != a12_allele[j, 1]
    # if alleles[j][1] == "":
    #     a12_allele[j, 0] = -1

    # if argsort[-1] == 0:
    #     # print("Ref allele most frequent")
    #     # Ref allele is most frequent - chose lowest allele from next most
    #     # frequent class
    #     f = count[argsort[-2]]
    # else:
    #     # print("Ref allele not most frequent")
    #     f = count[argsort[-1]]
    # a = 1
    # while count[a] != f:
    #     a += 1
    # minor = a

    # a12_allele[j, 0] = a
    # assert a12_allele[j, 0] != a12_allele[j, 1]
    # if alleles[j][1] == "":
    #     a12_allele[j, 0] = -1

    # print("count = ", count, argsort)
    # major = argsort[-1]
    # minor = -1
    # if len(argsort) > 1:
    #     minor = argsort[-2]
    #     while minor >= 0 and count[minor] == count[argsort[-2]]:
    #         minor -= 1
    return minor, major, translate(genotypes, [major, minor])


class Writer:
    def __init__(
        self, vcz_path, bed_path, fam_path, bim_path, include=None, exclude=None
    ):
        self.root = zarr.open(vcz_path, mode="r")

        self.bim_path = bim_path
        self.fam_path = fam_path
        self.bed_path = bed_path

    def _compute_alleles(self, G, alleles):
        """
        Returns the a12 alleles for the specified chunk of data.
        """
        max_alleles = alleles.shape[1]
        # if max_alleles != 2:
        #     raise ValueError(
        #         "Only biallelic VCFs supported currently: "
        #         "please comment on https://github.com/sgkit-dev/vcztools/issues/224 "
        #         "if this limitation affects you"
        #     )
        num_variants = G.shape[0]
        num_samples = G.shape[1]
        a12_allele = np.zeros((num_variants, 2), dtype=int) - 1
        for j in range(num_variants):
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
            if alleles[j][1] == "":
                a12_allele[j, 0] = -1
            # print(
            #     self.root["variant_contig"][j],
            #     self.root["variant_position"][j],
            #     [j],
            #     self.root["variant_allele"][j],
            #     count,
            #     argsort,
            #     a12_allele[j],
            # )
        return a12_allele

    def run(self):
        ci = retrieval.variant_chunk_iter(
            self.root,
            fields=[
                "call_genotype",
                "variant_allele",
                "variant_contig",
                "variant_position",
                "variant_id",
            ],
        )
        contig_id = self.root["contig_id"][:].astype(str)
        bim_rows = []
        with open(self.bed_path, "wb") as bed_file:
            bed_file.write(bytes([0x6C, 0x1B, 0x01]))

            for chunk in ci:
                iterator = zip(
                    chunk["variant_contig"],
                    chunk["variant_id"],
                    chunk["variant_position"],
                    chunk["variant_allele"],
                    chunk["call_genotype"],
                )
                for contig, variant_id, position, alleles, genotypes in iterator:
                    print("VAR", position, alleles, genotypes)
                    minor, major, genotypes = reorder_alleles(genotypes)
                    print("mapped:", minor, major, genotypes)
                    allele_1 = "0"
                    if minor != -1:
                        allele_1 = alleles[minor]
                    allele_2 = alleles[major]
                    buff = new_encode_genotypes(genotypes)
                    bed_file.write(buff)
                    bim_rows.append(
                        {
                            "Contig": contig_id[contig],
                            "VariantId": variant_id,
                            "GeneticPosition": 0,
                            "Position": position,
                            "Allele1": allele_1,
                            "Allele2": allele_2,
                        }
                    )

        bim_df = pd.DataFrame(bim_rows)
        with open(self.bim_path, "w") as f:
            f.write(bim_df.to_csv(header=False, sep="\t", index=False))

        with open(self.fam_path, "w") as f:
            f.write(generate_fam(self.root))


def write_plink(vcz_path, out_prefix, include=None, exclude=None):
    out_prefix = str(out_prefix)
    writer = Writer(
        vcz_path,
        bed_path=out_prefix + ".bed",
        fam_path=out_prefix + ".fam",
        bim_path=out_prefix + ".bim",
        include=include,
        exclude=exclude,
    )
    writer.run()
