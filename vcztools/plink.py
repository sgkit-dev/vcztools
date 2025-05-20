"""
Convert VCZ to plink 1 binary format.
"""

import pathlib

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
        if max_alleles != 2:
            raise ValueError(
                "Only biallelic VCFs supported currently: "
                "please comment on https://github.com/sgkit-dev/vcztools/issues/224 "
                "if this limitation affects you"
            )
        num_variants = G.shape[0]
        num_samples = G.shape[1]
        a12_allele = np.zeros((num_variants, 2), dtype=int) - 1
        for j, g in enumerate(G):
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

    def _write_genotypes(self):
        ci = retrieval.variant_chunk_iter(
            self.root, fields=["call_genotype", "variant_allele"]
        )
        call_genotype = self.root["call_genotype"]
        a12_allele = zarr.zeros(
            (call_genotype.shape[0], 2), chunks=call_genotype.chunks[0], dtype=int
        )
        with open(self.bed_path, "wb") as bed_file:
            bed_file.write(bytes([0x6C, 0x1B, 0x01]))

            for j, chunk in enumerate(ci):
                G = chunk["call_genotype"]
                a12 = self._compute_alleles(G, chunk["variant_allele"])
                buff = encode_genotypes(G, a12)
                bed_file.write(buff)
                a12_allele.blocks[j] = a12
        return a12_allele[:]

    def run(self):
        a12_allele = self._write_genotypes()

        with open(self.bim_path, "w") as f:
            f.write(generate_bim(self.root, a12_allele))

        with open(self.fam_path, "w") as f:
            f.write(generate_fam(self.root))


def write_plink(vcz_path, out, include=None, exclude=None):
    out_prefix = pathlib.Path(out)
    # out_prefix.mkdir(exist_ok=True)
    writer = Writer(
        vcz_path,
        bed_path=out_prefix.with_suffix(".bed"),
        fam_path=out_prefix.with_suffix(".fam"),
        bim_path=out_prefix.with_suffix(".bim"),
        include=include,
        exclude=exclude,
    )
    writer.run()
