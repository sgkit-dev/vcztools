"""
Convert VCZ to plink 1 binary format.
"""

import pathlib

import numpy as np
import pandas as pd
import zarr

from vcztools import _vcztools
from vcztools.utils import _as_fixed_length_unicode


def encode_genotypes(genotypes, a12_allele=None):
    G = np.asarray(genotypes, dtype=np.int8)
    if a12_allele is None:
        a12_allele = np.zeros((G.shape[0], 2), dtype=G.dtype)
        a12_allele[:, 0] = 1
    a12_allele = np.asarray(a12_allele, dtype=G.dtype)
    # TODO: not sure if this is taking a copy. See the point about
    # allocating a numpy array in the C code.
    return bytes(_vcztools.encode_plink(G, a12_allele).data)


def generate_fam(reader):
    # TODO generate an error if sample_id contains a space
    sample_id = _as_fixed_length_unicode(reader.all_sample_ids)
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


def generate_bim(reader, a12_allele):
    contig_id = _as_fixed_length_unicode(reader.contig_ids)
    has_variant_id = "variant_id" in reader.field_names

    fields = ["variant_allele", "variant_contig", "variant_position"]
    if has_variant_id:
        fields.append("variant_id")

    rows = []
    offset = 0
    for chunk in reader.variant_chunks(fields=fields):
        n = len(chunk["variant_position"])
        chunk_a12 = a12_allele[offset : offset + n]
        offset += n

        select = chunk_a12[:, 1] != -1
        if not np.any(select):
            continue

        alleles = _as_fixed_length_unicode(chunk["variant_allele"])[select]
        chunk_a12 = chunk_a12[select]
        nsel = int(np.sum(select))

        allele_1 = alleles[np.arange(nsel), chunk_a12[:, 0]]
        single_allele_sites = np.where(chunk_a12[:, 0] == -1)
        allele_1[single_allele_sites] = "0"

        if has_variant_id:
            variant_id = chunk["variant_id"][select]
        else:
            variant_id = np.array(["."] * nsel, dtype="S")

        rows.append(
            pd.DataFrame(
                {
                    "Chrom": contig_id[chunk["variant_contig"][select]],
                    "VariantId": variant_id,
                    "GeneticPosition": np.zeros(nsel, dtype=int),
                    "Position": chunk["variant_position"][select],
                    "Allele1": allele_1,
                    "Allele2": alleles[np.arange(nsel), chunk_a12[:, 1]],
                }
            )
        )

    if len(rows) == 0:
        return ""
    df = pd.concat(rows, ignore_index=True)
    return df.to_csv(header=False, sep="\t", index=False)


class Writer:
    def __init__(self, reader, bed_path, fam_path, bim_path):
        self.reader = reader

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
        return a12_allele

    def _write_genotypes(self):
        ci = self.reader.variant_chunks(fields=["call_genotype", "variant_allele"])
        # Scratch zarr array sized to the variants axis; blocks are
        # written in-order as we consume variant_chunks. This is an
        # in-process scratch store, not a read from the input — the
        # one remaining direct-zarr touch in this module.
        a12_allele = zarr.zeros(
            (self.reader.num_variants, 2),
            chunks=self.reader.variants_chunk_size,
            dtype=int,
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
            f.write(generate_bim(self.reader, a12_allele))

        with open(self.fam_path, "w") as f:
            f.write(generate_fam(self.reader))


def write_plink(reader, out):
    out_prefix = pathlib.Path(out)
    writer = Writer(
        reader,
        bed_path=out_prefix.with_suffix(".bed"),
        fam_path=out_prefix.with_suffix(".fam"),
        bim_path=out_prefix.with_suffix(".bim"),
    )
    writer.run()
