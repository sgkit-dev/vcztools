"""
Convert VCZ to PLINK 1 binary format (.bed/.bim/.fam) — the on-disk
layout that PLINK 1, 1.9 and 2 all read.

The CLI verb is ``view-bed``. The semantic choice of which allele
populates the A1 column follows **PLINK 2** (A1 = ALT, A2 = REF):
the modern, REF/ALT-stable convention. The .bed payload is
byte-identical to ``plink2 --vcf X --make-bed`` for biallelic
variants.

NOTE — this differs from PLINK 1.9's default behaviour. PLINK 1.9
reorders A1/A2 in memory on load to put the minor allele in A1,
unless invoked with ``--keep-allele-order`` (or
``--real-ref-alleles``). Downstream pipelines that read vcztools'
output with default PLINK 1.9 will see a different A1/A2 labelling
than the .bim claims, and any frequency-derived statistic whose sign
depends on the labelling will flip relative to a PLINK 2 run on the
same file. Pass ``--keep-allele-order`` to preserve our labelling.

Multi-allelic variants are rejected, mirroring ``plink2 --make-bed``,
which errors with "Error: <out>.bim cannot contain multiallelic
variants." Callers wanting to skip multi-allelic sites use the
``--max-alleles 2`` flag on ``view-bed`` (matching plink 2).

Single-allele (monomorphic) variants emit ``A1 = "."`` in the .bim
(plink 2's missing-allele encoding), with all genotype bits set to
the MISSING code in the .bed.

For consequences for the wider downstream tool ecosystem (REGENIE,
BOLT-LMM, GCTA, KING, flashpca, ADMIXTURE), see the documentation;
the short version is "insensitive in practice", with PLINK 1.9 the
only consumer that visibly relabels allele columns on read.
"""

import pathlib

import numpy as np
import pandas as pd

from vcztools import _vcztools
from vcztools.utils import _as_fixed_length_unicode


class MaxAllelesFilter:
    """Variant-scope :class:`~vcztools.variant_filter.VariantFilter` that
    keeps variants whose number of non-empty alleles is at most
    ``max_alleles``.

    For PLINK 1 binary output this is invoked with ``max_alleles=2``,
    matching ``plink2 --vcf X --make-bed --max-alleles 2``.
    """

    scope = "variant"
    referenced_fields = frozenset({"variant_allele"})

    def __init__(self, max_alleles):
        if max_alleles < 1:
            raise ValueError(f"max_alleles must be >= 1, got {max_alleles}")
        self.max_alleles = max_alleles

    def evaluate(self, chunk_data):
        alleles = chunk_data["variant_allele"]
        # variant_allele has shape (num_variants, max_alleles_in_store).
        # A variant is within max_alleles when every entry past column
        # `max_alleles - 1` is the empty string (the missing-allele
        # sentinel that bio2zarr writes for unused slots).
        if alleles.shape[1] <= self.max_alleles:
            return np.ones(alleles.shape[0], dtype=bool)
        tail = alleles[:, self.max_alleles :]
        return (tail == "").all(axis=1)


class _AndVariantFilter:
    """Combine multiple variant-scope :class:`VariantFilter` objects with
    a logical AND on their per-variant masks.

    Used by the CLI to compose a user-supplied ``-i``/``-e`` filter
    with the synthetic ``--max-alleles`` filter for ``view-bed``.
    Sample-scope filters are not supported; the constructor raises if
    any input filter has ``scope != "variant"``.
    """

    scope = "variant"

    def __init__(self, filters):
        self._filters = list(filters)
        for f in self._filters:
            if f.scope != "variant":
                raise ValueError(
                    "_AndVariantFilter can only combine variant-scope filters; "
                    "got a sample-scope filter."
                )
        self.referenced_fields = frozenset().union(
            *(f.referenced_fields for f in self._filters)
        )

    def evaluate(self, chunk_data):
        result = self._filters[0].evaluate(chunk_data)
        for f in self._filters[1:]:
            result = result & f.evaluate(chunk_data)
        return result


def encode_genotypes(genotypes, a12_allele=None):
    # The C extension requires C-contiguous int8 arrays. A reader-yielded
    # call_genotype that's been reordered by a sample subset is fancy-
    # indexed and not contiguous, so force a copy when needed.
    G = np.ascontiguousarray(genotypes, dtype=np.int8)
    if a12_allele is None:
        a12_allele = np.zeros((G.shape[0], 2), dtype=G.dtype)
        a12_allele[:, 0] = 1
    a12_allele = np.ascontiguousarray(a12_allele, dtype=G.dtype)
    return bytes(_vcztools.encode_plink(G, a12_allele).data)


def generate_fam(reader):
    # sample_ids excludes null samples and reflects any caller-applied
    # subset, matching the column axis of the BED that _write_genotypes
    # emits. raw_sample_ids would include null entries and desync FAM
    # rows from BED columns.
    sample_id = _as_fixed_length_unicode(reader.sample_ids)
    for s in sample_id:
        if any(c.isspace() for c in str(s)):
            raise ValueError(
                f"Sample ID {s!r} contains whitespace; "
                "PLINK FAM is whitespace-separated."
            )
    # FamilyID = "0" matches the default of `plink2 --vcf X --make-bed`,
    # which writes 0 unless the user passes --double-id or --id-delim.
    zeros = np.zeros(sample_id.shape, dtype=int)
    family_id = np.full(sample_id.shape, "0", dtype="U1")
    df = pd.DataFrame(
        {
            "FamilyID": family_id,
            "IndividualID": sample_id,
            "FatherID": zeros,
            "MotherId": zeros,
            "Sex": zeros,
            "Phenotype": np.full_like(zeros, -9),
        }
    )
    return df.to_csv(sep="\t", header=False, index=False, lineterminator="\n")


_CHR_PREFIX = "chr"
_PLINK2_STANDARD_CHROMS = frozenset(str(i) for i in range(1, 23)) | {"X", "Y", "MT"}


def _plink2_normalise_chrom(chrom):
    """Normalise a contig name to match plink 2's ``--make-bed`` output.

    plink 2 strips the ``chr`` prefix from standard human chromosomes
    (1–22, X, Y, MT) and rewrites ``chrM`` → ``MT``. Non-standard
    contigs (e.g. ``chrUnknown``) are passed through unchanged (under
    plink 2 they require ``--allow-extra-chr``; vcztools does not
    enforce that flag).
    """
    if not chrom.startswith(_CHR_PREFIX):
        return chrom
    suffix = chrom[len(_CHR_PREFIX) :]
    if suffix == "M":
        return "MT"
    if suffix in _PLINK2_STANDARD_CHROMS:
        return suffix
    return chrom


def generate_bim(reader, a12_allele):
    contig_id = _as_fixed_length_unicode(reader.contig_ids)
    contig_id = np.array(
        [_plink2_normalise_chrom(str(c)) for c in contig_id], dtype=contig_id.dtype
    )
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

        alleles = _as_fixed_length_unicode(chunk["variant_allele"])

        allele_1 = alleles[np.arange(n), chunk_a12[:, 0]]
        # A1 == -1 marks a single-allele (monomorphic) site. plink 2 uses
        # "." as the missing-allele indicator in .bim; match that.
        allele_1[chunk_a12[:, 0] == -1] = "."

        if has_variant_id:
            variant_id = _as_fixed_length_unicode(chunk["variant_id"])
        else:
            variant_id = np.array(["."] * n, dtype="U1")

        rows.append(
            pd.DataFrame(
                {
                    "Chrom": contig_id[chunk["variant_contig"]],
                    "VariantId": variant_id,
                    "GeneticPosition": np.zeros(n, dtype=int),
                    "Position": chunk["variant_position"],
                    "Allele1": allele_1,
                    "Allele2": alleles[np.arange(n), chunk_a12[:, 1]],
                }
            )
        )

    if len(rows) == 0:
        return ""
    df = pd.concat(rows, ignore_index=True)
    return df.to_csv(header=False, sep="\t", index=False, lineterminator="\n")


class Writer:
    def __init__(self, reader, bed_path, fam_path, bim_path):
        self.reader = reader

        self.bim_path = bim_path
        self.fam_path = fam_path
        self.bed_path = bed_path

    def _compute_alleles(self, alleles):
        """
        Return per-variant a12 indexes in the plink 2 REF/ALT convention.

        ``alleles`` has shape ``(n, max_alleles_in_store)``. The
        per-variant allele count is the number of non-empty entries in
        each row; the store-wide max is irrelevant once a downstream
        filter (e.g. ``--max-alleles``) has dropped multi-allelic rows.

        For each surviving variant:

        * ``a12[:, 0] = 1`` (A1 = ALT, i.e. ``variant_allele[:, 1]``)
        * ``a12[:, 1] = 0`` (A2 = REF, i.e. ``variant_allele[:, 0]``)

        For single-allele (monomorphic) variants — where ``alleles[j, 1]``
        is the empty string — ``a12[j, 0]`` is set to ``-1``. The .bim
        writer emits ``"."`` in that slot, matching plink 2's
        missing-allele convention.

        Multi-allelic variants raise ValueError, mirroring
        ``plink2 --make-bed`` (which errors out on multi-allelic .bim
        rows). Use ``--max-alleles 2`` to skip them.
        """
        if alleles.shape[1] > 2 and (alleles[:, 2:] != "").any():
            raise ValueError(
                "Multi-allelic variants are not supported in PLINK 1 "
                "binary output (plink 2 --make-bed has the same "
                "restriction). Use --max-alleles 2 to skip them, or "
                "split with bcftools norm -m- before conversion."
            )
        num_variants = alleles.shape[0]
        a12 = np.zeros((num_variants, 2), dtype=np.int8)
        a12[:, 0] = 1
        # alleles may have >2 columns when the store's max-allele
        # is higher than 2; the second column still holds ALT for any
        # surviving (≤2-allele) variant.
        a12[alleles[:, 1] == "", 0] = -1
        return a12

    def _write_genotypes(self):
        ci = self.reader.variant_chunks(fields=["call_genotype", "variant_allele"])
        # a12 is small (8*num_variants bytes per column) and only
        # materialised for variants surviving the reader's filter, so
        # collecting per-chunk arrays and concatenating is cheap and
        # robust to partial-chunk yields under variant filtering.
        a12_per_chunk = []
        with open(self.bed_path, "wb") as bed_file:
            bed_file.write(bytes([0x6C, 0x1B, 0x01]))

            for chunk in ci:
                G = chunk["call_genotype"]
                if G.ndim != 3 or G.shape[2] != 2:
                    raise ValueError(
                        "Only diploid genotypes are supported "
                        f"(call_genotype has shape {G.shape})"
                    )
                a12 = self._compute_alleles(chunk["variant_allele"])
                buff = encode_genotypes(G, a12)
                bed_file.write(buff)
                a12_per_chunk.append(a12)
        if len(a12_per_chunk) == 0:
            return np.zeros((0, 2), dtype=np.int8)
        return np.concatenate(a12_per_chunk, axis=0)

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
