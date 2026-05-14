"""
Convert VCZ to PLINK 1 binary format (.bed/.bim/.fam) — the on-disk
layout that PLINK 1, 1.9 and 2 all read.

The CLI verb is ``view-plink``. A1 = ALT, A2 = REF (plink 2's
convention); the .bed payload is byte-identical to
``plink2 --vcf X --make-bed`` for biallelic variants.

For the user-facing reference — A1/A2 rationale, downstream-tool
compatibility (plink 1.9, REGENIE, BOLT-LMM, ...), multi-allelic and
monomorphic encoding, sample-subset semantics, chromosome-name
normalisation, and known divergences from plink 2 — see
``docs/plink.md``.
"""

import logging
import pathlib
import time
from typing import ClassVar

import numpy as np
import pandas as pd

from vcztools import _vcztools, format_encoder, retrieval
from vcztools.utils import _as_fixed_length_unicode

logger = logging.getLogger(__name__)

BED_MAGIC = b"\x6c\x1b\x01"


def _check_biallelic(alleles):
    # plink 2 --make-bed errors on multi-allelic .bim rows; mirror that.
    # Use --max-alleles 2 to skip them, or split with bcftools norm -m-
    # before conversion.
    if alleles.shape[1] > 2 and (alleles[:, 2:] != "").any():
        raise ValueError(
            "Multi-allelic variants are not supported in PLINK 1 "
            "binary output (plink 2 --make-bed has the same "
            "restriction). Use --max-alleles 2 to skip them, or "
            "split with bcftools norm -m- before conversion."
        )


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


def generate_bim(reader):
    contig_id = _as_fixed_length_unicode(reader.contig_ids)
    contig_id = np.array(
        [_plink2_normalise_chrom(str(c)) for c in contig_id], dtype=contig_id.dtype
    )
    has_variant_id = "variant_id" in reader.field_names

    fields = ["variant_allele", "variant_contig", "variant_position"]
    if has_variant_id:
        fields.append("variant_id")

    rows = []
    for chunk in reader.variant_chunks(fields=fields):
        n = len(chunk["variant_position"])
        alleles = _as_fixed_length_unicode(chunk["variant_allele"])
        _check_biallelic(alleles)

        # A1 = ALT (column 1), A2 = REF (column 0). For single-allele
        # (monomorphic) rows — alleles[j, 1] == "" — emit "." in the A1
        # slot, matching plink 2's missing-allele convention.
        if alleles.shape[1] >= 2:
            allele_1 = alleles[:, 1].copy()
            allele_1[alleles[:, 1] == ""] = "."
        else:
            allele_1 = np.full(n, ".", dtype=alleles.dtype)
        allele_2 = alleles[:, 0]

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
                    "Allele2": allele_2,
                }
            )
        )

    if len(rows) == 0:
        return ""
    df = pd.concat(rows, ignore_index=True)
    return df.to_csv(header=False, sep="\t", index=False, lineterminator="\n")


def write_plink(reader, out):
    """Write PLINK 1 binary fileset (``.bed`` / ``.bim`` / ``.fam``) for
    ``reader`` under prefix ``out``.

    If a variant filter is configured on the reader, it is resolved
    in place via :meth:`~vcztools.retrieval.VczReader.materialise_variant_filter`
    before encoding so that BIM rows and BED rows are guaranteed to
    align. Sample-scope filters are not supported in PLINK 1 binary
    output: the ``.bed`` format is fixed-width per variant, so
    per-sample filtering doesn't translate.
    """
    reader.materialise_variant_filter()
    out_prefix = pathlib.Path(out)
    bed_path = out_prefix.with_suffix(".bed")
    bim_path = out_prefix.with_suffix(".bim")
    fam_path = out_prefix.with_suffix(".fam")

    with open(fam_path, "w") as f:
        f.write(generate_fam(reader))
    with open(bim_path, "w") as f:
        f.write(generate_bim(reader))
    _stream_bed_to_file(reader, bed_path)


def _stream_bed_to_file(reader, bed_path):
    start = time.perf_counter()
    encode_seconds = 0.0
    write_seconds = 0.0
    bytes_written = 0
    block_size = 1 << 20  # 1 MiB
    with BedEncoder(reader) as encoder, open(bed_path, "wb") as bed_file:
        off = 0
        while off < encoder.bed_size:
            t0 = time.perf_counter()
            buf = encoder.read(off, block_size)
            t1 = time.perf_counter()
            bed_file.write(buf)
            t2 = time.perf_counter()
            encode_seconds += t1 - t0
            write_seconds += t2 - t1
            off += len(buf)
            bytes_written += len(buf)
    elapsed = time.perf_counter() - start
    mib = bytes_written / (1024 * 1024)
    rate = mib / elapsed if elapsed > 0 else 0.0
    logger.info(
        f"write_plink: wrote {mib:.1f} MiB to {bed_path} in "
        f"{elapsed:.2f}s ({rate:.1f} MiB/s); "
        f"encode={encode_seconds:.2f}s, write={write_seconds:.2f}s"
    )


class BedEncoder(format_encoder.FormatEncoder):
    """PLINK 1 ``.bed`` byte-stream encoder over a VCZ store.

    Thin :class:`~vcztools.format_encoder.FormatEncoder` subclass: the
    base class supplies the chunk-resident state machine, POSIX-style
    :meth:`read`, iterator restart/advance arbitration, thread-pool
    lifecycle, and prefix (magic) serving. ``BedEncoder`` plugs in
    PLINK 1's 3-byte magic prefix and the C-kernel encode of one
    variant chunk.

    Scope is the ``.bed`` stream only. For the companion ``.bim`` and
    ``.fam`` files, use :func:`generate_bim` and :func:`generate_fam`
    directly.

    Honours :meth:`~vcztools.retrieval.VczReader.set_samples` and
    :meth:`~vcztools.retrieval.VczReader.set_variants` configured on
    the reader before construction. With ``set_variants``, the encoded
    ``.bed`` covers exactly the selected variants in chunk-plan order;
    :attr:`bed_size` and :attr:`num_variants` reflect the selection.

    Biallelic checking is performed lazily as chunks are decoded;
    multi-allelic variants raise ``ValueError`` during :meth:`read`,
    not at construction.

    :meth:`~vcztools.retrieval.VczReader.set_variant_filter` is not
    yet supported and raises ``NotImplementedError`` at construction;
    apply predicate filters externally before passing the reader in.

    Per-chunk PLINK encoding parallelises across variant-axis
    sub-blocks via a :class:`concurrent.futures.ThreadPoolExecutor`
    owned by the encoder. ``encode_threads`` (default 4) sets the
    pool size; ``encode_block_bytes`` (default 1 MiB) is the
    input-bytes target per sub-block. Chunks at or below the
    threshold encode synchronously on the calling thread.

    The 1 MiB block default targets typical L2 cache size; PLINK
    encoding is a tight memory-walk loop and benefits from each
    thread's working set fitting in L2. Bump for very wide cohorts
    if profiling shows scheduling overhead dominates encode time.
    """

    BED_MAGIC: ClassVar[bytes] = BED_MAGIC
    _default_encode_block_bytes: ClassVar[int] = 1 * 1024 * 1024

    def __init__(
        self,
        reader: retrieval.VczReader,
        *,
        encode_threads: int | None = None,
        encode_block_bytes: int | None = None,
    ):
        num_samples = int(reader.sample_ids.size)
        bytes_per_variant = (num_samples + 3) // 4
        super().__init__(
            reader,
            bytes_per_variant=bytes_per_variant,
            prefix_bytes=BED_MAGIC,
            iterator_fields=["call_genotype", "variant_allele"],
            encode_threads=encode_threads,
            encode_block_bytes=encode_block_bytes,
        )

    @property
    def bed_size(self) -> int:
        """Total ``.bed`` size in bytes (alias of
        :attr:`~vcztools.format_encoder.FormatEncoder.total_size`)."""
        return self.total_size

    def _encode_chunk(self, chunk: dict) -> bytes:
        _check_biallelic(chunk["variant_allele"])
        # Coerce once at method entry: a sample-subset call_genotype
        # is fancy-indexed and non-contiguous, so the copy must happen
        # before slicing. Once G is C-contiguous, axis-0 sub-blocks
        # are zero-copy views; the C kernel writes one row of
        # bytes_per_variant per variant.
        G = np.ascontiguousarray(chunk["call_genotype"], dtype=np.int8)

        def encode_range(start: int, end: int) -> bytes:
            return bytes(_vcztools.encode_plink(G[start:end]).data)

        return self._parallel_encode(
            num_variants=G.shape[0],
            encode_range=encode_range,
        )
