"""
Convert VCZ to PLINK 1 binary format (.bed/.bim/.fam) — the on-disk
layout that PLINK 1, 1.9 and 2 all read.

The CLI verb is ``view-bed``. A1 = ALT, A2 = REF (plink 2's
convention); the .bed payload is byte-identical to
``plink2 --vcf X --make-bed`` for biallelic variants.

For the user-facing reference — A1/A2 rationale, downstream-tool
compatibility (plink 1.9, REGENIE, BOLT-LMM, ...), multi-allelic and
monomorphic encoding, sample-subset semantics, chromosome-name
normalisation, and known divergences from plink 2 — see
``docs/plink.md``.
"""

import concurrent.futures as cf
import logging
import pathlib
import time
from typing import ClassVar

import numpy as np
import pandas as pd

from vcztools import _vcztools, retrieval
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
    bytes_written = 0
    block_size = 1 << 20  # 1 MiB
    with BedEncoder(reader) as encoder, open(bed_path, "wb") as bed_file:
        off = 0
        while off < encoder.bed_size:
            buf = encoder.read(off, block_size)
            bed_file.write(buf)
            off += len(buf)
            bytes_written += len(buf)
    elapsed = time.perf_counter() - start
    mib = bytes_written / (1024 * 1024)
    rate = mib / elapsed if elapsed > 0 else 0.0
    logger.info(
        f"write_plink: wrote {mib:.1f} MiB to {bed_path} in "
        f"{elapsed:.2f}s ({rate:.1f} MiB/s)"
    )


class BedEncoder:
    """PLINK 1 ``.bed`` byte-stream encoder over a VCZ store.

    Exposes the virtual ``.bed`` as a byte stream addressable by
    ``(off, size)``. Optimised for sequential consumers (FUSE /
    range-HTTP / forward streaming): sequential reads consume from a
    long-lived chunk iterator; non-sequential reads restart the
    iterator at the chunk covering the requested offset (one chunk's
    decode overhead).

    Concurrency: a single :class:`BedEncoder` instance is **not**
    thread-safe — :meth:`read` and :meth:`close` must be serialised
    by the caller (one encoder per consumer: thread, FUSE handle,
    range-HTTP connection). Multiple :class:`BedEncoder` instances
    may share one :class:`~vcztools.retrieval.VczReader` safely;
    each encoder runs an independent variant-chunk iteration on the
    reader. The caller owns the reader's lifetime — :meth:`close`
    tears down the encoder's iterator only, not the reader.

    Scope is the ``.bed`` stream only. For the companion ``.bim`` and
    ``.fam`` files, use :func:`generate_bim` and :func:`generate_fam`
    directly.

    Honours :meth:`~vcztools.retrieval.VczReader.set_samples` and
    :meth:`~vcztools.retrieval.VczReader.set_variants` configured on
    the reader before construction. With ``set_variants``, the encoded
    ``.bed`` covers exactly the selected variants in chunk-plan order;
    :attr:`bed_size` and :attr:`num_variants` reflect the selection.

    Construction is I/O-free: byte offsets are derived from the
    reader's chunk plan via
    :meth:`~vcztools.retrieval.VczReader.variant_counts_per_chunk`.
    Biallelic checking is performed lazily as chunks are decoded;
    multi-allelic variants raise ``ValueError`` during :meth:`read`,
    not at construction.

    :meth:`~vcztools.retrieval.VczReader.set_variant_filter` is not
    yet supported and raises ``NotImplementedError`` at construction;
    apply predicate filters externally before passing the reader in.

    Per-chunk PLINK encoding parallelises across variant-axis
    sub-blocks via a :class:`concurrent.futures.ThreadPoolExecutor`
    owned by the encoder. ``encode_threads`` (default 4) sets the
    pool size; ``encode_block_bytes`` (default 10 MiB) is the
    input-bytes target per sub-block. Chunks at or below the
    threshold encode synchronously on the calling thread. The pool
    is created in ``__init__`` and joined in :meth:`close` /
    :meth:`__exit__`.
    """

    BED_MAGIC: ClassVar[bytes] = BED_MAGIC

    def __init__(
        self,
        reader: retrieval.VczReader,
        *,
        encode_threads: int | None = None,
        encode_block_bytes: int | None = None,
    ):
        if encode_threads is None:
            encode_threads = 4
        if encode_block_bytes is None:
            encode_block_bytes = 10 * 1024 * 1024

        if reader.variant_filter is not None:
            raise NotImplementedError(
                "BedEncoder does not yet support readers with a "
                "set_variant_filter() configured. Apply the filter "
                "externally and pass the resulting reader, or use "
                "set_variants() to materialise the surviving indices."
            )
        if encode_threads < 1:
            raise ValueError(f"encode_threads must be >= 1 (got {encode_threads})")
        if encode_block_bytes < 1:
            raise ValueError(
                f"encode_block_bytes must be >= 1 (got {encode_block_bytes})"
            )

        self._reader = reader
        self._closed = False

        self._num_samples = int(reader.sample_ids.size)
        self._bytes_per_variant = (self._num_samples + 3) // 4

        # _chunk_byte_offsets: cumulative byte offset per plan entry,
        # starting at 3 (post-magic). Length = len(plan) + 1; the
        # trailing entry equals bed_size.
        self._compute_offsets()
        self._bed_size = int(self._chunk_byte_offsets[-1])

        self._iterator = None
        self._cur_byte_offset = None
        self._pending = bytearray()
        self._first_chunk_skip = 0

        self._encode_threads = encode_threads
        self._encode_block_bytes = encode_block_bytes
        self._executor = cf.ThreadPoolExecutor(
            max_workers=encode_threads,
            thread_name_prefix="vcztools-encode-plink",
        )

    def _compute_offsets(self) -> None:
        bpv = self._bytes_per_variant
        counts = self._reader.variant_counts_per_chunk()
        self._num_variants = int(counts.sum())
        self._chunk_byte_offsets = np.empty(len(counts) + 1, dtype=np.int64)
        self._chunk_byte_offsets[0] = 3
        np.cumsum(counts * bpv, out=self._chunk_byte_offsets[1:])
        self._chunk_byte_offsets[1:] += 3

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("encoder closed")

    @property
    def num_variants(self) -> int:
        self._check_open()
        return self._num_variants

    @property
    def num_samples(self) -> int:
        self._check_open()
        return self._num_samples

    @property
    def bytes_per_variant(self) -> int:
        self._check_open()
        return self._bytes_per_variant

    @property
    def bed_size(self) -> int:
        self._check_open()
        return self._bed_size

    def read(self, off: int, size: int) -> bytes:
        """Return up to ``size`` bytes from the virtual ``.bed`` at
        ``off``. POSIX-read semantics:

        - ``b""`` if ``off >= bed_size`` or ``size == 0``
        - ``size`` clamped to the end of the file
        - ``off < 0`` or ``size < 0`` raises ``ValueError``

        Sequential calls consume from a running chunk iterator;
        non-sequential calls reconstruct the iterator at ``off``'s
        chunk (one chunk's worth of decode overhead).
        """
        self._check_open()
        if off < 0:
            raise ValueError(f"off must be >= 0 (got {off})")
        if size < 0:
            raise ValueError(f"size must be >= 0 (got {size})")
        if off >= self._bed_size or size == 0:
            return b""
        end = min(off + size, self._bed_size)
        if self._cur_byte_offset != off:
            self._restart(off)
        out = self._drain(end - off)
        self._cur_byte_offset = off + len(out)
        return out

    def _restart(self, off: int) -> None:
        self._teardown_iterator()
        if off < 3:
            self._pending = bytearray(BED_MAGIC[off:3])
            self._first_chunk_skip = 0
            plan_pos = 0
        else:
            # searchsorted(side="right") - 1: largest plan position
            # whose start offset is <= off. _chunk_byte_offsets has
            # len(plan)+1 entries (the trailing entry is bed_size);
            # off < bed_size is guaranteed by read(), so the index is
            # in range.
            plan_pos = int(
                np.searchsorted(self._chunk_byte_offsets, off, side="right") - 1
            )
            self._first_chunk_skip = off - int(self._chunk_byte_offsets[plan_pos])
            self._pending = bytearray()
        self._iterator = self._reader.variant_chunks(
            fields=["call_genotype", "variant_allele"],
            start=plan_pos,
        )
        self._cur_byte_offset = off

    def _drain(self, n: int) -> bytes:
        out = bytearray()
        if len(self._pending) > 0:
            take = min(n, len(self._pending))
            out += self._pending[:take]
            del self._pending[:take]
            n -= take
        while n > 0:
            chunk = next(self._iterator)
            _check_biallelic(chunk["variant_allele"])
            G = chunk["call_genotype"]
            encoded = self._encode_genotypes(G)
            if self._first_chunk_skip > 0:
                encoded = encoded[self._first_chunk_skip :]
                self._first_chunk_skip = 0
            take = min(n, len(encoded))
            out += encoded[:take]
            self._pending = bytearray(encoded[take:])
            n -= take
        return bytes(out)

    def _encode_genotypes(self, genotypes: np.ndarray):
        # Coerce once at method entry: a sample-subset call_genotype
        # is fancy-indexed and non-contiguous, so the copy must happen
        # before slicing. Once G is C-contiguous, axis-0 sub-blocks
        # are zero-copy views.
        G = np.ascontiguousarray(genotypes, dtype=np.int8)
        if self._encode_threads <= 1 or G.nbytes <= self._encode_block_bytes:
            return bytes(_vcztools.encode_plink(G).data)

        num_variants = G.shape[0]
        bytes_per_variant = self._bytes_per_variant
        block_variants = max(1, self._encode_block_bytes // (G.shape[1] * 2))
        # Pre-allocate the full chunk's output once; each completed
        # future's result is copied into its variant-aligned slice.
        # Slicing along axis 0 (variants) preserves per-row independence:
        # the C kernel writes one row of bytes_per_variant per variant.
        output = np.empty(num_variants * bytes_per_variant, dtype=np.uint8)

        future_to_start = {}
        for start in range(0, num_variants, block_variants):
            block = G[start : start + block_variants]
            future = self._executor.submit(_vcztools.encode_plink, block)
            future_to_start[future] = start

        for future in cf.as_completed(future_to_start):
            start = future_to_start[future]
            end = min(start + block_variants, num_variants)
            out_start = start * bytes_per_variant
            out_end = end * bytes_per_variant
            output[out_start:out_end] = future.result()

        return output.data

    def _teardown_iterator(self) -> None:
        if self._iterator is not None:
            self._iterator.close()
            self._iterator = None
        self._pending = bytearray()
        self._first_chunk_skip = 0

    def close(self) -> None:
        """Tear down the active chunk iterator, shut down the encode
        thread pool, and drop iterator state. Does not close the
        underlying reader. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._teardown_iterator()
        self._executor.shutdown(wait=True)
        self._cur_byte_offset = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
