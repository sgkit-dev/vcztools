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

import pathlib
from typing import ClassVar

import numpy as np
import pandas as pd

from vcztools import _vcztools, retrieval
from vcztools.utils import _as_fixed_length_unicode

BED_MAGIC = b"\x6c\x1b\x01"


class MaxAllelesFilter:
    """Variant-scope :class:`~vcztools.variant_filter.VariantFilter` that
    keeps variants whose number of non-empty alleles is at most
    ``max_alleles``.

    For PLINK 1 binary output this is invoked with ``max_alleles=2``,
    matching ``plink2 --vcf X --make-bed --max-alleles 2``.

    Operates on ``variant_allele`` only; the per-variant decision is
    independent of ``call_genotype`` and any sample subset applied
    downstream.
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


def encode_genotypes(genotypes):
    # A1 = ALT (allele index 1), A2 = REF (allele index 0): plink 2's
    # --vcf X --make-bed convention. The C extension requires a
    # C-contiguous int8 array; a reader-yielded call_genotype that's
    # been reordered by a sample subset is fancy-indexed and not
    # contiguous, so force a copy when needed.
    G = np.ascontiguousarray(genotypes, dtype=np.int8)
    return bytes(_vcztools.encode_plink(G).data)


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


class Writer:
    def __init__(self, reader, bed_path, fam_path, bim_path):
        self.reader = reader

        self.bim_path = bim_path
        self.fam_path = fam_path
        self.bed_path = bed_path

    def _write_genotypes(self):
        ci = self.reader.variant_chunks(fields=["call_genotype", "variant_allele"])
        with open(self.bed_path, "wb") as bed_file:
            bed_file.write(BED_MAGIC)

            for chunk in ci:
                G = chunk["call_genotype"]
                if G.ndim != 3 or G.shape[2] != 2:
                    raise ValueError(
                        "Only diploid genotypes are supported "
                        f"(call_genotype has shape {G.shape})"
                    )
                _check_biallelic(chunk["variant_allele"])
                bed_file.write(encode_genotypes(G))

    def run(self):
        self._write_genotypes()

        with open(self.bim_path, "w") as f:
            f.write(generate_bim(self.reader))

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
    Construction performs an eager biallelic check by reading
    ``variant_allele`` for the selected chunks only — variants outside
    the selection are not inspected.

    :meth:`~vcztools.retrieval.VczReader.set_variant_filter` is not
    yet supported and raises ``NotImplementedError`` at construction;
    apply predicate filters externally before passing the reader in.
    """

    BED_MAGIC: ClassVar[bytes] = BED_MAGIC

    def __init__(self, reader: retrieval.VczReader):
        if reader.variant_filter is not None:
            raise NotImplementedError(
                "BedEncoder does not yet support readers with a "
                "set_variant_filter() configured. Apply the filter "
                "externally and pass the resulting reader, or use "
                "set_variants() to materialise the surviving indices."
            )

        self._reader = reader
        self._closed = False

        self._num_samples = int(reader.sample_ids.size)
        self._bytes_per_variant = (self._num_samples + 3) // 4

        # _chunk_byte_offsets: cumulative byte offset per yielded chunk,
        # starting at 3 (post-magic). _chunk_plan_indices: plan position
        # of each yielded chunk, used by _restart to resume iteration.
        # Without a variant_filter (the only case currently supported),
        # plan_pos == yielded_pos, but we record the mapping explicitly
        # so the binary search in _restart doesn't bake in that
        # assumption.
        self._pre_walk()
        self._bed_size = int(self._chunk_byte_offsets[-1])

        self._iterator = None
        self._cur_byte_offset = None
        self._pending = bytearray()
        self._first_chunk_skip = 0

    def _pre_walk(self) -> None:
        bpv = self._bytes_per_variant
        byte_offsets = [3]
        plan_indices = []
        num_variants = 0
        for plan_pos, chunk_data in enumerate(
            self._reader.variant_chunks(fields=["variant_allele"])
        ):
            alleles = chunk_data["variant_allele"]
            _check_biallelic(alleles)
            n = len(alleles)
            num_variants += n
            plan_indices.append(plan_pos)
            byte_offsets.append(byte_offsets[-1] + n * bpv)
        self._chunk_byte_offsets = np.array(byte_offsets, dtype=np.int64)
        self._chunk_plan_indices = np.array(plan_indices, dtype=np.int64)
        self._num_variants = num_variants

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
            if self._chunk_plan_indices.size > 0:
                target_plan_pos = int(self._chunk_plan_indices[0])
            else:
                target_plan_pos = 0
        else:
            # searchsorted(side="right") - 1: largest yielded position
            # whose start offset is <= off. _chunk_byte_offsets has
            # len(yielded)+1 entries (the trailing entry is bed_size);
            # off < bed_size is guaranteed by read(), so the index is
            # in range.
            yielded_pos = (
                np.searchsorted(self._chunk_byte_offsets, off, side="right") - 1
            )
            target_plan_pos = int(self._chunk_plan_indices[yielded_pos])
            self._first_chunk_skip = off - int(self._chunk_byte_offsets[yielded_pos])
            self._pending = bytearray()
        self._iterator = self._reader.variant_chunks(
            fields=["call_genotype"],
            start=target_plan_pos,
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
            G = chunk["call_genotype"]
            encoded = encode_genotypes(G)
            if self._first_chunk_skip > 0:
                encoded = encoded[self._first_chunk_skip :]
                self._first_chunk_skip = 0
            take = min(n, len(encoded))
            out += encoded[:take]
            self._pending = bytearray(encoded[take:])
            n -= take
        return bytes(out)

    def _teardown_iterator(self) -> None:
        if self._iterator is not None:
            self._iterator.close()
            self._iterator = None
        self._pending = bytearray()
        self._first_chunk_skip = 0

    def close(self) -> None:
        """Tear down the active chunk iterator and drop iterator
        state. Does not close the underlying reader. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._teardown_iterator()
        self._cur_byte_offset = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
