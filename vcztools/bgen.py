"""
Convert VCZ to Oxford BGEN format (``.bgen`` + ``.sample`` + ``.bgen.bgi``).

The CLI verb is ``view-bgen``. Output profile: layout 2, zlib-compressed,
8 bits/probability, biallelic, diploid, embedded sample IDs. This is the
consumer lowest-common-denominator: REGENIE, SAIGE, BOLT-LMM, BGENIE,
qctool, and PLINK 2 all accept it without further conversion.

Hard calls in ``call_genotype`` are encoded as 1.0/0.0 probabilities; at
8-bit precision this round-trips exactly. Phase is propagated per-variant
from ``call_genotype_phased`` if present (a variant emits phased iff
every sample is phased for that variant).

For FUSE / HTTP-range-serving applications that need random-access into
the encoded byte stream, :class:`BgenEncoder` is a sibling of
:class:`vcztools.plink.BedEncoder` that produces a fixed-size BGEN
stream (Python API only).

For the user-facing reference — multi-allelic policy, downstream-tool
compatibility, sidecar conventions — see ``docs/bgen.md``.
"""

import dataclasses
import logging
import pathlib
import sqlite3
import struct
import time
import zlib
from typing import ClassVar

import numpy as np

from vcztools import format_encoder, retrieval

logger = logging.getLogger(__name__)

BGEN_MAGIC = b"bgen"
LAYOUT_2 = 2
COMPRESSION_ZLIB = 1
SAMPLE_IDS_PRESENT = 1 << 31
HEADER_LENGTH = 20
BITS_PER_PROB = 8

_PLOIDY_DIPLOID = 0x02
_PLOIDY_MISSING = 0x82  # high bit (missing) | ploidy 2


def _check_biallelic(alleles):
    # BGEN layout 2 supports multi-allelic, but every major downstream
    # consumer (REGENIE, SAIGE, BOLT-LMM) assumes biallelic. Mirroring
    # view-plink: reject up-front; pass --max-alleles 2 to skip them.
    if alleles.shape[1] > 2 and (alleles[:, 2:] != "").any():
        raise ValueError(
            "Multi-allelic variants are not supported in BGEN output. "
            "Use --max-alleles 2 to skip them, or split with bcftools "
            "norm -m- before conversion."
        )


def _check_diploid(genotypes):
    if genotypes.ndim != 3 or genotypes.shape[2] != 2:
        raise ValueError(
            f"BGEN output requires diploid genotypes "
            f"(call_genotype.shape[2] == 2); got shape {genotypes.shape!r}."
        )


def _flags_word():
    return COMPRESSION_ZLIB | (LAYOUT_2 << 2) | SAMPLE_IDS_PRESENT


def _build_sample_id_block(sample_ids):
    """Build the BGEN sample identifier block bytes.

    Layout:
        uint32 L_SI       length of this block, including these 4 bytes
        uint32 N          n_samples
        for each sample:
          uint16 L_id_i   length of sample id i
          bytes  id_i     UTF-8 sample id
    """
    n = len(sample_ids)
    parts = []
    body = bytearray()
    for sid in sample_ids:
        encoded = str(sid).encode("utf-8")
        if len(encoded) > 0xFFFF:
            raise ValueError(
                f"Sample ID {sid!r} exceeds BGEN's 65535-byte sample-id limit."
            )
        body += struct.pack("<H", len(encoded))
        body += encoded
    block_length = 4 + 4 + len(body)  # L_SI + N + body
    parts.append(struct.pack("<II", block_length, n))
    parts.append(bytes(body))
    return b"".join(parts)


def _build_header(num_variants, num_samples, sample_id_block):
    """Build the 4-byte offset prefix + 20-byte header + sample-id block."""
    # offset is measured from the end of itself (i.e. from byte 4): it
    # equals the header length plus the sample-id block length.
    offset = HEADER_LENGTH + len(sample_id_block)
    out = bytearray()
    out += struct.pack("<I", offset)
    out += struct.pack("<I", HEADER_LENGTH)
    out += struct.pack("<I", num_variants)
    out += struct.pack("<I", num_samples)
    out += BGEN_MAGIC
    out += struct.pack("<I", _flags_word())
    out += sample_id_block
    return bytes(out)


def _encode_field_chunk(values, *, max_len=None, field_name=None):
    """Vectorised UTF-8 byte preparation for a chunk's worth of variant
    string-field values.

    ``values`` is a numpy string array of length ``N``. Returns a
    ``len()``-able sequence whose element ``i`` is the UTF-8 byte
    form of ``values[i]``, optionally NUL-padded.

    - ``max_len is None``: variable mode. Returns an ``S``-dtype
      numpy array; element ``i`` is the raw UTF-8 form of
      ``values[i]``. ``len(arr[i])`` equals the actual byte length.
    - ``max_len is int``: fixed mode. Returns a 2D ``uint8`` array
      of shape ``(N, max_len)``; each row is NUL-padded to exactly
      ``max_len`` bytes. Overflow raises ``ValueError`` naming
      ``field_name`` and the offending value. The 2D ``uint8`` view
      is necessary because numpy's ``S``-dtype strips trailing NULs
      on element access — ``len(arr[i])`` would return the unpadded
      length, breaking the length-prefix invariant.

    Both modes preserve the invariant ``len(arr[i]) == length-prefix``
    so the downstream block encoder can read the prefix off the bytes
    themselves.
    """
    encoded = np.strings.encode(values, "utf-8")
    if max_len is None:
        return encoded
    lens = np.strings.str_len(encoded)
    if lens.size > 0 and int(lens.max()) > max_len:
        bad = int(np.argmax(lens > max_len))
        raise ValueError(
            f"{field_name} {str(values[bad])!r} encodes to "
            f"{int(lens[bad])} UTF-8 bytes, exceeds configured max "
            f"of {max_len}"
        )
    return encoded.astype(f"S{max_len}").view(np.uint8).reshape(-1, max_len)


@dataclasses.dataclass
class _ChunkBytes:
    """Per-chunk byte-prepared inputs to the BGEN variant-block encoder.

    The five string ``*`` arrays follow :func:`_encode_field_chunk`'s
    dtype convention: an ``S``-dtype array in variable mode (length of
    element ``i`` is the actual UTF-8 byte length) or a
    ``(N, max_len) uint8`` view in fixed mode (length of element ``i``
    is ``max_len``). Either way ``len(arr[i])`` equals the BGEN
    length-prefix value that :func:`_encode_variant_block` writes.

    ``geno_blocks`` holds the uncompressed BGEN genotype-block bytes
    for every variant in the chunk: shape ``(V, 10 + 3 * num_samples)``
    uint8. ``_encode_variant_block`` zlib-compresses one row at a time
    and packs the surrounding variant-block framing.
    """

    varid: np.ndarray
    rsid: np.ndarray
    chrom: np.ndarray
    allele1: np.ndarray
    allele2: np.ndarray
    position: np.ndarray
    geno_blocks: np.ndarray
    mixed_phase_count: int


def _prepare_chunk(
    chunk,
    contig_ids,
    *,
    start,
    end,
    varid_max_len=None,
    rsid_max_len=None,
    chrom_max_len=None,
    allele_max_len=None,
):
    """Validate and prepare per-variant byte inputs for variants
    ``[start:end]`` of ``chunk``. ``contig_ids`` resolves
    ``variant_contig`` integer indices to chromosome name strings.
    Each ``*_max_len`` controls whether the corresponding string field
    is NUL-padded to a fixed width (``int``) or written at its actual
    UTF-8 byte length (``None``).

    Mirrors the conventions shared by :class:`BgenEncoder` and
    :func:`write_bgen`: ``rsid=""`` and ``allele2=""`` normalise to
    ``"."``; variant id mirrors rsid; the variant is treated as phased
    only when *every* sample is phased on that variant
    (``mixed_phase_count`` records partial-phase variants).

    ``varid`` is validated before ``rsid`` so a too-narrow
    ``varid_max_len`` surfaces as a ``"varid"`` error even when both
    fields would overflow with the default ``rsid_max_len=64``.

    Chunk-axis reads are sliced at the top of the body so workers can
    invoke this function on disjoint variant ranges concurrently —
    ``BgenEncoder._encode_chunk`` dispatches one call per worker per
    slice via :meth:`_parallel_encode`.
    """
    G = chunk["call_genotype"][start:end]
    alleles = chunk["variant_allele"][start:end]
    positions = chunk["variant_position"][start:end]
    contigs = chunk["variant_contig"][start:end]
    varids = chunk.get("variant_id")
    if varids is not None:
        varids = varids[start:end]
    phased_arr = chunk.get("call_genotype_phased")
    if phased_arr is not None:
        phased_arr = phased_arr[start:end]

    _check_biallelic(alleles)
    _check_diploid(G)
    n = G.shape[0]

    chrom_arr = np.asarray(contig_ids)[contigs]
    a1_arr = alleles[:, 0]
    if alleles.shape[1] >= 2:
        a2_arr = np.where(alleles[:, 1] == "", ".", alleles[:, 1])
    else:
        a2_arr = np.full(n, ".")
    if varids is not None:
        rsid_arr = np.where(varids == "", ".", varids)
    else:
        rsid_arr = np.full(n, ".")

    varid_b = _encode_field_chunk(rsid_arr, max_len=varid_max_len, field_name="varid")
    rsid_b = _encode_field_chunk(rsid_arr, max_len=rsid_max_len, field_name="rsid")
    chrom_b = _encode_field_chunk(chrom_arr, max_len=chrom_max_len, field_name="chrom")
    a1_b = _encode_field_chunk(a1_arr, max_len=allele_max_len, field_name="allele1")
    a2_b = _encode_field_chunk(a2_arr, max_len=allele_max_len, field_name="allele2")

    if phased_arr is not None:
        all_phased = phased_arr.all(axis=1)
        any_phased = phased_arr.any(axis=1)
        mixed_phase_count = int(np.sum(any_phased & ~all_phased))
        phased = np.asarray(all_phased)
    else:
        mixed_phase_count = 0
        phased = np.zeros(n, dtype=bool)

    geno_blocks = _build_geno_blocks(G, phased)

    return _ChunkBytes(
        varid=varid_b,
        rsid=rsid_b,
        chrom=chrom_b,
        allele1=a1_b,
        allele2=a2_b,
        position=positions,
        geno_blocks=geno_blocks,
        mixed_phase_count=mixed_phase_count,
    )


def _build_geno_blocks(G, phased):
    """Vectorised build of every variant's uncompressed BGEN
    genotype-block bytes for one chunk. Returns a
    ``(V, 10 + 3 * num_samples)`` uint8 array; row ``i`` is the
    spec-compliant uncompressed block for variant ``i``.

    Layout per row (Layout 2 / 8-bit / biallelic / diploid):

        uint32 N            n_samples
        uint16 K = 2        n_alleles
        uint8  P_min = 2
        uint8  P_max = 2
        N bytes             ploidy/missing per sample
        uint8  phased       per-variant flag
        uint8  B = 8        bits per probability
        2*N bytes           per-sample probability bytes (B0, B1 interleaved)
    """
    v, s, _ = G.shape
    a = G[..., 0]
    b = G[..., 1]
    missing = (a < 0) | (b < 0)

    # Phased and unphased disagree on what B0/B1 carry; compute both
    # bitwise forms vectorised and select per variant with the phase
    # mask broadcast over the sample axis.
    a_zero = a == 0
    b_zero = b == 0
    homref = a_zero & b_zero
    het = (a_zero & (b == 1)) | ((a == 1) & b_zero)
    phased_row = phased[:, None]
    b0_bit = np.where(phased_row, a_zero, homref)
    b1_bit = np.where(phased_row, b_zero, het)
    B0 = np.where(b0_bit, 0xFF, 0).astype(np.uint8)
    B1 = np.where(b1_bit, 0xFF, 0).astype(np.uint8)
    # Spec says missing samples' probability bytes should be ignored;
    # zero them so the bytes are deterministic for testing / hashing.
    B0[missing] = 0
    B1[missing] = 0

    ploidy = np.full((v, s), _PLOIDY_DIPLOID, dtype=np.uint8)
    ploidy[missing] = _PLOIDY_MISSING

    geno_blocks = np.empty((v, 10 + 3 * s), dtype=np.uint8)
    header = struct.pack("<IHBB", s, 2, 2, 2)  # 8 bytes; constant per chunk
    geno_blocks[:, 0:8] = np.frombuffer(header, dtype=np.uint8)
    geno_blocks[:, 8 : 8 + s] = ploidy
    geno_blocks[:, 8 + s] = phased.astype(np.uint8)
    geno_blocks[:, 8 + s + 1] = BITS_PER_PROB
    # Per-sample probability bytes interleave B0 then B1; matches the
    # per-variant ordering produced by the legacy genotype block builder.
    geno_blocks[:, 8 + s + 2 :: 2] = B0
    geno_blocks[:, 8 + s + 3 :: 2] = B1
    return geno_blocks


def _encode_variant_block(
    *,
    varid_bytes,
    rsid_bytes,
    chrom_bytes,
    position,
    allele1_bytes,
    allele2_bytes,
    geno_block_bytes,
    compression_level,
):
    """Encode one full BGEN variant data block (identifying data +
    compressed genotype block).

    Layout:

        uint16 len(varid_bytes)  ; varid_bytes
        uint16 len(rsid_bytes)   ; rsid_bytes
        uint16 len(chrom_bytes)  ; chrom_bytes
        uint32 position
        uint16 K = 2
        uint32 len(allele1_bytes) ; allele1_bytes
        uint32 len(allele2_bytes) ; allele2_bytes
        uint32 C = 4 + len(compressed)
        uint32 D = len(uncompressed_geno_block)
        compressed (zlib, level = ``compression_level``)

    All string fields arrive pre-encoded as UTF-8 bytes via
    :func:`_encode_field_chunk` — in variable mode (``max_len=None``)
    their length is the actual encoding; in fixed mode their length is
    the configured ``*_max`` and the bytes carry trailing NUL padding.
    Either way ``len(bytes)`` is the length-prefix that goes on the
    wire, so the encoder doesn't need to know which mode it's in.

    ``geno_block_bytes`` is the uncompressed BGEN genotype-block bytes
    (spec layout: N/K/Pmin/Pmax header, ploidy, phased flag, B, prob
    bytes) — assembled at chunk-level by :func:`_build_geno_blocks`
    in :func:`_prepare_chunk`.

    bgen-reader strips trailing NULs from variable-length string
    fields, so the NUL-padded form round-trips as the original UTF-8
    string. ``compression_level`` is forwarded verbatim to
    :func:`zlib.compress`; :class:`BgenEncoder` always passes ``0``
    (stored) to keep block sizes deterministic.
    """
    compressed = zlib.compress(geno_block_bytes, compression_level)

    out = bytearray()
    out += struct.pack("<H", len(varid_bytes))
    out.extend(varid_bytes)
    out += struct.pack("<H", len(rsid_bytes))
    out.extend(rsid_bytes)
    out += struct.pack("<H", len(chrom_bytes))
    out.extend(chrom_bytes)
    out += struct.pack("<I", int(position))
    out += struct.pack("<H", 2)  # K = 2 alleles
    out += struct.pack("<I", len(allele1_bytes))
    out.extend(allele1_bytes)
    out += struct.pack("<I", len(allele2_bytes))
    out.extend(allele2_bytes)
    # C includes its own 4-byte size of D plus the compressed payload.
    C = 4 + len(compressed)
    D = len(geno_block_bytes)
    out += struct.pack("<II", C, D)
    out += compressed
    return bytes(out)


def _detect_variant_phase(phased_array_for_variant):
    """All samples True → phased. Otherwise (incl. all-False and mixed)
    → unphased. BGEN has one phase flag per variant; mixed-phase
    variants degrade silently to unphased."""
    return bool(phased_array_for_variant.all())


class BgenEncoder(format_encoder.FormatEncoder):
    """Random-access, fixed-size BGEN byte-stream encoder over a VCZ store.

    Thin :class:`~vcztools.format_encoder.FormatEncoder` subclass: the
    base class supplies the chunk-resident state machine, POSIX-style
    :meth:`read`, iterator restart/advance arbitration, thread-pool
    lifecycle, and prefix (BGEN header) serving. ``BgenEncoder`` plugs
    in the layout-2 header bytes and the per-chunk fixed-size variant
    encoding.

    The byte stream is a valid BGEN layout-2 file with the compression
    flag set to ``ZLIB``. Every variant block uses zlib **level 0**
    (stored, no DEFLATE) so the compressed payload size is a deterministic
    function of the uncompressed genotype block size — the variant block
    is therefore exactly :attr:`bytes_per_variant` bytes wide and
    ``byte offset → variant index`` is O(1):

        bytes_per_variant
          = 28 + varid_max + rsid_max + chrom_max
              + 2 * allele_max + zlib_stored_size(10 + 3 * num_samples)

    where the constant 28 = 3 * 2 (string length prefixes, uint16) + 4
    (position) + 2 (K) + 2 * 4 (allele length prefixes, uint32) + 2 * 4
    (C and D length prefixes, uint32).

    Variable-length string fields (variant id, rsid, chromosome, alleles)
    are NUL-padded to the configured ``*_max`` widths. The bgen-reader
    reference reader strips trailing NULs from string fields, so the
    padded values round-trip cleanly. Overflowing any configured max
    raises :class:`ValueError` during chunk encoding, naming the field
    and the offending value.

    Defaults are tuned for biobank biallelic SNP arrays:
    ``allele_max=1`` for SNPs; ``rsid_max=varid_max=64`` when the
    source has a ``variant_id`` field, else ``1`` (the encoder emits
    ``"."`` for every rsid in that case). ``chrom_max`` is always
    derived from ``reader.contig_ids`` — no manual knob. Indel or SV
    stores must opt in to larger ``allele_max`` / ``rsid_max`` values.

    Variant scope: biallelic and diploid (multi-allelic / non-diploid
    raises ``ValueError`` lazily as chunks are decoded).

    The encoder serves only the ``.bgen`` byte stream. The matching
    bgenix ``.bgi`` SQLite sidecar can be produced by passing
    :attr:`variant_offsets` (computed from the encoder's fixed-size
    layout ``header_size + i * bytes_per_variant``) to the module-level
    :func:`write_bgen_index`. The ``.sample`` sidecar is produced by
    :func:`generate_sample`.

    :meth:`~vcztools.retrieval.VczReader.set_variant_filter` is not
    supported and raises ``NotImplementedError`` at construction;
    materialise the filter or use ``set_variants`` first. Unlike
    :func:`write_bgen`, the encoder is I/O-free in ``__init__``.
    """

    HEADER_FLAGS: ClassVar[int] = _flags_word()

    def __init__(
        self,
        reader: retrieval.VczReader,
        *,
        varid_max: int | None = None,
        rsid_max: int | None = None,
        allele_max: int | None = None,
        encode_threads: int | None = None,
        encode_block_bytes: int | None = None,
    ):
        has_variant_id = "variant_id" in reader.field_names
        if rsid_max is None:
            # When variant_id is absent, _encode_chunk emits "." (1 byte)
            # for every rsid — no point reserving 64 bytes per variant.
            rsid_max = 64 if has_variant_id else 1
        if varid_max is None:
            # _encode_chunk sets varid = rsid, so they share fate.
            varid_max = rsid_max
        if allele_max is None:
            allele_max = 1

        # chrom_max is derived from contig_ids via vectorised numpy ops:
        # str_len on the UTF-8-encoded form gives byte length per element.
        # The max(initial=1) floor covers empty contig_ids and the all-
        # empty-string degenerate case; bytes_per_variant needs >= 1.
        contig_ids = reader.contig_ids
        chrom_max = int(
            np.strings.str_len(np.strings.encode(contig_ids)).max(initial=1)
        )

        # rsid_max validated before varid_max: when the user passes
        # rsid_max=X (and varid_max=None), varid_max propagates to X
        # above, so rsid_max owns the error message.
        for name, value, ceiling in (
            ("rsid_max", rsid_max, 0xFFFF),
            ("varid_max", varid_max, 0xFFFF),
            ("allele_max", allele_max, 0xFFFFFFFF),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1 (got {value})")
            if value > ceiling:
                raise ValueError(
                    f"{name}={value} exceeds BGEN length-prefix width {ceiling}"
                )
        if chrom_max > 0xFFFF:
            raise ValueError(
                f"longest contig is {chrom_max} bytes, exceeds BGEN "
                f"length-prefix width 65535"
            )

        self._varid_max = varid_max
        self._rsid_max = rsid_max
        self._chrom_max = chrom_max
        self._allele_max = allele_max
        self._has_variant_id = has_variant_id
        self._has_phased = "call_genotype_phased" in reader.field_names
        self._contig_ids = contig_ids
        self._mixed_phase_count = 0

        num_samples = int(reader.sample_ids.size)
        # Pre-compute the zlib-stored size of one genotype block.
        # zlib.compress(level=0) emits stored DEFLATE blocks whose output
        # size is a deterministic function of the input length, so every
        # variant block is the same width.
        geno_size = 10 + 3 * num_samples
        compressed_geno_size = len(zlib.compress(b"\x00" * geno_size, 0))
        self._compressed_geno_size = compressed_geno_size
        bytes_per_variant = (
            28
            + varid_max
            + rsid_max
            + chrom_max
            + 2 * allele_max
            + compressed_geno_size
        )

        sample_id_block = _build_sample_id_block(reader.sample_ids)
        num_variants = int(reader.variant_counts_per_chunk().sum())
        prefix_bytes = _build_header(num_variants, num_samples, sample_id_block)

        iterator_fields = [
            "call_genotype",
            "variant_allele",
            "variant_contig",
            "variant_position",
        ]
        if self._has_variant_id:
            iterator_fields.append("variant_id")
        if self._has_phased:
            iterator_fields.append("call_genotype_phased")

        super().__init__(
            reader,
            bytes_per_variant=bytes_per_variant,
            prefix_bytes=prefix_bytes,
            iterator_fields=iterator_fields,
            encode_threads=encode_threads,
            encode_block_bytes=encode_block_bytes,
        )

    @property
    def header_size(self) -> int:
        """BGEN header byte length (alias of
        :attr:`~vcztools.format_encoder.FormatEncoder.prefix_size`)."""
        return self.prefix_size

    @property
    def bgen_size(self) -> int:
        """Total ``.bgen`` size in bytes (alias of
        :attr:`~vcztools.format_encoder.FormatEncoder.total_size`)."""
        return self.total_size

    def _encode_chunk(self, chunk: dict) -> bytes:
        # _prepare_chunk runs inside _encode_variant_range, so each
        # worker handles its slice's vectorised numpy prep in parallel
        # with the others. Per-slice mixed_phase_count partials are
        # appended to a closure-captured list (list.append is GIL-safe
        # under CPython) and summed once _parallel_encode returns.
        num_variants = int(chunk["call_genotype"].shape[0])
        phase_counts: list[int] = []

        def encode_range(start: int, end: int) -> bytes:
            return self._encode_variant_range(chunk, start, end, phase_counts)

        output = self._parallel_encode(
            num_variants=num_variants, encode_range=encode_range
        )
        self._mixed_phase_count += sum(phase_counts)
        return output

    def _encode_variant_range(
        self, chunk: dict, start: int, end: int, phase_counts: list[int]
    ) -> bytes:
        prep = _prepare_chunk(
            chunk,
            self._contig_ids,
            start=start,
            end=end,
            varid_max_len=self._varid_max,
            rsid_max_len=self._rsid_max,
            chrom_max_len=self._chrom_max,
            allele_max_len=self._allele_max,
        )
        phase_counts.append(prep.mixed_phase_count)
        bpv = self._bytes_per_variant
        n = end - start
        out = bytearray(n * bpv)
        for j in range(n):
            block = _encode_variant_block(
                varid_bytes=prep.varid[j],
                rsid_bytes=prep.rsid[j],
                chrom_bytes=prep.chrom[j],
                position=int(prep.position[j]),
                allele1_bytes=prep.allele1[j],
                allele2_bytes=prep.allele2[j],
                geno_block_bytes=bytes(prep.geno_blocks[j]),
                compression_level=0,
            )
            out[j * bpv : (j + 1) * bpv] = block
        return bytes(out)

    def _close_hook(self) -> None:
        if self._mixed_phase_count > 0:
            self._logger.warning(
                f"BgenEncoder: {self._mixed_phase_count} variant(s) had "
                "mixed phase across samples; emitted as unphased (BGEN "
                "has one phase flag per variant)."
            )

    @property
    def variant_offsets(self) -> np.ndarray:
        """Byte boundaries of every variant block in the encoded BGEN
        stream, of shape ``(num_variants + 1,)``: variant ``i`` occupies
        ``[variant_offsets[i], variant_offsets[i+1])``. Suitable for
        :func:`write_bgen_index`."""
        return self.prefix_size + self.bytes_per_variant * np.arange(
            self.num_variants + 1, dtype=np.int64
        )


def generate_sample(reader):
    """Oxford ``.sample`` text. Two header rows, then one row per sample.

    The minimal sample file: ``ID_1`` and ``ID_2`` are both the sample
    name (matches the ``--double-id`` style); ``missing`` is the
    per-sample missing-rate column with column-type ``0`` (identifier).
    Whitespace in sample IDs is rejected (the file format is
    whitespace-separated)."""
    sample_ids = reader.sample_ids
    for sid in sample_ids:
        s = str(sid)
        if any(c.isspace() for c in s):
            raise ValueError(
                f"Sample ID {s!r} contains whitespace; "
                "Oxford .sample files are whitespace-separated."
            )
    lines = ["ID_1 ID_2 missing", "0 0 0"]
    for sid in sample_ids:
        s = str(sid)
        lines.append(f"{s} {s} 0")
    return "\n".join(lines) + "\n"


_BGI_SCHEMA = """
CREATE TABLE Variant (
  chromosome TEXT NOT NULL,
  position INTEGER NOT NULL,
  rsid TEXT NOT NULL,
  number_of_alleles INTEGER NOT NULL,
  allele1 TEXT NOT NULL,
  allele2 TEXT NULL,
  file_start_position INTEGER NOT NULL,
  size_in_bytes INTEGER NOT NULL,
  PRIMARY KEY (chromosome, position, rsid, allele1, allele2, file_start_position)
);
CREATE INDEX position_idx ON Variant (chromosome, position);
"""


_BGI_INSERT_SQL = (
    "INSERT INTO Variant (chromosome, position, rsid, number_of_alleles, "
    "allele1, allele2, file_start_position, size_in_bytes) "
    "VALUES (?, ?, ?, 2, ?, ?, ?, ?)"
)


def write_bgen_index(reader, bgi_path, variant_offsets):
    """Write the bgenix ``.bgen.bgi`` SQLite sidecar for ``reader``.

    ``variant_offsets`` is an integer array of length
    ``num_variants + 1`` giving the byte boundaries of each variant
    block: variant ``i`` occupies
    ``[variant_offsets[i], variant_offsets[i+1])``. Use
    :attr:`BgenEncoder.variant_offsets` for the fixed-size encoder
    path, or the cumulative sum of per-variant block sizes (plus the
    BGEN prefix length) for the variable-size :func:`write_bgen`
    path.

    Variant-metadata columns (chromosome, position, rsid, alleles)
    are read via the reader's standard ``variant_chunks`` API and
    assembled as numpy arrays; per-row tuples are streamed into
    SQLite via a generator so peak memory stays at the column
    arrays rather than an N-row tuple list. Variant scope is
    biallelic only; multi-allelic raises ``ValueError`` lazily on
    the chunk containing the offending row.
    """
    start_total = time.perf_counter()

    num_variants = int(reader.variant_counts_per_chunk().sum())
    variant_offsets = np.asarray(variant_offsets)
    if variant_offsets.shape != (num_variants + 1,):
        raise ValueError(
            f"variant_offsets must have shape ({num_variants + 1},); "
            f"got {variant_offsets.shape}"
        )
    if not np.issubdtype(variant_offsets.dtype, np.integer):
        raise ValueError(
            f"variant_offsets must be integer-typed; got {variant_offsets.dtype}"
        )

    bgi_path = pathlib.Path(bgi_path)
    if bgi_path.exists():
        bgi_path.unlink()

    has_variant_id = "variant_id" in reader.field_names
    fields = ["variant_allele", "variant_contig", "variant_position"]
    if has_variant_id:
        fields.append("variant_id")
    contig_ids = np.asarray(reader.contig_ids)

    chrom_parts: list[np.ndarray] = []
    pos_parts: list[np.ndarray] = []
    a1_parts: list[np.ndarray] = []
    a2_parts: list[np.ndarray] = []
    rsid_parts: list[np.ndarray] = []

    read_start = time.perf_counter()
    for chunk in reader.variant_chunks(fields=fields):
        alleles = chunk["variant_allele"]
        _check_biallelic(alleles)
        n = alleles.shape[0]
        chrom_parts.append(contig_ids[chunk["variant_contig"]])
        pos_parts.append(np.asarray(chunk["variant_position"]))
        a1_parts.append(np.asarray(alleles[:, 0]))
        if alleles.shape[1] >= 2:
            a2_parts.append(np.asarray(alleles[:, 1]))
        else:
            a2_parts.append(np.full(n, "."))
        if has_variant_id:
            rsid_parts.append(np.asarray(chunk["variant_id"]))
        else:
            rsid_parts.append(np.full(n, "."))
    read_seconds = time.perf_counter() - read_start

    assemble_start = time.perf_counter()
    rows_iter = None
    if num_variants > 0:
        chrom_arr = np.concatenate(chrom_parts)
        pos_arr = np.concatenate(pos_parts)
        a1_arr = np.concatenate(a1_parts)
        a2_arr = np.concatenate(a2_parts)
        rsid_arr = np.concatenate(rsid_parts)
        # Match the .bim/.bgi monomorphic / missing-rsid convention used
        # by write_bgen and BgenEncoder.
        rsid_arr = np.where(rsid_arr == "", ".", rsid_arr)
        a2_arr = np.where(a2_arr == "", ".", a2_arr)
        if chrom_arr.shape[0] != num_variants:
            raise ValueError(
                f"reader yielded {chrom_arr.shape[0]} variants but "
                f"variant_offsets implies {num_variants}"
            )
        starts = variant_offsets[:-1]
        sizes = np.diff(variant_offsets)

        def rows_iter():
            for i in range(num_variants):
                yield (
                    str(chrom_arr[i]),
                    int(pos_arr[i]),
                    str(rsid_arr[i]),
                    str(a1_arr[i]),
                    str(a2_arr[i]),
                    int(starts[i]),
                    int(sizes[i]),
                )

    assemble_seconds = time.perf_counter() - assemble_start

    insert_start = time.perf_counter()
    conn = sqlite3.connect(str(bgi_path))
    try:
        conn.executescript(_BGI_SCHEMA)
        if num_variants > 0:
            conn.executemany(_BGI_INSERT_SQL, rows_iter())
        conn.commit()
    finally:
        conn.close()
    insert_seconds = time.perf_counter() - insert_start

    elapsed = time.perf_counter() - start_total
    mib = bgi_path.stat().st_size / (1024 * 1024)
    rate = mib / elapsed if elapsed > 0 else 0.0
    logger.info(
        f"write_bgen_index: wrote {num_variants} variants ({mib:.2f} MiB) "
        f"to {bgi_path} in {elapsed:.2f}s ({rate:.1f} MiB/s); "
        f"read={read_seconds:.2f}s, assemble={assemble_seconds:.2f}s, "
        f"insert={insert_seconds:.2f}s"
    )


def write_bgen(reader, out, *, compression_level: int = -1):
    """Write an Oxford BGEN fileset (``.bgen`` / ``.sample`` /
    ``.bgen.bgi``) for ``reader`` under prefix ``out``.

    If a variant filter is configured on the reader, it is resolved
    in place via :meth:`~vcztools.retrieval.VczReader.materialise_variant_filter`
    so that variant-block ordering and ``.bgi`` entries align with the
    BGEN payload.

    Variant scope: biallelic only (multi-allelic raises ``ValueError``).
    Genotype source: hard calls from ``call_genotype`` encoded as 1.0
    probability on the called genotype (8-bit precision round-trips
    exactly). Phase: per-variant from ``call_genotype_phased`` if the
    field exists in the store; otherwise unphased.

    ``compression_level`` is forwarded to :func:`zlib.compress` for each
    variant's genotype probability block; accepts ``-1..9`` (``-1`` =
    zlib default ≈ level 6; ``0`` = stored, still framed as zlib;
    ``9`` = maximum). The BGEN flag word always advertises
    ``COMPRESSION_ZLIB`` regardless of level.
    """
    reader.materialise_variant_filter()
    out_prefix = pathlib.Path(out)
    bgen_path = out_prefix.with_suffix(".bgen")
    sample_path = out_prefix.with_suffix(".sample")
    bgi_path = pathlib.Path(str(bgen_path) + ".bgi")

    with open(sample_path, "w") as f:
        f.write(generate_sample(reader))

    variant_offsets = _stream_bgen_to_file(
        reader, bgen_path, compression_level=compression_level
    )
    write_bgen_index(reader, bgi_path, variant_offsets)


def _stream_bgen_to_file(reader, bgen_path, *, compression_level: int):
    start = time.perf_counter()
    encode_seconds = 0.0
    write_seconds = 0.0
    bytes_written = 0

    num_variants = int(reader.variant_counts_per_chunk().sum())
    sample_ids = reader.sample_ids
    num_samples = int(sample_ids.size)

    contig_id = reader.contig_ids
    has_variant_id = "variant_id" in reader.field_names
    has_phased = "call_genotype_phased" in reader.field_names

    fields = ["call_genotype", "variant_allele", "variant_contig", "variant_position"]
    if has_variant_id:
        fields.append("variant_id")
    if has_phased:
        fields.append("call_genotype_phased")

    sample_id_block = _build_sample_id_block(sample_ids)
    header_bytes = _build_header(num_variants, num_samples, sample_id_block)

    variant_offsets = np.empty(num_variants + 1, dtype=np.int64)
    variant_offsets[0] = len(header_bytes)
    mixed_phase_count = 0
    idx = 0

    with open(bgen_path, "wb") as bgen_file:
        bgen_file.write(header_bytes)
        bytes_written += len(header_bytes)

        for chunk in reader.variant_chunks(fields=fields):
            t_prep = time.perf_counter()
            n = int(chunk["call_genotype"].shape[0])
            prep = _prepare_chunk(chunk, contig_id, start=0, end=n)
            mixed_phase_count += prep.mixed_phase_count
            encode_seconds += time.perf_counter() - t_prep

            for j in range(n):
                t0 = time.perf_counter()
                block = _encode_variant_block(
                    varid_bytes=prep.varid[j],
                    rsid_bytes=prep.rsid[j],
                    chrom_bytes=prep.chrom[j],
                    position=int(prep.position[j]),
                    allele1_bytes=prep.allele1[j],
                    allele2_bytes=prep.allele2[j],
                    geno_block_bytes=bytes(prep.geno_blocks[j]),
                    compression_level=compression_level,
                )
                t1 = time.perf_counter()
                bgen_file.write(block)
                t2 = time.perf_counter()
                encode_seconds += t1 - t0
                write_seconds += t2 - t1

                variant_offsets[idx + 1] = variant_offsets[idx] + len(block)
                idx += 1
                bytes_written += len(block)

    elapsed = time.perf_counter() - start
    mib = bytes_written / (1024 * 1024)
    rate = mib / elapsed if elapsed > 0 else 0.0
    logger.info(
        f"write_bgen: wrote {mib:.1f} MiB to {bgen_path} in "
        f"{elapsed:.2f}s ({rate:.1f} MiB/s); "
        f"compression_level={compression_level}; "
        f"encode={encode_seconds:.2f}s, write={write_seconds:.2f}s"
    )
    if mixed_phase_count > 0:
        logger.warning(
            f"write_bgen: {mixed_phase_count} variant(s) had mixed phase "
            "across samples; emitted as unphased (BGEN has one phase flag "
            "per variant)."
        )
    return variant_offsets
