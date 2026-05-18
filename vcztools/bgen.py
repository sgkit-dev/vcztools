"""
Convert VCZ to Oxford BGEN format.

The CLI verb is ``view-bgen``. By default the ``.bgen`` payload is streamed
to stdout; passing ``-o STEM`` switches to file output with the ``.bgen.bgi``
(bgenix SQLite index) and ``.sample`` (Oxford text) sidecars also produced
by default and individually suppressible. The Python API
(:func:`write_bgen`) is shaped the same way: the BGEN payload destination
is path-or-file-like, and each sidecar is requested by passing a
filesystem path (``None`` skips it). Output profile: layout 2,
zlib-compressed, 8 bits/probability, biallelic, embedded sample IDs in the
``.bgen`` header. Haploid and mixed-ploidy stores are supported via
per-sample ploidy bytes; this is the consumer lowest-common-denominator:
REGENIE, SAIGE, BOLT-LMM, BGENIE, qctool, and PLINK 2 all accept it
without further conversion.

Hard calls in ``call_genotype`` are encoded as 1.0/0.0 probabilities; at
8-bit precision this round-trips exactly. Phase is propagated per-variant
from ``call_genotype_phased`` if present (a variant emits phased iff
every sample is phased for that variant).

VCZ stores haploid genotypes one of two ways: as ``(V, S, 1)`` arrays,
or as ``(V, S, 2)`` arrays where slot 1 contains the ``-2`` haploid-
padding sentinel. The encoder accepts both; ``(V, S, 1)`` is promoted
to the ``-2``-padded form before the C kernel sees it.

For FUSE / HTTP-range-serving applications that need random-access into
the encoded byte stream, :class:`BgenEncoder` is a sibling of
:class:`vcztools.plink.BedEncoder` that produces a fixed-size BGEN
stream (Python API only). :class:`BgenEncoder` requires uniform ploidy
across the store (all haploid or all diploid); mixed-ploidy stores
must go through :func:`write_bgen`.

For the user-facing reference — multi-allelic policy, downstream-tool
compatibility, sidecar conventions — see ``docs/bgen.md``.
"""

import concurrent.futures as cf
import dataclasses
import logging
import pathlib
import sqlite3
import struct
import time
import zlib
from typing import ClassVar

import numpy as np

from vcztools import _vcztools, format_encoder, retrieval, utils

logger = logging.getLogger(__name__)

BGEN_MAGIC = b"bgen"
LAYOUT_2 = 2
COMPRESSION_ZLIB = 1
SAMPLE_IDS_PRESENT = 1 << 31
HEADER_LENGTH = 20
BITS_PER_PROB = 8
VCZ_INT_FILL = -2


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


def _check_ploidy_dim(genotypes):
    """Accept either pure haploid ``(V, S, 1)`` or diploid-shaped
    ``(V, S, 2)`` (with optional ``-2`` haploid padding in slot 1)."""
    if genotypes.ndim != 3 or genotypes.shape[2] not in (1, 2):
        raise ValueError(
            f"BGEN output requires call_genotype.shape[2] in (1, 2); "
            f"got shape {genotypes.shape!r}."
        )


def _normalize_genotype_ploidy(genotypes):
    """Promote a ``(V, S, 1)`` haploid array to the ``(V, S, 2)`` with
    ``-2`` in slot 1 convention the C kernel expects. Pass-through for
    ``(V, S, 2)``."""
    if genotypes.shape[2] == 1:
        v, s, _ = genotypes.shape
        out = np.empty((v, s, 2), dtype=np.int8)
        out[:, :, 0] = genotypes[:, :, 0]
        out[:, :, 1] = VCZ_INT_FILL
        return out
    return genotypes


def _flags_word(embed_samples=True):
    flags = COMPRESSION_ZLIB | (LAYOUT_2 << 2)
    if embed_samples:
        flags |= SAMPLE_IDS_PRESENT
    return flags


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


def _build_header(num_variants, num_samples, sample_id_block, embed_samples=True):
    """Build the 4-byte offset prefix + 20-byte header + sample-id block.

    When ``embed_samples`` is False the ``SAMPLE_IDS_PRESENT`` flag bit is
    cleared and the caller must pass ``sample_id_block=b""``; the resulting
    BGEN has ``offset = HEADER_LENGTH`` (the sample-id block is absent).
    """
    # offset is measured from the end of itself (i.e. from byte 4): it
    # equals the header length plus the sample-id block length.
    offset = HEADER_LENGTH + len(sample_id_block)
    out = bytearray()
    out += struct.pack("<I", offset)
    out += struct.pack("<I", HEADER_LENGTH)
    out += struct.pack("<I", num_variants)
    out += struct.pack("<I", num_samples)
    out += BGEN_MAGIC
    out += struct.pack("<I", _flags_word(embed_samples))
    out += sample_id_block
    return bytes(out)


@dataclasses.dataclass
class _ChunkBytes:
    """Per-slice byte-prepared inputs to the BGEN variant-block encoder.

    Each of the five string fields is an ``S``-dtype array; the row
    width is the chunk-wide max byte length. ``len(arr[i])`` (or
    ``bytes(arr[i])``) returns the actual UTF-8 bytes of the field
    for variant ``i`` — numpy strips trailing NULs on element access,
    which is exactly what :func:`_encode_variant_block` writes as the
    BGEN length prefix.

    For the :class:`BgenEncoder` (fixed-stride) path, the byte sums
    over all five fields are pinned to ``total_string_length`` per
    variant by the padding field (built in
    :func:`_prepare_chunk_strings`); for the :func:`write_bgen`
    variable-stride path the padding slot is just ``b"."``.

    ``genotypes`` is the slice's ``(V, S, 2)`` int8 array (normalised
    haploid → ``-2``-padded form); ``phased`` is the per-variant bool
    flag.
    """

    varid: np.ndarray
    rsid: np.ndarray
    chrom: np.ndarray
    allele1: np.ndarray
    allele2: np.ndarray
    position: np.ndarray
    genotypes: np.ndarray
    phased: np.ndarray
    mixed_phase_count: int


@dataclasses.dataclass
class _ChunkStrings:
    """Full-chunk byte-prepared variant string fields.

    Built once per chunk on the main thread (see
    :func:`_prepare_chunk_strings`); worker slices read rows out of
    these instead of running their own encode + pad. The string-prep
    step holds the GIL, so hoisting it out of the worker pool is what
    lets parallel slice encodes actually overlap.

    Each array is an ``S``-dtype 1D array with row width equal to
    the chunk-wide max byte length for that field.
    """

    varid: np.ndarray
    rsid: np.ndarray
    chrom: np.ndarray
    allele1: np.ndarray
    allele2: np.ndarray


def _to_fixed_bytes(arr, lengths):
    """Cast ``arr`` to an ``S{max_len}`` byte array, where ``max_len``
    is the maximum of the pre-computed per-element ``lengths`` array
    (clamped to at least 1 since numpy rejects ``S0``).
    """
    if lengths.size == 0:
        return arr.astype("S1")
    max_len = max(int(lengths.max()), 1)
    return arr.astype(f"S{max_len}")


def _padding_slack(chrom_len, a1_len, a2_len, variant_id_len, total_string_length):
    """Per-variant slack budget for the BGEN combined-byte-length
    layout, given the four pre-computed actual-length arrays and the
    total budget. Validates that every entry is at least ``1`` — the
    padding slot's leading ``"."`` always has to fit — and raises
    :class:`ValueError` naming the offending variant on failure.

    Callers that already need the length arrays for downstream sizing
    (``_prepare_chunk_strings``) compute them once and pass them in;
    callers that only need slack (``_bgi_rsid_column``) recompute
    locally before calling.
    """
    used = chrom_len + a1_len + a2_len + variant_id_len
    slack = total_string_length - used
    n = used.shape[0]
    if n > 0:
        min_slack = int(slack.min())
        if min_slack < 1:
            bad = int(np.argmin(slack))
            raise ValueError(
                f"variant {bad}: chrom/allele1/allele2/variant_id "
                f"byte sum is {int(used[bad])}, leaving no room "
                f"for the padding field's leading '.' under "
                f"total_string_length={total_string_length}"
            )
    return slack


def _build_padding_bytes(n, slack, pad_byte):
    """Per-variant padding-field row builder for the BGEN encoders.

    Returns an ``S{max_slack}`` byte array of length ``n``; each row
    is ``b"." + pad_byte * (slack[v] - 1)``, NUL-padded out to
    ``max_slack``. Numpy strips trailing NULs on element access so
    ``bytes(arr[i])`` recovers exactly the per-variant padding string.
    Caller must guarantee ``slack[v] >= 1`` for every row (the leading
    ``"."`` always has to fit) — :func:`_padding_slack` enforces this.
    """
    if n == 0:
        return np.zeros(0, dtype="S1")
    max_slack = max(int(slack.max()), 1)
    out = np.full((n, max_slack), pad_byte[0], dtype=np.uint8)
    col = np.arange(max_slack)
    out[col[None, :] >= slack[:, None]] = 0
    out[:, 0] = ord(".")
    return out.view(f"S{max_slack}").reshape(n)


def _prepare_chunk_strings(
    chunk,
    contig_ids,
    *,
    variant_id_field="rsid",
    total_string_length=None,
    pad_byte=b".",
):
    """Build the five per-variant string byte arrays consumed by the
    BGEN encoder paths.

    Each of the four "actual length" fields — ``chrom``, ``allele1``,
    ``allele2``, and whichever of ``varid``/``rsid`` carries the
    variant id — is converted to an ``S``-dtype byte array sized to
    the chunk-wide max byte length, using the per-element length
    array computed up front. The fifth field is the padding field,
    which absorbs per-variant slack:

    - :class:`BgenEncoder` (``total_string_length`` is an int): the
      padding field is ``b"." + pad_byte * (slack - 1)`` per variant,
      where ``slack = total_string_length - sum-of-other-four``.
      Raises :class:`ValueError` if any variant has ``slack < 1`` (the
      leading ``"."`` always has to fit).
    - :func:`write_bgen` (``total_string_length is None``): the
      padding field is the literal ``"."`` (one byte) for every
      variant.

    ``variant_id_field`` selects which BGEN slot the variant id
    occupies (``"rsid"`` or ``"varid"``); the other slot is the
    padding field. Empty ``variant_id`` values normalise to ``"."``;
    a missing ``variant_id`` field degrades every row to ``"."``.
    """
    if variant_id_field not in ("rsid", "varid"):
        raise ValueError(
            f"variant_id_field must be 'rsid' or 'varid' (got {variant_id_field!r})"
        )
    alleles = chunk["variant_allele"]
    _check_biallelic(alleles)
    n = alleles.shape[0]
    contigs = chunk["variant_contig"]
    varids = chunk.get("variant_id")

    chrom_arr = np.asarray(contig_ids)[contigs]
    a1_arr = alleles[:, 0]
    if alleles.shape[1] >= 2:
        a2_arr = np.where(alleles[:, 1] == "", ".", alleles[:, 1])
    else:
        a2_arr = np.full(n, ".")
    if varids is not None:
        variant_id_arr = np.where(varids == "", ".", varids)
    else:
        variant_id_arr = np.full(n, ".")

    # Compute per-element byte lengths once; reused for the slack
    # check (BgenEncoder path) and for sizing the S-dtype rows.
    chrom_len = np.strings.str_len(chrom_arr)
    a1_len = np.strings.str_len(a1_arr)
    a2_len = np.strings.str_len(a2_arr)
    variant_id_len = np.strings.str_len(variant_id_arr)

    if total_string_length is None:
        # write_bgen path: padding is a literal b"." per variant.
        padding_bytes = np.full(n, b".", dtype="S1")
    else:
        slack = _padding_slack(
            chrom_len, a1_len, a2_len, variant_id_len, total_string_length
        )
        padding_bytes = _build_padding_bytes(n, slack, pad_byte)

    variant_id_bytes = _to_fixed_bytes(variant_id_arr, variant_id_len)
    if variant_id_field == "rsid":
        rsid_bytes, varid_bytes = variant_id_bytes, padding_bytes
    else:
        rsid_bytes, varid_bytes = padding_bytes, variant_id_bytes

    return _ChunkStrings(
        varid=varid_bytes,
        rsid=rsid_bytes,
        chrom=_to_fixed_bytes(chrom_arr, chrom_len),
        allele1=_to_fixed_bytes(a1_arr, a1_len),
        allele2=_to_fixed_bytes(a2_arr, a2_len),
    )


def _prepare_chunk(chunk, chunk_strings, *, start, end):
    """Slice-level prep for variants ``[start:end]`` of ``chunk``.
    Takes the pre-built full-chunk :class:`_ChunkStrings` from
    :func:`_prepare_chunk_strings` and returns a :class:`_ChunkBytes`
    ready for the BGEN C kernel.

    The genotype block is no longer materialised here: the C kernel
    builds it on the fly from ``genotypes`` and ``phased``, avoiding
    the (V, ~3*num_samples) intermediate buffer that the geno-block
    builder used to allocate. :func:`write_bgen` runs the per-variant
    Python loop, so it calls :func:`_vcztools.encode_bgen_geno_blocks`
    on the slice's ``genotypes`` separately when it needs the
    materialised form.
    """
    G = chunk["call_genotype"][start:end]
    positions = chunk["variant_position"][start:end]
    phased_arr = chunk.get("call_genotype_phased")
    if phased_arr is not None:
        phased_arr = phased_arr[start:end]

    _check_ploidy_dim(G)
    G = _normalize_genotype_ploidy(G)
    n = G.shape[0]

    if phased_arr is not None:
        all_phased = phased_arr.all(axis=1)
        any_phased = phased_arr.any(axis=1)
        mixed_phase_count = int(np.sum(any_phased & ~all_phased))
        phased = np.asarray(all_phased)
    else:
        mixed_phase_count = 0
        phased = np.zeros(n, dtype=bool)

    # Sample-subset paths can deliver call_genotype as a non-contiguous
    # fancy-indexed view; the C kernel requires NPY_ARRAY_IN_ARRAY.
    G = np.ascontiguousarray(G, dtype=np.int8)

    return _ChunkBytes(
        varid=chunk_strings.varid[start:end],
        rsid=chunk_strings.rsid[start:end],
        chrom=chunk_strings.chrom[start:end],
        allele1=chunk_strings.allele1[start:end],
        allele2=chunk_strings.allele2[start:end],
        position=positions,
        genotypes=G,
        phased=phased,
        mixed_phase_count=mixed_phase_count,
    )


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

    All string fields arrive pre-encoded as ``S``-dtype rows from
    :func:`_prepare_chunk_strings`. Element access on an ``S``-dtype
    array strips trailing NULs, so ``len(bytes(arr[i]))`` is the actual
    UTF-8 byte length — exactly the length prefix that goes on the
    wire.

    ``geno_block_bytes`` is the uncompressed BGEN genotype-block bytes
    (spec layout: N/K/Pmin/Pmax header, ploidy, phased flag, B, prob
    bytes) — assembled at chunk-level by
    ``_vcztools.encode_bgen_geno_blocks`` in :func:`_prepare_chunk`.

    ``compression_level`` is forwarded verbatim to :func:`zlib.compress`;
    :class:`BgenEncoder` always passes ``0`` (stored) to keep block sizes
    deterministic.
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


def _encode_chunk_slice(chunk, chunk_strings, start, end, *, compression_level):
    """Encode variants ``[start:end]`` of ``chunk`` into one concatenated
    byte buffer. Returns ``(slice_bytes, per_variant_lengths,
    mixed_phase_count)``.

    Self-contained for worker-thread use by the parallel
    :func:`write_bgen` path. ``chunk_strings`` is built once per chunk
    on the main thread (see :func:`_prepare_chunk_strings`) and passed
    in so worker slices only do genotype-axis work, not utf-8 encoding.
    The per-variant Python loop still needs materialised geno blocks
    (BgenEncoder fuses that step into its C kernel; write_bgen doesn't),
    so the geno-block builder runs here on the slice's genotypes.
    """
    prep = _prepare_chunk(chunk, chunk_strings, start=start, end=end)
    geno_blocks, geno_block_lens = _vcztools.encode_bgen_geno_blocks(
        prep.genotypes, prep.phased
    )
    n = end - start
    out = bytearray()
    lens = []
    for j in range(n):
        block_len = int(geno_block_lens[j])
        block = _encode_variant_block(
            varid_bytes=prep.varid[j],
            rsid_bytes=prep.rsid[j],
            chrom_bytes=prep.chrom[j],
            position=int(prep.position[j]),
            allele1_bytes=prep.allele1[j],
            allele2_bytes=prep.allele2[j],
            geno_block_bytes=bytes(geno_blocks[j, :block_len]),
            compression_level=compression_level,
        )
        out.extend(block)
        lens.append(len(block))
    return bytes(out), lens, prep.mixed_phase_count


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
          = 28 + total_string_length + zlib_stored_size(geno_size)

        geno_size = 10 + (uniform_ploidy + 1) * num_samples

    where the constant 28 = 3 * 2 (string length prefixes, uint16) + 4
    (position) + 2 (K) + 2 * 4 (allele length prefixes, uint32) + 2 * 4
    (C and D length prefixes, uint32). ``uniform_ploidy`` is derived
    from ``reader.call_genotype.shape[2]`` and is either 1 (haploid;
    ``geno_size = 10 + 2 * num_samples``) or 2 (diploid;
    ``geno_size = 10 + 3 * num_samples``).

    The five BGEN string fields (varid, rsid, chrom, allele1, allele2)
    share a single ``total_string_length`` budget per variant. Four of
    them — chrom, allele1, allele2, and whichever of varid/rsid is
    selected by ``variant_id_field`` — are emitted at their actual UTF-8
    byte lengths. The fifth slot is the padding field, holding
    ``b"." + pad_byte * (slack - 1)`` where ``slack`` is whatever's left
    of ``total_string_length`` after the other four. If a variant's
    actual content sums past ``total_string_length - 1`` (i.e. the
    padding field can't even fit its leading ``"."``), encoding raises
    :class:`ValueError`. Defaults are tuned for biobank biallelic SNP
    arrays: ``total_string_length=64``, ``pad_byte=b"."``,
    ``variant_id_field="rsid"``.

    Variant scope: biallelic and uniform ploidy. ``call_genotype``
    must have ``shape[2] == 1`` (haploid) or ``shape[2] == 2``
    (diploid). Even within the diploid path, every sample must remain
    diploid — a chunk that contains the ``-2`` haploid-padding
    sentinel raises :class:`NotImplementedError` on read; use
    :func:`write_bgen` for mixed-ploidy stores. Multi-allelic input
    raises ``ValueError`` lazily as chunks are decoded.

    The encoder serves only the ``.bgen`` byte stream. The matching
    bgenix ``.bgi`` SQLite sidecar can be produced by passing
    :attr:`variant_offsets` (computed from the encoder's fixed-size
    layout ``header_size + i * bytes_per_variant``) to the module-level
    :func:`write_bgi`. The ``.sample`` sidecar is produced by
    :func:`write_sample`.

    :meth:`~vcztools.retrieval.VczReader.set_variant_filter` is not
    supported and raises ``NotImplementedError`` at construction;
    materialise the filter or use ``set_variants`` first. Unlike
    :func:`write_bgen`, the encoder is I/O-free in ``__init__``.

    ``unphased=True`` forces every variant's phased flag to ``0``,
    ignoring ``call_genotype_phased`` if present — see
    :func:`write_bgen` for the use case.
    """

    HEADER_FLAGS: ClassVar[int] = _flags_word()

    def __init__(
        self,
        reader: retrieval.VczReader,
        *,
        total_string_length: int | None = None,
        pad_byte: bytes | None = None,
        variant_id_field: str | None = None,
        embed_header_samples: bool = True,
        encode_threads: int | None = None,
        encode_block_bytes: int | None = None,
        unphased: bool = False,
    ):
        if total_string_length is None:
            total_string_length = 64
        if pad_byte is None:
            pad_byte = b"."
        if variant_id_field is None:
            variant_id_field = "rsid"

        if not isinstance(pad_byte, bytes) or len(pad_byte) != 1:
            raise ValueError(f"pad_byte must be a single byte (got {pad_byte!r})")
        if pad_byte == b"\x00":
            # numpy S-dtype strips trailing NULs on element access, so
            # a NUL pad byte would collapse the padding field down to
            # just the leading "." on the wire and the variant block
            # would be shorter than bytes_per_variant.
            raise ValueError("pad_byte must not be the NUL byte")
        if variant_id_field not in ("rsid", "varid"):
            raise ValueError(
                f"variant_id_field must be 'rsid' or 'varid' (got {variant_id_field!r})"
            )
        if total_string_length < 5:
            # Each of the five string slots needs at least one byte
            # (the leading "." in the padding field, and one byte each
            # for chrom/allele1/allele2/variant_id).
            raise ValueError(
                f"total_string_length must be >= 5 (got {total_string_length})"
            )
        if total_string_length > 0xFFFF:
            # Both variant-id slots use uint16 BGEN length prefixes;
            # the chrom slot does too. With a >0xFFFF budget the
            # padding slot could overflow its prefix.
            raise ValueError(
                f"total_string_length={total_string_length} exceeds "
                f"BGEN length-prefix width 65535"
            )

        contig_ids = reader.contig_ids
        chrom_max = int(
            np.strings.str_len(np.strings.encode(contig_ids)).max(initial=1)
        )
        if chrom_max > 0xFFFF:
            raise ValueError(
                f"longest contig is {chrom_max} bytes, exceeds BGEN "
                f"length-prefix width 65535"
            )

        self._total_string_length = total_string_length
        self._pad_byte = pad_byte
        self._variant_id_field = variant_id_field
        self._has_variant_id = "variant_id" in reader.field_names
        self._has_phased = (
            not unphased
        ) and "call_genotype_phased" in reader.field_names
        self._contig_ids = contig_ids
        self._mixed_phase_count = 0

        num_samples = int(reader.sample_ids.size)
        # Uniform per-variant block width requires uniform ploidy. The
        # source ploidy dimension drives the geno block size: haploid
        # (shape[2]==1) stores 1 ploidy + 1 prob byte per sample,
        # diploid (shape[2]==2) stores 1 ploidy + 2 prob bytes per
        # sample. Mixed-ploidy stores (any -2 sentinel under shape[2]==2)
        # break the fixed-size contract and are rejected lazily in
        # _encode_variant_range with a NotImplementedError.
        gt_info = reader.get_field_info("call_genotype")
        uniform_ploidy = gt_info.shape[2]
        if uniform_ploidy not in (1, 2):
            raise ValueError(
                f"BgenEncoder requires call_genotype.shape[2] in (1, 2); "
                f"got {gt_info.shape!r}."
            )
        self._uniform_ploidy = uniform_ploidy
        self._uniform_geno_size = 10 + (uniform_ploidy + 1) * num_samples
        bytes_per_variant = _vcztools.bgen_variant_block_size(
            num_samples=num_samples,
            uniform_ploidy=uniform_ploidy,
            total_string_length=total_string_length,
        )

        if embed_header_samples:
            sample_id_block = _build_sample_id_block(reader.sample_ids)
        else:
            sample_id_block = b""
        num_variants = int(reader.variant_counts_per_chunk().sum())
        prefix_bytes = _build_header(
            num_variants,
            num_samples,
            sample_id_block,
            embed_samples=embed_header_samples,
        )

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

    @property
    def variant_id_field(self) -> str:
        """Which BGEN id slot (``"rsid"`` or ``"varid"``) the encoder
        routes ``variant_id`` into; the other slot is the padding field."""
        return self._variant_id_field

    @property
    def total_string_length(self) -> int:
        """Combined byte budget for the five BGEN string slots —
        :func:`write_bgi` needs this to reproduce the per-variant padding
        the encoder wrote into the unused id slot."""
        return self._total_string_length

    @property
    def pad_byte(self) -> bytes:
        """Single byte used to fill the padding slot beyond its leading
        ``b"."``. Default ``b"."``, never the NUL byte."""
        return self._pad_byte

    def _encode_chunk(self, chunk: dict) -> bytes:
        # String prep is hoisted out of the worker pool: building the
        # per-variant byte arrays (and the padding-field bytes) holds
        # the GIL, so running it once per chunk on the main thread
        # (rather than once per worker slice) is what lets parallel
        # slice encodes actually overlap. Per-slice mixed_phase_count
        # partials are appended to a closure-captured list (list.append
        # is GIL-safe under CPython) and summed once _parallel_encode
        # returns.
        chunk_strings = _prepare_chunk_strings(
            chunk,
            self._contig_ids,
            variant_id_field=self._variant_id_field,
            total_string_length=self._total_string_length,
            pad_byte=self._pad_byte,
        )
        num_variants = int(chunk["call_genotype"].shape[0])
        phase_counts: list[int] = []

        def encode_range(start: int, end: int, out_view: np.ndarray) -> None:
            self._encode_variant_range(
                chunk, chunk_strings, start, end, phase_counts, out_view
            )

        output = self._parallel_encode(
            num_variants=num_variants, encode_range=encode_range
        )
        self._mixed_phase_count += sum(phase_counts)
        return output

    def _encode_variant_range(
        self,
        chunk: dict,
        chunk_strings: _ChunkStrings,
        start: int,
        end: int,
        phase_counts: list[int],
        out_view: np.ndarray,
    ) -> None:
        prep = _prepare_chunk(chunk, chunk_strings, start=start, end=end)
        phase_counts.append(prep.mixed_phase_count)
        position = np.ascontiguousarray(prep.position, dtype=np.int32)
        # The C kernel reads each S-dtype row as a (N, item_size) uint8
        # buffer and computes the actual byte length per variant via
        # an internal NUL-bound scan. Pre-viewing here keeps the C
        # binding's dtype/shape checks straightforward.
        varid_2d = prep.varid.view(np.uint8).reshape(-1, prep.varid.dtype.itemsize)
        rsid_2d = prep.rsid.view(np.uint8).reshape(-1, prep.rsid.dtype.itemsize)
        chrom_2d = prep.chrom.view(np.uint8).reshape(-1, prep.chrom.dtype.itemsize)
        allele1_2d = prep.allele1.view(np.uint8).reshape(
            -1, prep.allele1.dtype.itemsize
        )
        allele2_2d = prep.allele2.view(np.uint8).reshape(
            -1, prep.allele2.dtype.itemsize
        )
        # The kernel builds each variant's BGEN genotype block on the
        # fly from prep.genotypes + prep.phased; mixed-ploidy variants
        # surface via VCZ_ERR_BGEN_MIXED_PLOIDY (handle_library_error
        # converts to NotImplementedError pointing at write_bgen).
        _vcztools.encode_bgen_chunk_slice_level0(
            varid_2d,
            rsid_2d,
            chrom_2d,
            allele1_2d,
            allele2_2d,
            position,
            prep.genotypes,
            prep.phased,
            out_view,
            self._uniform_ploidy,
            self._total_string_length,
        )

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
        :func:`write_bgi`."""
        return self.prefix_size + self.bytes_per_variant * np.arange(
            self.num_variants + 1, dtype=np.int64
        )


def write_sample(reader, output):
    """Write the Oxford ``.sample`` text for ``reader`` to ``output``.

    ``output`` is a filesystem path (``str`` / ``pathlib.Path``) or a
    writable text file-like object. The minimal sample file: two header
    rows (``ID_1 ID_2 missing`` and ``0 0 0``), then one row per sample
    with ``ID_1`` and ``ID_2`` both set to the sample name (matches the
    ``--double-id`` style) and the ``missing`` column set to ``0``.
    Whitespace in sample IDs is rejected (the file format is
    whitespace-separated).
    """
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
    with utils.open_file_like(output, mode="w") as f:
        f.write("\n".join(lines) + "\n")


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


def _bgi_rsid_column(
    chrom_arr: np.ndarray,
    a1_arr: np.ndarray,
    a2_arr: np.ndarray,
    variant_id_arr: np.ndarray,
    *,
    variant_id_field: str,
    total_string_length: int | None,
    pad_byte: bytes,
) -> np.ndarray:
    """Build the .bgi ``rsid`` column to match the bytes the encoder
    wrote into the BGEN ``rsid`` slot.

    When ``variant_id_field == "rsid"``, the slot carries
    ``variant_id`` directly. When ``variant_id_field == "varid"``, the
    slot is the padding field; we reuse :func:`_padding_slack` and
    :func:`_build_padding_bytes` (the same path that prepares the
    encoder's on-disk bytes) so the .bgi column matches what bgenix
    would record when reindexing the same BGEN file, then decode each
    row's actual bytes (NUL-stripped on element access) to a Python
    ``str`` for SQLite.
    """
    if variant_id_field == "rsid":
        return variant_id_arr
    n = chrom_arr.shape[0]
    if total_string_length is None:
        # write_bgen path: padding is a literal "." per variant.
        return np.full(n, ".")
    slack = _padding_slack(
        np.strings.str_len(chrom_arr),
        np.strings.str_len(a1_arr),
        np.strings.str_len(a2_arr),
        np.strings.str_len(variant_id_arr),
        total_string_length,
    )
    padding_bytes = _build_padding_bytes(n, slack, pad_byte)
    return np.array([row.decode("ascii") for row in padding_bytes], dtype=object)


def write_bgi(
    reader,
    output,
    variant_offsets,
    *,
    variant_id_field: str = "rsid",
    total_string_length: int | None = None,
    pad_byte: bytes = b".",
):
    """Write the bgenix ``.bgen.bgi`` SQLite sidecar for ``reader``.

    ``output`` is a filesystem path (``str`` / ``pathlib.Path``); the
    SQLite database is created at that location (file-like objects are
    not supported — ``sqlite3.connect`` needs a real path). If the file
    already exists, it is unlinked first so the schema can be recreated
    without primary-key conflicts.

    ``variant_offsets`` is an integer array of length
    ``num_variants + 1`` giving the byte boundaries of each variant
    block: variant ``i`` occupies
    ``[variant_offsets[i], variant_offsets[i+1])``. Use
    :attr:`BgenEncoder.variant_offsets` for the fixed-size encoder
    path, or the cumulative sum of per-variant block sizes (plus the
    BGEN prefix length) for the variable-size :func:`write_bgen`
    path.

    ``variant_id_field`` mirrors :func:`write_bgen` /
    :class:`BgenEncoder`: when ``"rsid"`` (default), ``variant_id``
    populates the .bgi ``rsid`` column directly; when ``"varid"``, the
    BGEN ``rsid`` slot was the padding field at encode time, so the
    .bgi ``rsid`` column carries the same padding bytes the BGEN file
    holds. ``total_string_length`` and ``pad_byte`` are the same
    parameters as on :class:`BgenEncoder` and reconstruct the
    per-variant padding strings: ``None`` matches :func:`write_bgen`'s
    single-byte ``"."`` padding, an integer matches the encoder's
    ``"." + pad_byte * (slack - 1)`` pattern.

    Variant-metadata columns (chromosome, position, rsid, alleles)
    are read via the reader's standard ``variant_chunks`` API and
    assembled as numpy arrays; per-row tuples are streamed into
    SQLite via a generator so peak memory stays at the column
    arrays rather than an N-row tuple list. Variant scope is
    biallelic only; multi-allelic raises ``ValueError`` lazily on
    the chunk containing the offending row.
    """
    if variant_id_field not in ("rsid", "varid"):
        raise ValueError(
            f"variant_id_field must be 'rsid' or 'varid' (got {variant_id_field!r})"
        )

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

    bgi_path = pathlib.Path(output)
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
    variant_id_parts: list[np.ndarray] = []

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
            variant_id_parts.append(np.asarray(chunk["variant_id"]))
        else:
            variant_id_parts.append(np.full(n, "."))
    read_seconds = time.perf_counter() - read_start

    assemble_start = time.perf_counter()
    rows_iter = None
    if num_variants > 0:
        chrom_arr = np.concatenate(chrom_parts)
        pos_arr = np.concatenate(pos_parts)
        a1_arr = np.concatenate(a1_parts)
        a2_arr = np.concatenate(a2_parts)
        variant_id_arr = np.concatenate(variant_id_parts)
        # Match the .bim/.bgi monomorphic / missing-rsid convention used
        # by write_bgen and BgenEncoder.
        variant_id_arr = np.where(variant_id_arr == "", ".", variant_id_arr)
        a2_arr = np.where(a2_arr == "", ".", a2_arr)
        if chrom_arr.shape[0] != num_variants:
            raise ValueError(
                f"reader yielded {chrom_arr.shape[0]} variants but "
                f"variant_offsets implies {num_variants}"
            )
        rsid_arr = _bgi_rsid_column(
            chrom_arr,
            a1_arr,
            a2_arr,
            variant_id_arr,
            variant_id_field=variant_id_field,
            total_string_length=total_string_length,
            pad_byte=pad_byte,
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
        f"write_bgi: wrote {num_variants} variants ({mib:.2f} MiB) "
        f"to {bgi_path} in {elapsed:.2f}s ({rate:.1f} MiB/s); "
        f"read={read_seconds:.2f}s, assemble={assemble_seconds:.2f}s, "
        f"insert={insert_seconds:.2f}s"
    )


def write_bgen(
    reader,
    output,
    *,
    sample_path=None,
    bgi_path=None,
    embed_header_samples: bool | None = None,
    compression_level: int | None = None,
    encode_threads: int | None = None,
    unphased: bool = False,
    variant_id_field: str | None = None,
):
    """Write an Oxford BGEN payload for ``reader`` to ``output``.

    ``output`` is either a filesystem path (``str`` / ``pathlib.Path``)
    or a writable binary file-like object (anything with a ``.write``
    method, including ``sys.stdout.buffer``); paths are opened via
    :func:`vcztools.utils.open_file_like` in ``"wb"`` mode. The function
    never seeks the output.

    ``sample_path`` and ``bgi_path`` request the optional Oxford
    ``.sample`` text sidecar and the bgenix ``.bgen.bgi`` SQLite sidecar
    at the given filesystem paths; ``None`` (default) skips that
    sidecar. The ``.sample`` is written first; the ``.bgi`` is written
    last using the variant-block byte offsets accumulated while
    streaming the BGEN payload.

    If a variant filter is configured on the reader, it is resolved
    in place via :meth:`~vcztools.retrieval.VczReader.materialise_variant_filter`
    so that variant-block ordering and ``.bgi`` entries align with the
    BGEN payload.

    Variant scope: biallelic only (multi-allelic raises ``ValueError``).
    Genotype source: hard calls from ``call_genotype`` encoded as 1.0
    probability on the called genotype (8-bit precision round-trips
    exactly). Phase: per-variant from ``call_genotype_phased`` if the
    field exists in the store; otherwise unphased.

    ``embed_header_samples`` controls whether the BGEN header carries
    sample IDs. When False, the ``SAMPLE_IDS_PRESENT`` flag is cleared
    and the sample-id block is omitted. Most downstream tools require
    sample IDs from either the BGEN header or a ``.sample`` sidecar; if
    neither is produced the function logs a warning.

    ``compression_level`` is forwarded to :func:`zlib.compress` for each
    variant's genotype probability block; accepts ``-1..9`` (``-1`` =
    zlib default ≈ level 6; ``0`` = stored, still framed as zlib;
    ``9`` = maximum). The default is ``1`` — fast compression. Hard-call
    BGEN payloads are short, low-entropy byte runs (mostly 1.0/0.0 in
    8-bit form, repeated across samples), so the marginal compression
    above level 1 is small relative to the CPU cost: level 6 (zlib
    default) typically shrinks the file by ~10-30% but spends several
    times more CPU. Since the BGEN flag word always advertises
    ``COMPRESSION_ZLIB`` regardless of level, every reader handles the
    output. Pass ``--compression-level 9`` (or ``compression_level=9``)
    when archival size matters more than encode throughput.

    ``encode_threads`` sizes the worker pool that runs per-slice
    :func:`_prepare_chunk` + per-variant ``_encode_variant_block`` for
    each chunk. Slice bytes are written back to the output in variant
    order on the main thread so byte layout and ``.bgi`` offsets stay
    deterministic. ``None`` selects the default (4).

    ``unphased=True`` forces every variant's phased flag to ``0``,
    ignoring ``call_genotype_phased`` if present. Use this when the
    downstream tool only accepts unphased BGEN (e.g. qctool's
    ``-snp-stats``, whose ``ToGP`` setter rejects per-haplotype-per-
    allele probabilities).

    ``variant_id_field`` chooses which BGEN slot — ``"rsid"`` (default)
    or ``"varid"`` — carries the zarr ``variant_id``. The other slot
    is written as the literal ``"."`` for every variant.
    """
    if encode_threads is None:
        encode_threads = 4
    if encode_threads < 1:
        raise ValueError(f"encode_threads must be >= 1 (got {encode_threads})")
    if embed_header_samples is None:
        embed_header_samples = True
    if compression_level is None:
        compression_level = 1
    if variant_id_field is None:
        variant_id_field = "rsid"

    reader.materialise_variant_filter()

    if not embed_header_samples and sample_path is None:
        logger.warning(
            "write_bgen: embed_header_samples=False and sample_path=None "
            "leave sample IDs nowhere in the output; downstream tools "
            "(REGENIE, SAIGE, bgen-reader) will not be able to associate "
            "genotypes with sample IDs."
        )

    if sample_path is not None:
        write_sample(reader, sample_path)

    if isinstance(output, (str, pathlib.Path)):
        display_name = str(output)
    else:
        display_name = getattr(output, "name", "<stream>")

    start = time.perf_counter()
    encode_seconds = 0.0
    write_seconds = 0.0
    bytes_written = 0

    num_variants = int(reader.variant_counts_per_chunk().sum())
    sample_ids = reader.sample_ids
    num_samples = int(sample_ids.size)

    contig_id = reader.contig_ids
    has_variant_id = "variant_id" in reader.field_names
    has_phased = (not unphased) and "call_genotype_phased" in reader.field_names

    fields = ["call_genotype", "variant_allele", "variant_contig", "variant_position"]
    if has_variant_id:
        fields.append("variant_id")
    if has_phased:
        fields.append("call_genotype_phased")

    if embed_header_samples:
        sample_id_block = _build_sample_id_block(sample_ids)
    else:
        sample_id_block = b""
    header_bytes = _build_header(
        num_variants,
        num_samples,
        sample_id_block,
        embed_samples=embed_header_samples,
    )

    variant_offsets = np.empty(num_variants + 1, dtype=np.int64)
    variant_offsets[0] = len(header_bytes)
    mixed_phase_count = 0
    idx = 0

    with (
        utils.open_file_like(output, mode="wb") as bgen_stream,
        cf.ThreadPoolExecutor(
            max_workers=encode_threads,
            thread_name_prefix="write-bgen-encode",
        ) as executor,
    ):
        bgen_stream.write(header_bytes)
        bytes_written += len(header_bytes)

        for chunk in reader.variant_chunks(fields=fields):
            n = int(chunk["call_genotype"].shape[0])
            # Dispatch slice encodes in submit order; collect in submit
            # order so the file byte layout (and variant_offsets) stay
            # deterministic regardless of which worker finishes first.
            # String prep is hoisted out of the worker pool — it holds
            # the GIL, so running it once per chunk lets the workers
            # actually overlap.
            chunk_strings = _prepare_chunk_strings(
                chunk, contig_id, variant_id_field=variant_id_field
            )
            slice_variants = max(1, (n + encode_threads - 1) // encode_threads)
            slice_ranges = [
                (s, min(s + slice_variants, n)) for s in range(0, n, slice_variants)
            ]
            t_enc = time.perf_counter()
            futures = [
                executor.submit(
                    _encode_chunk_slice,
                    chunk,
                    chunk_strings,
                    s,
                    e,
                    compression_level=compression_level,
                )
                for (s, e) in slice_ranges
            ]
            for fut in futures:
                slice_bytes, lens, slice_mpc = fut.result()
                t_after_enc = time.perf_counter()
                encode_seconds += t_after_enc - t_enc
                bgen_stream.write(slice_bytes)
                write_seconds += time.perf_counter() - t_after_enc
                t_enc = time.perf_counter()

                mixed_phase_count += slice_mpc
                slice_lens = np.asarray(lens, dtype=np.int64)
                n_slice = slice_lens.size
                variant_offsets[idx + 1 : idx + 1 + n_slice] = variant_offsets[
                    idx
                ] + np.cumsum(slice_lens)
                idx += n_slice
                bytes_written += len(slice_bytes)

    elapsed = time.perf_counter() - start
    mib = bytes_written / (1024 * 1024)
    rate = mib / elapsed if elapsed > 0 else 0.0
    logger.info(
        f"write_bgen: wrote {mib:.1f} MiB to {display_name} in "
        f"{elapsed:.2f}s ({rate:.1f} MiB/s); "
        f"compression_level={compression_level}, encode_threads={encode_threads}; "
        f"encode={encode_seconds:.2f}s, write={write_seconds:.2f}s"
    )
    if mixed_phase_count > 0:
        logger.warning(
            f"write_bgen: {mixed_phase_count} variant(s) had mixed phase "
            "across samples; emitted as unphased (BGEN has one phase flag "
            "per variant)."
        )

    if bgi_path is not None:
        write_bgi(reader, bgi_path, variant_offsets, variant_id_field=variant_id_field)
