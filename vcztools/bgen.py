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


def _encode_genotype_block(genotypes, phased):
    """Encode one variant's uncompressed genotype probability block.

    ``genotypes`` is the per-variant slice of ``call_genotype`` with shape
    ``(num_samples, 2)`` and dtype int8. Negative entries (``-1`` or
    ``-2``) mark missing alleles; a sample is treated as missing iff any
    of its alleles is negative. ``phased`` is a bool.

    Layout 2 / 8-bit / biallelic / diploid:

        uint32 N            n_samples
        uint16 K = 2        n_alleles
        uint8  P_min = 2
        uint8  P_max = 2
        N bytes             ploidy/missing per sample
        uint8  phased
        uint8  B = 8        bits per probability
        2*N bytes           per-sample probability bytes
    """
    a = genotypes[:, 0]
    b = genotypes[:, 1]
    n = a.shape[0]
    missing = (a < 0) | (b < 0)

    if phased:
        # Per-haplotype P(allele 0) at 8-bit precision: 0xFF if the
        # haplotype carries the reference allele, 0x00 if it carries
        # the alternate.
        B0 = np.where(a == 0, 0xFF, 0x00).astype(np.uint8)
        B1 = np.where(b == 0, 0xFF, 0x00).astype(np.uint8)
    else:
        # Unphased biallelic diploid stores P(00), P(01); P(11) is
        # implicit. In colex order over multisets of size 2 from
        # {0, 1}: (0,0), (0,1), (1,1).
        homref = (a == 0) & (b == 0)
        het = ((a == 0) & (b == 1)) | ((a == 1) & (b == 0))
        B0 = np.where(homref, 0xFF, 0x00).astype(np.uint8)
        B1 = np.where(het, 0xFF, 0x00).astype(np.uint8)

    # Spec says missing samples' probability bytes should be ignored;
    # zero them to keep the bytes deterministic for testing / hashing.
    B0[missing] = 0
    B1[missing] = 0

    ploidy_bytes = np.full(n, _PLOIDY_DIPLOID, dtype=np.uint8)
    ploidy_bytes[missing] = _PLOIDY_MISSING

    prob_bytes = np.empty(2 * n, dtype=np.uint8)
    prob_bytes[0::2] = B0
    prob_bytes[1::2] = B1

    out = bytearray()
    out += struct.pack("<IHBB", n, 2, 2, 2)
    out += ploidy_bytes.tobytes()
    out += struct.pack("<BB", 1 if phased else 0, BITS_PER_PROB)
    out += prob_bytes.tobytes()
    return bytes(out)


def _encode_variant_block(
    *,
    varid,
    rsid,
    chrom,
    position,
    allele1,
    allele2,
    num_samples,
    genotypes,
    phased,
    compression_level,
):
    """Encode one full BGEN variant data block (identifying data +
    compressed genotype block).

    The compressed block is preceded by ``C`` (total compressed length
    including the 4-byte ``D`` field) and ``D`` (uncompressed length).
    """
    geno_block = _encode_genotype_block(genotypes, phased)
    compressed = zlib.compress(geno_block, compression_level)

    out = bytearray()
    varid_b = str(varid).encode("utf-8")
    rsid_b = str(rsid).encode("utf-8")
    chrom_b = str(chrom).encode("utf-8")
    a1_b = str(allele1).encode("utf-8")
    a2_b = str(allele2).encode("utf-8")

    out += struct.pack("<H", len(varid_b))
    out += varid_b
    out += struct.pack("<H", len(rsid_b))
    out += rsid_b
    out += struct.pack("<H", len(chrom_b))
    out += chrom_b
    out += struct.pack("<I", int(position))
    out += struct.pack("<H", 2)  # K = 2 alleles
    out += struct.pack("<I", len(a1_b))
    out += a1_b
    out += struct.pack("<I", len(a2_b))
    out += a2_b
    # C includes its own 4-byte size of D plus the compressed payload.
    C = 4 + len(compressed)
    D = len(geno_block)
    out += struct.pack("<II", C, D)
    out += compressed
    return bytes(out)


def _pad_to_fixed_length(value, max_len, *, field_name):
    """UTF-8 encode ``value`` and right-pad with NUL to exactly ``max_len``
    bytes. Raise ``ValueError`` if the encoded form exceeds ``max_len``.

    bgen-reader strips trailing NULs from variable string fields, so
    a NUL-padded value round-trips as the original UTF-8 string.
    """
    encoded = str(value).encode("utf-8")
    if len(encoded) > max_len:
        raise ValueError(
            f"{field_name} {value!r} encodes to {len(encoded)} UTF-8 bytes, "
            f"exceeds configured max of {max_len}"
        )
    return encoded + b"\x00" * (max_len - len(encoded))


def _encode_variant_block_fixed_size(
    *,
    varid_padded,
    rsid_padded,
    chrom_padded,
    position,
    allele1_padded,
    allele2_padded,
    varid_max,
    rsid_max,
    chrom_max,
    allele_max,
    genotypes,
    phased,
):
    """Emit one fixed-size BGEN variant data block.

    Layout (length prefixes are written as the configured *_max values;
    string payloads are NUL-padded to those widths):

        uint16 varid_max ; varid_max bytes (UTF-8, NUL-padded)
        uint16 rsid_max  ; rsid_max  bytes (UTF-8, NUL-padded)
        uint16 chrom_max ; chrom_max bytes (UTF-8, NUL-padded)
        uint32 position
        uint16 K = 2
        uint32 allele_max ; allele_max bytes (UTF-8, NUL-padded)
        uint32 allele_max ; allele_max bytes (UTF-8, NUL-padded)
        uint32 C = 4 + len(compressed)
        uint32 D = len(uncompressed_geno_block)
        compressed: zlib level 0 (stored) of the uncompressed geno block

    Compression flag is ``ZLIB`` but level is always ``0`` (stored, no
    DEFLATE). Stored zlib output for an input of fixed length is itself
    of fixed length, so every variant block is the same number of bytes.
    The published BGEN spec also supports a ``compression == 0`` mode
    that would save the 11+ bytes of zlib framing per variant, but the
    reference cbgen/limix-bgen library has an inconsistency: its
    ``bgen_variant_next`` always reads a uint32 length prefix, while
    its layout-2 read-header for ``compression == 0`` does not consume
    one. Using ``ZLIB`` with level 0 sidesteps this.
    """
    geno_block = _encode_genotype_block(genotypes, phased)
    compressed = zlib.compress(geno_block, 0)
    out = bytearray()
    out += struct.pack("<H", varid_max)
    out += varid_padded
    out += struct.pack("<H", rsid_max)
    out += rsid_padded
    out += struct.pack("<H", chrom_max)
    out += chrom_padded
    out += struct.pack("<I", int(position))
    out += struct.pack("<H", 2)
    out += struct.pack("<I", allele_max)
    out += allele1_padded
    out += struct.pack("<I", allele_max)
    out += allele2_padded
    C = 4 + len(compressed)
    D = len(geno_block)
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

    The encoder serves only the ``.bgen`` byte stream. The ``.sample``
    sidecar is produced by :func:`generate_sample`; a ``.bgi`` index, if
    desired, can be built from the fixed-size offsets
    (``header_size + i * bytes_per_variant``) without iterating the
    encoder.

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
        _check_biallelic(chunk["variant_allele"])
        _check_diploid(chunk["call_genotype"])
        # Derive per-variant string / position / phase inputs from the
        # chunk dict. Same conventions as _stream_bgen_to_file (rsid="."
        # for missing, varid = rsid, "." for monomorphic alt).
        G = chunk["call_genotype"]
        alleles = chunk["variant_allele"]
        positions = chunk["variant_position"]
        contigs = chunk["variant_contig"]
        varids = chunk.get("variant_id")
        phased_arr = chunk.get("call_genotype_phased")

        num_variants = G.shape[0]
        # Per-variant string padding runs on the calling thread before
        # any pool fan-out so _pad_to_fixed_length overflow raises
        # deterministically rather than from inside a worker.
        inputs = []
        for j in range(num_variants):
            if phased_arr is not None:
                row_phased = phased_arr[j]
                all_phased = bool(row_phased.all())
                any_phased = bool(row_phased.any())
                phased = all_phased
                if any_phased and not all_phased:
                    self._mixed_phase_count += 1
            else:
                phased = False

            a1 = str(alleles[j, 0])
            a2 = str(alleles[j, 1]) if alleles.shape[1] >= 2 else ""
            if a2 == "":
                a2 = "."
            chrom = str(self._contig_ids[int(contigs[j])])
            position = int(positions[j])
            rsid = str(varids[j]) if varids is not None else "."
            if rsid == "":
                rsid = "."
            varid = rsid

            inputs.append(
                (
                    _pad_to_fixed_length(varid, self._varid_max, field_name="varid"),
                    _pad_to_fixed_length(rsid, self._rsid_max, field_name="rsid"),
                    _pad_to_fixed_length(chrom, self._chrom_max, field_name="chrom"),
                    position,
                    _pad_to_fixed_length(a1, self._allele_max, field_name="allele1"),
                    _pad_to_fixed_length(a2, self._allele_max, field_name="allele2"),
                    phased,
                )
            )

        def encode_range(start: int, end: int) -> bytes:
            return self._encode_variant_range(G, inputs, start, end)

        return self._parallel_encode(
            num_variants=num_variants,
            encode_range=encode_range,
        )

    def _encode_variant_range(self, G, inputs, start, end):
        bpv = self._bytes_per_variant
        out = bytearray((end - start) * bpv)
        for j in range(start, end):
            varid_p, rsid_p, chrom_p, position, a1_p, a2_p, phased = inputs[j]
            block = _encode_variant_block_fixed_size(
                varid_padded=varid_p,
                rsid_padded=rsid_p,
                chrom_padded=chrom_p,
                position=position,
                allele1_padded=a1_p,
                allele2_padded=a2_p,
                varid_max=self._varid_max,
                rsid_max=self._rsid_max,
                chrom_max=self._chrom_max,
                allele_max=self._allele_max,
                genotypes=G[j],
                phased=phased,
            )
            local = j - start
            out[local * bpv : (local + 1) * bpv] = block
        return bytes(out)

    def _close_hook(self) -> None:
        if self._mixed_phase_count > 0:
            self._logger.warning(
                f"BgenEncoder: {self._mixed_phase_count} variant(s) had "
                "mixed phase across samples; emitted as unphased (BGEN "
                "has one phase flag per variant)."
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

CREATE TABLE Metadata (
  filename TEXT NOT NULL,
  file_size INTEGER NOT NULL,
  last_write_time INTEGER NOT NULL,
  first_1000_bytes BLOB NOT NULL,
  index_creation_time INTEGER NOT NULL
);
"""


@dataclasses.dataclass
class _BgiEntry:
    chromosome: str
    position: int
    rsid: str
    number_of_alleles: int
    allele1: str
    allele2: str
    file_start_position: int
    size_in_bytes: int


def _write_bgi_index(bgi_path, bgen_path, entries):
    """Write the bgenix SQLite index. Schema documented at
    https://enkre.net/cgi-bin/code/bgen/wiki/The bgenix index file format."""
    bgi_path = pathlib.Path(bgi_path)
    if bgi_path.exists():
        bgi_path.unlink()
    conn = sqlite3.connect(str(bgi_path))
    try:
        conn.executescript(_BGI_SCHEMA)
        conn.executemany(
            "INSERT INTO Variant (chromosome, position, rsid, number_of_alleles, "
            "allele1, allele2, file_start_position, size_in_bytes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    e.chromosome,
                    e.position,
                    e.rsid,
                    e.number_of_alleles,
                    e.allele1,
                    e.allele2,
                    e.file_start_position,
                    e.size_in_bytes,
                )
                for e in entries
            ],
        )
        bgen_stat = pathlib.Path(bgen_path).stat()
        with open(bgen_path, "rb") as f:
            first_bytes = f.read(1000)
        conn.execute(
            "INSERT INTO Metadata (filename, file_size, last_write_time, "
            "first_1000_bytes, index_creation_time) VALUES (?, ?, ?, ?, ?)",
            (
                pathlib.Path(bgen_path).name,
                bgen_stat.st_size,
                int(bgen_stat.st_mtime),
                first_bytes,
                int(time.time()),
            ),
        )
        conn.commit()
    finally:
        conn.close()


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

    entries = _stream_bgen_to_file(
        reader, bgen_path, compression_level=compression_level
    )
    _write_bgi_index(bgi_path, bgen_path, entries)


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

    entries: list[_BgiEntry] = []
    mixed_phase_count = 0

    with open(bgen_path, "wb") as bgen_file:
        bgen_file.write(header_bytes)
        bytes_written += len(header_bytes)
        file_offset = len(header_bytes)

        for chunk in reader.variant_chunks(fields=fields):
            G = chunk["call_genotype"]
            alleles = chunk["variant_allele"]
            positions = chunk["variant_position"]
            contigs = chunk["variant_contig"]
            varids = chunk.get("variant_id")
            phased_arr = chunk.get("call_genotype_phased")

            _check_biallelic(alleles)
            _check_diploid(G)

            for j in range(G.shape[0]):
                t0 = time.perf_counter()
                if phased_arr is not None:
                    row_phased = phased_arr[j]
                    all_phased = bool(row_phased.all())
                    any_phased = bool(row_phased.any())
                    phased = all_phased
                    if any_phased and not all_phased:
                        mixed_phase_count += 1
                else:
                    phased = False

                a1 = str(alleles[j, 0])
                a2 = str(alleles[j, 1]) if alleles.shape[1] >= 2 else ""
                # Match generate_bim's monomorphic convention: emit "."
                # when the alt slot is empty.
                if a2 == "":
                    a2 = "."
                chrom = str(contig_id[int(contigs[j])])
                position = int(positions[j])
                rsid = str(varids[j]) if varids is not None else "."
                if rsid == "":
                    rsid = "."
                varid = rsid

                block = _encode_variant_block(
                    varid=varid,
                    rsid=rsid,
                    chrom=chrom,
                    position=position,
                    allele1=a1,
                    allele2=a2,
                    num_samples=num_samples,
                    genotypes=G[j],
                    phased=phased,
                    compression_level=compression_level,
                )
                t1 = time.perf_counter()
                bgen_file.write(block)
                t2 = time.perf_counter()
                encode_seconds += t1 - t0
                write_seconds += t2 - t1

                entries.append(
                    _BgiEntry(
                        chromosome=chrom,
                        position=position,
                        rsid=rsid,
                        number_of_alleles=2,
                        allele1=a1,
                        allele2=a2,
                        file_start_position=file_offset,
                        size_in_bytes=len(block),
                    )
                )
                file_offset += len(block)
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
    return entries
