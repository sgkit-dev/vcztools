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
import threading
from collections.abc import Iterator
from typing import ClassVar

import numpy as np
import pandas as pd

from vcztools import _vcztools, retrieval, utils
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


def compute_a12(alleles):
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

    A1/A2 are derived from ``variant_allele`` shape; observed
    genotypes in any sample subset do not influence the labelling.
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
    # surviving (≤2-allele) variant. A store-wide max of 1 (only
    # possible for empty stores or stores written with monomorphic-only
    # variants) means every row is monomorphic.
    if alleles.shape[1] < 2:
        a12[:, 0] = -1
    else:
        a12[alleles[:, 1] == "", 0] = -1
    return a12


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

    def _write_genotypes(self):
        ci = self.reader.variant_chunks(fields=["call_genotype", "variant_allele"])
        # a12 is small (8*num_variants bytes per column) and only
        # materialised for variants surviving the reader's filter, so
        # collecting per-chunk arrays and concatenating is cheap and
        # robust to partial-chunk yields under variant filtering.
        a12_per_chunk = []
        with open(self.bed_path, "wb") as bed_file:
            bed_file.write(BED_MAGIC)

            for chunk in ci:
                G = chunk["call_genotype"]
                if G.ndim != 3 or G.shape[2] != 2:
                    raise ValueError(
                        "Only diploid genotypes are supported "
                        f"(call_genotype has shape {G.shape})"
                    )
                a12 = compute_a12(chunk["variant_allele"])
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


class PlinkStreamingSource:
    """Read-only, streaming view of a VCZ store as a PLINK 1 binary
    fileset (``.bed`` / ``.bim`` / ``.fam``).

    Composes the existing PLINK 1 primitives (``compute_a12``,
    ``encode_genotypes``, ``generate_bim``, ``generate_fam``) with
    ``vcztools.retrieval.VczReader`` chunk iteration. No materialised
    on-disk fileset; the caller streams or randomly reads bytes that
    are byte-identical to what ``write_plink`` would produce for the
    same store.

    Designed for FUSE / range-HTTP / preview consumers: ``.bim`` and
    ``.fam`` are computed eagerly at construction (small enough to keep
    in memory); ``.bed`` bytes are produced on demand from
    :meth:`stream_bed`, :meth:`read_bed`, :meth:`read_tail`, or
    :meth:`read_variants`.

    Construction reads ``variant_allele`` once and traverses the
    variant-axis fields needed for ``.bim``/``.fam``; no genotype IO.

    Each method that does a ranged genotype read constructs a fresh
    ``VczReader`` (``set_variants`` is one-shot per reader). This
    is concurrency-safe: multiple in-flight ``stream_bed`` /
    ``read_variants`` calls each own their reader and read the
    immutable cached ``a12``, ``.bim`` and ``.fam`` buffers without
    locking. Only :meth:`close` mutates state, guarded by an internal
    lock for idempotency.

    Parameters
    ----------
    root
        An already-opened :class:`zarr.Group` for a VCZ store. Use
        :func:`vcztools.open_zarr` to open a path with the desired
        backend before constructing the source.
    readahead_workers
        Forwarded to every ``VczReader`` this source builds.
    readahead_bytes
        Forwarded to every ``VczReader`` this source builds.
    """

    BED_MAGIC: ClassVar[bytes] = BED_MAGIC
    DEFAULT_STREAM_CHUNK: ClassVar[int] = 1 << 20
    SPARSE_VARIANT_THRESHOLD: ClassVar[float] = 0.01

    def __init__(
        self,
        root,
        *,
        readahead_workers: int | None = None,
        readahead_bytes: int | None = None,
    ):
        self._root = root
        self._readahead_workers = readahead_workers
        self._readahead_bytes = readahead_bytes
        self._closed = False
        self._lock = threading.Lock()

        self._reader = retrieval.VczReader(
            root,
            readahead_workers=readahead_workers,
            readahead_bytes=readahead_bytes,
        )
        # Snapshot static metadata so post-close diagnostics still
        # reach a consistent state (the reader itself is dropped on
        # close()).
        self._num_variants = self._reader.num_variants
        # BED columns track the resolved (non-null) sample axis, which
        # also drives FAM row count — keeping them in sync is what
        # PLINK consumers expect.
        self._num_samples = int(self._reader.sample_ids.size)
        self._variants_chunk_size = self._reader.variants_chunk_size
        self._bytes_per_variant = (self._num_samples + 3) // 4
        self._bed_size = 3 + self._num_variants * self._bytes_per_variant

        # variant_allele is a small variant-axis array (n_var x
        # max_alleles, short string dtype). One-shot read at init keeps
        # downstream a12 lookups O(1) and memory-resident.
        variant_allele = self._reader.root["variant_allele"][:]
        self._a12 = compute_a12(variant_allele)

        self._fam_bytes = generate_fam(self._reader).encode("utf-8")
        self._bim_bytes = generate_bim(self._reader, self._a12).encode("utf-8")

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("source closed")

    @property
    def num_variants(self) -> int:
        self._check_open()
        return self._num_variants

    @property
    def num_samples(self) -> int:
        """Samples in the BED column axis (post null-sample filter)."""
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

    @property
    def bim_size(self) -> int:
        self._check_open()
        return len(self._bim_bytes)

    @property
    def fam_size(self) -> int:
        self._check_open()
        return len(self._fam_bytes)

    @property
    def bim_bytes(self) -> bytes:
        self._check_open()
        return self._bim_bytes

    @property
    def fam_bytes(self) -> bytes:
        self._check_open()
        return self._fam_bytes

    def _build_contiguous_plan(self, start: int, stop: int) -> list:
        """List of :class:`ChunkRead` covering global variants ``[start,
        stop)``. Selection is ``None`` (raw block) for fully-covered
        chunks and ``slice(local_start, local_end)`` for partial covers.
        """
        chunk_size = self._variants_chunk_size
        n_var = self._num_variants
        plan = []
        first = start // chunk_size
        last_excl = (stop - 1) // chunk_size + 1
        for ci in range(first, last_excl):
            chunk_global = ci * chunk_size
            local_start = max(0, start - chunk_global)
            local_end = min(chunk_size, stop - chunk_global)
            actual_chunk_end = min(chunk_size, n_var - chunk_global)
            if local_start == 0 and local_end == actual_chunk_end:
                selection = None
            else:
                selection = slice(local_start, local_end)
            plan.append(utils.ChunkRead(index=ci, selection=selection))
        return plan

    def _iterate_bed(self, plan, a12_run) -> Iterator[bytes]:
        """Build a fresh reader, install ``plan``, and yield encoded
        BED row bytes per chunk. ``a12_run`` is the a12 array for the
        whole run; we walk it with an offset that mirrors the per-chunk
        variant count.
        """
        reader = retrieval.VczReader(
            self._root,
            readahead_workers=self._readahead_workers,
            readahead_bytes=self._readahead_bytes,
        )
        reader.set_variants(plan)
        offset = 0
        for chunk in reader.variant_chunks(fields=["call_genotype"]):
            G = chunk["call_genotype"]
            n = G.shape[0]
            a12_chunk = a12_run[offset : offset + n]
            offset += n
            yield encode_genotypes(G, a12_chunk)

    def read_variants(
        self,
        indexes,
        *,
        strategy: str = "auto",
    ) -> bytes:
        """Return the encoded BED rows for the requested variants
        (concatenated, no magic header).

        ``indexes`` is either a ``slice`` (``step`` must be ``None`` or
        ``1``) or a sorted 1-D integer ndarray. Out-of-range ndarray
        indexes raise ``IndexError``; unsorted raises ``ValueError``;
        duplicates are kept (each emitted once per occurrence).

        ``strategy`` is one of:

        - ``"contiguous"`` reads the smallest contiguous variant range
          covering the input and sub-slices the result. Cheap when the
          requested variants are dense in their span.
        - ``"sparse"`` builds a per-chunk plan from the indexes via
          ``vcztools.regions.chunk_plan_from_indexes``. Cheap when the
          indexes are scattered across many chunks.
        - ``"auto"`` (default) picks ``"sparse"`` only when the input
          density is below ``SPARSE_VARIANT_THRESHOLD`` AND the span
          exceeds ``SPARSE_VARIANT_THRESHOLD`` * ``num_variants``.
        """
        self._check_open()

        if isinstance(indexes, slice):
            return self._read_variants_slice(indexes, strategy=strategy)

        arr = np.asarray(indexes)
        if arr.ndim != 1:
            raise ValueError(f"indexes must be 1-D (got ndim={arr.ndim})")
        if arr.dtype.kind not in ("i", "u"):
            raise ValueError(f"indexes must be integer dtype (got {arr.dtype})")
        if arr.size == 0:
            return b""
        if arr.min() < 0 or arr.max() >= self._num_variants:
            raise IndexError(f"indexes out of range [0, {self._num_variants})")
        if (np.diff(arr) < 0).any():
            raise ValueError("indexes must be sorted ascending")

        arr = arr.astype(np.int64, copy=False)
        return self._read_variants_array(arr, strategy=strategy)

    def _read_variants_slice(self, sl: slice, *, strategy: str) -> bytes:
        if sl.step not in (None, 1):
            raise ValueError("only step=1 slices are supported")
        start = 0 if sl.start is None else int(sl.start)
        stop = self._num_variants if sl.stop is None else int(sl.stop)
        start = max(0, min(start, self._num_variants))
        stop = max(start, min(stop, self._num_variants))
        if stop <= start:
            return b""
        # A slice is always contiguous; honour an explicit "sparse"
        # request by wrapping the range in an arange ndarray.
        if strategy == "sparse":
            return self._read_variants_array(
                np.arange(start, stop, dtype=np.int64), strategy="sparse"
            )
        if strategy not in ("auto", "contiguous"):
            raise ValueError(f"unknown strategy: {strategy!r}")
        return self._read_contiguous_range(start, stop, picked=None)

    def _read_variants_array(self, arr: np.ndarray, *, strategy: str) -> bytes:
        if strategy == "auto":
            span = int(arr[-1] - arr[0]) + 1
            density = arr.size / self._num_variants
            span_frac = span / self._num_variants
            if (
                density < self.SPARSE_VARIANT_THRESHOLD
                and span_frac > self.SPARSE_VARIANT_THRESHOLD
            ):
                strategy = "sparse"
            else:
                strategy = "contiguous"
        if strategy == "contiguous":
            return self._read_contiguous_range(
                int(arr[0]), int(arr[-1]) + 1, picked=arr
            )
        if strategy == "sparse":
            return self._read_sparse(arr)
        raise ValueError(f"unknown strategy: {strategy!r}")

    def _read_contiguous_range(
        self, start: int, stop: int, *, picked: np.ndarray | None
    ) -> bytes:
        plan = self._build_contiguous_plan(start, stop)
        a12_run = self._a12[start:stop]
        full = b"".join(self._iterate_bed(plan, a12_run))
        if picked is None:
            return full
        bpv = self._bytes_per_variant
        out = bytearray()
        for global_idx in picked:
            local = int(global_idx) - start
            out += full[local * bpv : (local + 1) * bpv]
        return bytes(out)

    def _read_sparse(self, arr: np.ndarray) -> bytes:
        a12_run = self._a12[arr]
        # Pass the ndarray through set_variants so the reader uses
        # chunk_plan_from_indexes internally — same code path the CLI
        # uses for region queries.
        return b"".join(self._iterate_bed(arr, a12_run))

    def read_bed(self, offset: int, size: int) -> bytes:
        """Return ``size`` bytes from the virtual ``.bed`` starting at
        ``offset``. Returns ``b""`` if ``offset >= bed_size``; ``size``
        is silently clamped to the end of the file. ``offset < 0`` and
        ``size < 0`` raise ``ValueError``.
        """
        self._check_open()
        if offset < 0:
            raise ValueError(f"offset must be >= 0 (got {offset})")
        if size < 0:
            raise ValueError(f"size must be >= 0 (got {size})")
        if offset >= self._bed_size or size == 0:
            return b""
        end = min(offset + size, self._bed_size)
        bpv = self._bytes_per_variant

        pieces = []
        if offset < 3:
            pieces.append(BED_MAGIC[offset : min(end, 3)])

        geno_start = max(offset, 3)
        if end > geno_start:
            v_start = (geno_start - 3) // bpv
            v_end = (end - 3 + bpv - 1) // bpv
            v_bytes = self.read_variants(slice(v_start, v_end), strategy="contiguous")
            slice_start = geno_start - (3 + v_start * bpv)
            slice_end = slice_start + (end - geno_start)
            pieces.append(v_bytes[slice_start:slice_end])

        return b"".join(pieces)

    def read_tail(self, nbytes: int = 4096) -> bytes:
        """Return up to ``nbytes`` bytes from the end of the virtual
        ``.bed``. ``nbytes <= 0`` raises ``ValueError``;
        ``nbytes > bed_size`` clamps.
        """
        self._check_open()
        if nbytes <= 0:
            raise ValueError(f"nbytes must be > 0 (got {nbytes})")
        nbytes = min(nbytes, self._bed_size)
        return self.read_bed(self._bed_size - nbytes, nbytes)

    def stream_bed(self, *, chunk_size: int | None = None) -> Iterator[bytes]:
        """Yield the full ``.bed`` byte content (including the
        ``BED_MAGIC`` header) in fragments of approximately
        ``chunk_size`` bytes. Default is ``DEFAULT_STREAM_CHUNK``.

        Multiple in-flight calls are independent — each owns its own
        ``VczReader`` and reads the cached ``a12`` immutably. Callers
        consuming the iterator must drive it to exhaustion (or close
        it) to release the underlying readahead pipeline.
        """
        self._check_open()
        if chunk_size is None:
            chunk_size = self.DEFAULT_STREAM_CHUNK
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0 (got {chunk_size})")

        reader = retrieval.VczReader(
            self._root,
            readahead_workers=self._readahead_workers,
            readahead_bytes=self._readahead_bytes,
        )
        # No set_variants: default plan resolves to full-store iteration.
        buffer = bytearray(BED_MAGIC)
        offset = 0
        for chunk in reader.variant_chunks(fields=["call_genotype"]):
            G = chunk["call_genotype"]
            n = G.shape[0]
            a12_chunk = self._a12[offset : offset + n]
            offset += n
            buffer.extend(encode_genotypes(G, a12_chunk))
            while len(buffer) >= chunk_size:
                yield bytes(buffer[:chunk_size])
                del buffer[:chunk_size]
        if len(buffer) > 0:
            yield bytes(buffer)

    def close(self) -> None:
        """Drop the cached ``.bim``/``.fam`` bytes, the a12 array, and
        the bookkeeping ``VczReader``. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._a12 = None
            self._bim_bytes = None
            self._fam_bytes = None
            self._reader = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
