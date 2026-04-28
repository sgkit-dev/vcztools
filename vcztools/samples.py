import dataclasses
import logging

import numpy as np

from vcztools import utils
from vcztools.utils import search

logger = logging.getLogger(__name__)


def _normalise_local_selection(
    local_sel: np.ndarray, samples_chunk_size: int
) -> np.ndarray | slice | None:
    """Collapse a contiguous, sorted, no-duplicate per-chunk selection
    into a basic-indexing form so :class:`~vcztools.retrieval.CachedVariantChunk`
    can skip a no-op fancy-index gather.

    - Full chunk in order → ``None`` (CachedVariantChunk emits the raw block).
    - Contiguous range in order → ``slice(start, stop)`` (basic
      indexing, returns a view; later concatenate copies once instead
      of twice).
    - Anything else → the original ndarray (gather as before).

    The check is O(N) per chunk; build_chunk_plan runs once per query,
    so the cost is paid once and the saving is per-variant-chunk.
    """
    if local_sel.size == 0:
        return local_sel
    start = int(local_sel[0])
    stop = start + int(local_sel.size)
    # Quick reject: if last element doesn't match a contiguous range,
    # skip the array_equal scan.
    if int(local_sel[-1]) != stop - 1:
        return local_sel
    if not np.array_equal(local_sel, np.arange(start, stop)):
        return local_sel
    if start == 0 and stop == samples_chunk_size:
        return None
    return slice(start, stop)


@dataclasses.dataclass
class SampleChunkPlan:
    """Plan for reading a subset of sample chunks.

    ``chunk_reads`` lists the sample chunks to read, each with a
    per-chunk local selection (``selection=None`` means "full chunk").
    ``permutation`` reorders the concatenation of the per-chunk
    subsets into the caller's requested sample order; ``None`` means
    the concatenation is already in the caller's order.
    """

    chunk_reads: list[utils.ChunkRead]
    permutation: np.ndarray | None = None


def build_chunk_plan(
    samples_selection: np.ndarray,
    samples_chunk_size: int,
) -> SampleChunkPlan:
    """Translate a global sample selection into a sample-chunk read plan.

    Buckets global sample indexes into chunks, pre-subsets per chunk,
    and computes an optional permutation that restores the caller's
    input order after chunk-sorted concatenation. ``permutation is
    None`` when the caller's order already matches the chunk-sorted
    concatenation (the fast path).
    """
    samples_selection = np.asarray(samples_selection, dtype=np.int64)
    chunk_of_each = samples_selection // samples_chunk_size
    sort_idx = np.argsort(chunk_of_each, kind="stable")
    sorted_samples = samples_selection[sort_idx]
    sorted_chunks = chunk_of_each[sort_idx]
    chunk_indexes, counts = np.unique(sorted_chunks, return_counts=True)

    chunk_reads = []
    offset = 0
    for ci, count in zip(chunk_indexes, counts):
        local_sel = (
            sorted_samples[offset : offset + count] - int(ci) * samples_chunk_size
        )
        chunk_reads.append(
            utils.ChunkRead(
                index=int(ci),
                selection=_normalise_local_selection(local_sel, samples_chunk_size),
            )
        )
        offset += count

    permutation = np.argsort(sort_idx)
    if np.array_equal(permutation, np.arange(len(permutation))):
        permutation = None
    return SampleChunkPlan(chunk_reads=chunk_reads, permutation=permutation)


def resolve_sample_selection(
    samples: list[str] | None,
    raw_sample_ids: np.ndarray,
    *,
    complement: bool = False,
    ignore_missing_samples: bool = False,
) -> np.ndarray | None:
    """Resolve a bcftools-style sample selection into integer indexes.

    This is the CLI-layer helper for ``-s``/``-S``/``--force-samples``.
    The returned integer ndarray (or ``None``) is what
    :class:`~vcztools.retrieval.VczReader` consumes directly as
    ``samples=``.

    - ``samples=None`` and ``complement=False`` → returns ``None``
      (reader selects every non-null sample).
    - ``samples=None`` and ``complement=True`` → returns every
      non-null sample index (complement of an empty exclude set).
    - ``samples=list[str]`` and ``complement=False`` → returns the
      indexes of those names in ``raw_sample_ids``, in input order,
      after unknown-name handling.
    - ``samples=list[str]`` and ``complement=True`` → returns indexes
      of every non-null sample NOT in the exclude set, in
      ``raw_sample_ids`` order.

    Unknown names either raise ``ValueError`` or are dropped with a
    warning (``ignore_missing_samples=True``). Duplicates in the
    non-complement case raise; in the complement case they are
    deduped silently (matches ``bcftools view -s ^foo,foo``).

    Null (empty-string) entries in ``raw_sample_ids`` are never
    returned.
    """
    non_null_select = raw_sample_ids != ""

    if samples is None:
        if not complement:
            return None
        return np.flatnonzero(non_null_select)

    sample_ids = np.asarray(samples, dtype=np.dtypes.StringDType())
    unknown_samples = np.setdiff1d(sample_ids, raw_sample_ids)
    if len(unknown_samples) > 0:
        if ignore_missing_samples:
            logger.warning(
                "subset called for sample(s) not in header: "
                f"{','.join(unknown_samples)}."
            )
            sample_ids = np.delete(sample_ids, search(sample_ids, unknown_samples))
        else:
            raise ValueError(
                "subset called for sample(s) not in header: "
                f"{','.join(unknown_samples)}. "
                'Use "--force-samples" to ignore this error.'
            )

    if not complement:
        unique_ids, counts = np.unique(sample_ids, return_counts=True)
        duplicates = unique_ids[counts > 1]
        if duplicates.size > 0:
            raise ValueError(f'Duplicate sample name "{duplicates[0]}".')
        # Drop empty-string requests — they would otherwise map to a
        # null header position, and those are never returned.
        sample_ids = sample_ids[sample_ids != ""]
        # Match raw_sample_ids' dtype so searchsorted is a safe cast.
        # Every remaining sample_id is a known entry in raw_sample_ids
        # (per the unknown-check above), so this can't truncate.
        return search(raw_sample_ids, sample_ids.astype(raw_sample_ids.dtype))

    select = non_null_select & ~np.isin(
        raw_sample_ids, sample_ids.astype(raw_sample_ids.dtype)
    )
    return np.flatnonzero(select)


def read_samples_file(path: str) -> list[str]:
    """Read a samples file (one sample ID per line) into a list.

    Blank lines are ignored. The file is decoded as UTF-8 regardless of
    the platform locale, matching the VCF 4.3 spec's encoding for
    sample IDs.
    """
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
