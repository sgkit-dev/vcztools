import dataclasses
import logging

import numpy as np

from vcztools.utils import search

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SampleChunkPlan:
    """Plan for reading a subset of sample chunks.

    ``chunk_indexes`` are the sorted unique sample chunks that together
    cover the requested selection. ``local_selection`` is an index into
    the concatenation of those chunks along the samples axis — applying
    it yields samples in the caller's requested order.
    """

    chunk_indexes: np.ndarray
    local_selection: np.ndarray


def build_chunk_plan(
    samples_selection: np.ndarray,
    num_samples: int,
    samples_chunk_size: int,
) -> SampleChunkPlan:
    """Translate a global sample selection into a sample-chunk read plan.

    Given integer indices into the full samples axis plus the axis's
    chunk layout, return the set of sample chunks that must be read and
    the indices into their concatenation that recover the original
    selection (in input order).
    """
    samples_selection = np.asarray(samples_selection)
    chunk_of_each = samples_selection // samples_chunk_size
    chunk_indexes = np.unique(chunk_of_each)

    chunk_starts = chunk_indexes * samples_chunk_size
    chunk_ends = np.minimum(chunk_starts + samples_chunk_size, num_samples)
    chunk_sizes = chunk_ends - chunk_starts
    offsets = np.concatenate(([0], np.cumsum(chunk_sizes[:-1])))

    chunk_pos_of_each = np.searchsorted(chunk_indexes, chunk_of_each)
    local_in_chunk = samples_selection - chunk_starts[chunk_pos_of_each]
    local_selection = offsets[chunk_pos_of_each] + local_in_chunk
    return SampleChunkPlan(chunk_indexes=chunk_indexes, local_selection=local_selection)


def resolve_sample_selection(
    samples: list[str] | None,
    all_samples: np.ndarray,
    *,
    complement: bool = False,
    ignore_missing_samples: bool = False,
) -> list[int] | None:
    """Resolve a bcftools-style sample selection into integer indexes.

    This is the CLI-layer helper for ``-s``/``-S``/``--force-samples``.
    The returned list of indexes (or ``None``) is what
    :class:`~vcztools.retrieval.VczReader` consumes directly as
    ``samples=``.

    - ``samples=None`` and ``complement=False`` → returns ``None``
      (reader selects every real sample).
    - ``samples=None`` and ``complement=True`` → returns an empty list
      (everything excluded).
    - ``samples=list[str]`` and ``complement=False`` → returns the
      indexes of those names in ``all_samples``, in input order,
      after unknown-name handling.
    - ``samples=list[str]`` and ``complement=True`` → returns indexes
      of every real sample NOT in the exclude set, in
      ``all_samples`` order.

    Unknown names either raise ``ValueError`` or are dropped with a
    warning (``ignore_missing_samples=True``). Duplicates in the
    non-complement case raise; in the complement case they are
    deduped silently (matches ``bcftools view -s ^foo,foo``).

    Masked (empty-string) entries in ``all_samples`` are never
    returned.
    """
    if samples is None:
        if not complement:
            return None
        return [i for i, s in enumerate(all_samples.tolist()) if s != ""]

    sample_ids = np.asarray(samples, dtype=np.dtypes.StringDType())
    unknown_samples = np.setdiff1d(sample_ids, all_samples)
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

    name_to_index = {s: i for i, s in enumerate(all_samples.tolist()) if s != ""}

    if not complement:
        unique_ids, counts = np.unique(sample_ids, return_counts=True)
        duplicates = unique_ids[counts > 1]
        if duplicates.size > 0:
            raise ValueError(f'Duplicate sample name "{duplicates[0]}".')
        return [name_to_index[s] for s in sample_ids.tolist() if s in name_to_index]

    excluded = set(sample_ids.tolist())
    return [idx for name, idx in name_to_index.items() if name not in excluded]


def read_samples_file(path: str) -> list[str]:
    """Read a samples file (one sample ID per line) into a list.

    Blank lines are ignored. The file is decoded as UTF-8 regardless of
    the platform locale, matching the VCF 4.3 spec's encoding for
    sample IDs.
    """
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
