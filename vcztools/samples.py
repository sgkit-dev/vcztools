import dataclasses
import logging

import numpy as np

from vcztools.utils import _as_fixed_length_unicode, search

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


def parse_samples(
    samples: list[str] | None,
    all_samples: np.ndarray,
    *,
    ignore_missing_samples: bool = False,
    complement: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a list of sample IDs against ``all_samples``.

    ``samples=None`` selects every sample (skipping any empty-string entries
    in the VCZ's ``sample_id`` array). A list selects those samples, or —
    when ``complement=True`` — every sample *except* those.

    Returns ``(sample_ids, samples_selection)``: the array of selected IDs
    and the integer indices of the selection into ``all_samples``. The
    selection is always a concrete array (never ``None``).
    """

    # set a mask if any sample is missing
    mask = all_samples == ""
    all_samples_mask = mask if np.any(mask) else None

    if samples is None:
        if all_samples_mask is None:
            return all_samples, np.arange(all_samples.size)
        sample_ids = all_samples[~all_samples_mask]
        selection = np.arange(all_samples.size)[~all_samples_mask]
        return sample_ids, selection

    sample_ids = np.asarray(samples, dtype=np.dtypes.StringDType())

    unknown_samples = np.setdiff1d(sample_ids, all_samples)
    if len(unknown_samples) > 0:
        if ignore_missing_samples:
            # remove unknown samples from sample_ids
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

    all_samples = _as_fixed_length_unicode(all_samples)
    sample_ids = _as_fixed_length_unicode(sample_ids)

    samples_selection = search(all_samples, sample_ids)
    if complement:
        samples_selection = np.setdiff1d(np.arange(all_samples.size), samples_selection)
    if all_samples_mask is not None:
        # Remove indices of masked (empty-string) samples from the selection.
        # Using indices rather than names avoids an int/string dtype mismatch
        # in np.setdiff1d.
        masked_indices = np.flatnonzero(all_samples_mask)
        samples_selection = np.setdiff1d(samples_selection, masked_indices)
    sample_ids = all_samples[samples_selection]
    return sample_ids, samples_selection


def read_samples_file(path: str) -> list[str]:
    """Read a samples file (one sample ID per line) into a list.

    Blank lines are ignored. The file is decoded as UTF-8 regardless of
    the platform locale, matching the VCF 4.3 spec's encoding for
    sample IDs.
    """
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
