import logging

import numpy as np

from vcztools.utils import _as_fixed_length_unicode, search

logger = logging.getLogger(__name__)


def parse_samples(
    samples: list[str] | str | None,
    all_samples: np.ndarray,
    *,
    force_samples: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Parse a bcftools-style samples string, or a list of sample IDs.

    Returns an array of the sample IDs, and an array indicating the selection
    from all samples.
    """

    # set a mask if any sample is missing
    mask = all_samples == ""
    all_samples_mask = mask if np.any(mask) else None

    if samples is None:
        if all_samples_mask is None:
            return all_samples, None
        else:
            sample_ids = all_samples[~all_samples_mask]
            selection = np.arange(all_samples.size)[~all_samples_mask]
        return sample_ids, selection
    elif isinstance(samples, list):
        exclude_samples = False
        sample_ids = np.array(samples)
    else:
        exclude_samples = samples.startswith("^")
        samples = samples.lstrip("^")
        sample_ids = np.array(samples.split(","))

    if np.all(sample_ids == np.array("")):
        sample_ids = np.empty((0,), dtype=np.dtypes.StringDType())

    unknown_samples = np.setdiff1d(sample_ids, all_samples)
    if len(unknown_samples) > 0:
        if force_samples:
            # remove unknown samples from sample_ids
            logger.warning(
                "subset called for sample(s) not in header: "
                f'{",".join(unknown_samples)}.'
            )
            sample_ids = np.delete(sample_ids, search(sample_ids, unknown_samples))
        else:
            raise ValueError(
                "subset called for sample(s) not in header: "
                f'{",".join(unknown_samples)}. '
                'Use "--force-samples" to ignore this error.'
            )

    all_samples = _as_fixed_length_unicode(all_samples)
    sample_ids = _as_fixed_length_unicode(sample_ids)

    samples_selection = search(all_samples, sample_ids)
    if exclude_samples:
        samples_selection = np.setdiff1d(np.arange(all_samples.size), samples_selection)
    if all_samples_mask is not None:
        masked_sample_ids = all_samples[all_samples_mask]
        samples_selection = np.setdiff1d(samples_selection, masked_sample_ids)
    sample_ids = all_samples[samples_selection]
    return sample_ids, samples_selection


def parse_samples_file(samples_file: str) -> str:
    """Parse a file of sample IDs.

    Returns a comma-delimited string of sample IDs,
    optionally preceeded by a ^ character to indicate complement.
    """
    samples = ""
    exclude_samples_file = samples_file.startswith("^")
    samples_file = samples_file.lstrip("^")

    with open(samples_file) as file:
        if exclude_samples_file:
            samples = "^" + samples
        samples += ",".join(line.strip() for line in file.readlines())

        return samples
