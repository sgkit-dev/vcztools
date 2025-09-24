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

    if samples is None:
        return all_samples, None
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
    sample_ids = all_samples[samples_selection]
    return sample_ids, samples_selection
