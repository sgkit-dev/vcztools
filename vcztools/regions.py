import re
from typing import Any, List, Optional

import numpy as np


def parse_targets_string(targets: str) -> tuple[str, Optional[int], Optional[int]]:
    """Return the contig, start position and end position from a targets string."""
    if re.search(r":\d+-\d*$", targets):
        contig, start_end = targets.rsplit(":", 1)
        start, end = start_end.split("-")
        return contig, int(start), int(end) if len(end) > 0 else None
    elif re.search(r":\d+$", targets):
        contig, start = targets.rsplit(":", 1)
        return contig, int(start), int(start)
    else:
        contig = targets
        return contig, None, None


def region_to_slice(
    all_contigs: List[str],
    variant_contig: Any,
    variant_position: Any,
    contig: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> slice:
    """Convert a VCF region to a Python slice."""

    # subtract one from start since VCF is 1-based (fully closed),
    # but Python slices are 0-based (half open)
    start = None if start is None else start - 1

    contig_index = all_contigs.index(contig)
    contig_range = np.searchsorted(variant_contig, [contig_index, contig_index + 1])

    if start is None and end is None:
        start_index, end_index = contig_range
    else:
        contig_pos = variant_position[slice(contig_range[0], contig_range[1])]
        if start is None:
            assert end is not None  # covered by case above
            start_index = contig_range[0]
            end_index = contig_range[0] + np.searchsorted(contig_pos, [end])[0]
        elif end is None:
            start_index = contig_range[0] + np.searchsorted(contig_pos, [start])[0]
            end_index = contig_range[1]
        else:
            start_index, end_index = contig_range[0] + np.searchsorted(
                contig_pos, [start, end]
            )

    return slice(start_index, end_index)
