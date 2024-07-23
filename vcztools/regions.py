import re
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyranges


def parse_region(region: str) -> tuple[str, Optional[int], Optional[int]]:
    """Return the contig, start position and end position from a region string."""
    if re.search(r":\d+-\d*$", region):
        contig, start_end = region.rsplit(":", 1)
        start, end = start_end.split("-")
        return contig, int(start), int(end) if len(end) > 0 else None
    elif re.search(r":\d+$", region):
        contig, start = region.rsplit(":", 1)
        return contig, int(start), int(start)
    else:
        contig = region
        return contig, None, None


def parse_regions(targets: str) -> list[tuple[str, Optional[int], Optional[int]]]:
    return [parse_region(region) for region in targets.split(",")]


def parse_targets(targets: str) -> list[tuple[str, Optional[int], Optional[int]]]:
    return [parse_region(region) for region in targets.split(",")]


def regions_to_selection(
    all_contigs: list[str],
    variant_contig: Any,
    variant_position: Any,
    variant_length: Any,
    regions: list[tuple[str, Optional[int], Optional[int]]],
):
    # subtract 1 from start coordinate to convert intervals
    # from VCF (1-based, fully-closed) to Python (0-based, half-open)
    variant_start = variant_position - 1
    variant_end = variant_start + variant_length
    df = pd.DataFrame(
        {"Chromosome": variant_contig, "Start": variant_start, "End": variant_end}
    )

    # save original index as column so we can retrieve it after finding overlap
    df["index"] = df.index
    variants = pyranges.PyRanges(df)

    chromosomes = []
    starts = []
    ends = []
    for contig, start, end in regions:
        if start is None:
            start = 0
        else:
            start -= 1

        if end is None:
            end = np.iinfo(np.int64).max

        chromosomes.append(all_contigs.index(contig))
        starts.append(start)
        ends.append(end)

    query = pyranges.PyRanges(chromosomes=chromosomes, starts=starts, ends=ends)

    overlap = variants.overlap(query)
    if overlap.empty:
        return np.empty((0,), dtype=np.int64)
    return overlap.df["index"].to_numpy()
