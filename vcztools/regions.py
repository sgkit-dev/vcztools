import re
from typing import Any, Optional

import numpy as np
import pandas as pd
from pyranges import PyRanges


def parse_region_string(region: str) -> tuple[str, Optional[int], Optional[int]]:
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


def regions_to_pyranges(
    regions: list[tuple[str, Optional[int], Optional[int]]], all_contigs: list[str]
) -> PyRanges:
    """Convert region tuples to a PyRanges object."""

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

    return PyRanges(chromosomes=chromosomes, starts=starts, ends=ends)


def parse_regions(regions: Optional[str], all_contigs: list[str]) -> Optional[PyRanges]:
    """Return a PyRanges object from a comma-separated set of region strings."""
    if regions is None:
        return None
    return regions_to_pyranges(
        [parse_region_string(region) for region in regions.split(",")], all_contigs
    )


def parse_targets(
    targets: Optional[str], all_contigs: list[str]
) -> tuple[Optional[PyRanges], bool]:
    """Return a PyRanges object from a comma-separated set of region strings,
    optionally preceeded by a ^ character to indicate complement."""
    if targets is None:
        return None, False
    complement = targets.startswith("^")
    return (
        parse_regions(targets[1:] if complement else targets, all_contigs),
        complement,
    )


def regions_to_chunk_indexes(
    regions: Optional[PyRanges],
    targets: Optional[PyRanges],
    complement: bool,
    regions_index: Any,
):
    """Return chunks indexes that overlap the given regions or targets.

    If both regions and targets are specified then only regions are used
    to find overlapping chunks (since targets are used later to refine).

    If only targets are specified then they are used to find overlapping chunks,
    taking into account the complement flag.
    """

    # Create pyranges for chunks using the region index.
    # For regions use max end position, for targets just end position
    chunk_index = regions_index[:, 0]
    contig_id = regions_index[:, 1]
    start_position = regions_index[:, 2]
    end_position = regions_index[:, 3]
    max_end_position = regions_index[:, 4]
    df = pd.DataFrame(
        {
            "chunk_index": chunk_index,
            "Chromosome": contig_id,
            "Start": start_position,
            "End": max_end_position if regions is not None else end_position,
        }
    )
    chunk_regions = PyRanges(df)

    if regions is not None:
        overlap = chunk_regions.overlap(regions)
    elif complement:
        overlap = chunk_regions.subtract(targets)
    else:
        overlap = chunk_regions.overlap(targets)
    if overlap.empty:
        return np.empty((0,), dtype=np.int64)
    chunk_indexes = overlap.df["chunk_index"].to_numpy()
    chunk_indexes = np.unique(chunk_indexes)
    return chunk_indexes


def regions_to_selection(
    regions: Optional[PyRanges],
    targets: Optional[PyRanges],
    complement: bool,
    variant_contig: Any,
    variant_position: Any,
    variant_length: Any,
):
    """Return a variant selection that corresponds to the given regions and targets.

    If both regions and targets are specified then they are both used to find
    overlapping variants.
    """

    # subtract 1 from start coordinate to convert intervals
    # from VCF (1-based, fully-closed) to Python (0-based, half-open)
    variant_start = variant_position - 1

    if regions is not None:
        variant_end = variant_start + variant_length
        df = pd.DataFrame(
            {"Chromosome": variant_contig, "Start": variant_start, "End": variant_end}
        )
        # save original index as column so we can retrieve it after finding overlap
        df["index"] = df.index
        variant_regions = PyRanges(df)
    else:
        variant_regions = None

    if targets is not None:
        targets_variant_end = variant_position  # length 1
        df = pd.DataFrame(
            {
                "Chromosome": variant_contig,
                "Start": variant_start,
                "End": targets_variant_end,
            }
        )
        # save original index as column so we can retrieve it after finding overlap
        df["index"] = df.index
        variant_targets = PyRanges(df)
    else:
        variant_targets = None

    if variant_regions is not None:
        regions_overlap = variant_regions.overlap(regions)
    else:
        regions_overlap = None

    if variant_targets is not None:
        if complement:
            targets_overlap = variant_targets.subtract(targets)
        else:
            targets_overlap = variant_targets.overlap(targets)
    else:
        targets_overlap = None

    if regions_overlap is not None and targets_overlap is not None:
        overlap = regions_overlap.overlap(targets_overlap)
    elif regions_overlap is not None:
        overlap = regions_overlap
    else:
        overlap = targets_overlap

    if overlap.empty:
        return np.empty((0,), dtype=np.int64)
    return overlap.df["index"].to_numpy()
