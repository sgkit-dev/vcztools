import re
from typing import Any

import numpy as np
from ruranges_py import overlaps, subtract


def parse_region_string(region: str) -> tuple[str, int | None, int | None]:
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


class GenomicRanges:
    def __init__(self, contigs, starts, ends):
        # note that ruranges groups must be unsigned
        self.contigs = np.ascontiguousarray(contigs, dtype=np.uint64)
        self.starts = np.ascontiguousarray(starts)
        self.ends = np.ascontiguousarray(ends)

    def overlaps(self, other: "GenomicRanges"):
        overlap = overlaps(
            groups=self.contigs,
            starts=self.starts,
            ends=self.ends,
            groups2=other.contigs,
            starts2=other.starts,
            ends2=other.ends,
        )
        return overlap[0]  # indices of overlapping regions with self

    def subtract(self, other: "GenomicRanges"):
        overlap = subtract(
            groups=self.contigs,
            starts=self.starts,
            ends=self.ends,
            groups2=other.contigs,
            starts2=other.starts,
            ends2=other.ends,
        )
        return overlap[0]  # indices of overlapping regions with self

    def __str__(self) -> str:
        return (
            f"GenomicRanges(contigs={self.contigs}, "
            f"starts={self.starts}, ends={self.ends})"
        )


def regions_to_ranges(
    regions: list[tuple[str, int | None, int | None]], all_contigs: list[str]
) -> GenomicRanges:
    """Convert region tuples to a GenomicRanges object."""

    contigs = []
    starts = []
    ends = []
    for contig, start, end in regions:
        if start is None:
            start = 0
        else:
            start -= 1

        if end is None:
            end = np.iinfo(np.int64).max

        contigs.append(all_contigs.index(contig))
        starts.append(start)
        ends.append(end)

    return GenomicRanges(np.array(contigs), np.array(starts), np.array(ends))


def parse_regions(
    regions: list[str] | str | None, all_contigs: list[str]
) -> GenomicRanges | None:
    """Return a GenomicRanges object from a comma-separated set of region strings,
    or a list of region strings."""
    if regions is None:
        return None
    elif isinstance(regions, list):
        regions_list = regions
    else:
        regions_list = regions.split(",")
    return regions_to_ranges(
        [parse_region_string(region) for region in regions_list], all_contigs
    )


def parse_targets(
    targets: list[str] | str | None, all_contigs: list[str]
) -> tuple[GenomicRanges | None, bool]:
    """Return a GenomicRanges object from a comma-separated set of region strings,
    optionally preceeded by a ^ character to indicate complement,
    or a list of region strings."""
    if targets is None:
        return None, False
    elif isinstance(targets, list):
        targets_list = targets
        complement = False
    else:
        complement = targets.startswith("^")
        targets_list = (targets[1:] if complement else targets).split(",")
    return (
        parse_regions(targets_list, all_contigs),
        complement,
    )


def regions_to_chunk_indexes(
    regions: GenomicRanges | None,
    targets: GenomicRanges | None,
    complement: bool,
    regions_index: Any,
):
    """Return chunks indexes that overlap the given regions or targets.

    If both regions and targets are specified then only regions are used
    to find overlapping chunks (since targets are used later to refine).

    If only targets are specified then they are used to find overlapping chunks,
    taking into account the complement flag.
    """

    # Create GenomicRanges for chunks using the region index.
    # For regions use max end position, for targets just end position
    chunk_index = regions_index[:, 0]
    contig_id = regions_index[:, 1]
    start_position = regions_index[:, 2]
    end_position = regions_index[:, 3]
    max_end_position = regions_index[:, 4]
    # subtract 1 from start coordinate to convert intervals
    # from VCF (1-based, fully-closed) to Python (0-based, half-open)
    chunk_regions = GenomicRanges(
        contig_id,
        start_position - 1,
        max_end_position if regions is not None else end_position,
    )

    if regions is not None:
        overlap = chunk_regions.overlaps(regions)
    elif complement:
        overlap = chunk_regions.subtract(targets)
    else:
        overlap = chunk_regions.overlaps(targets)
    chunk_indexes = chunk_index[overlap]
    chunk_indexes = np.unique(chunk_indexes)
    return chunk_indexes


def regions_to_selection(
    regions: GenomicRanges | None,
    targets: GenomicRanges | None,
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
        variant_regions = GenomicRanges(variant_contig, variant_start, variant_end)
    else:
        variant_regions = None

    if targets is not None:
        targets_variant_end = variant_position  # length 1
        variant_targets = GenomicRanges(
            variant_contig, variant_start, targets_variant_end
        )
    else:
        variant_targets = None

    if variant_regions is not None:
        regions_overlap = variant_regions.overlaps(regions)
    else:
        regions_overlap = None

    if variant_targets is not None:
        if complement:
            targets_overlap = variant_targets.subtract(targets)
        else:
            targets_overlap = variant_targets.overlaps(targets)
    else:
        targets_overlap = None

    if regions_overlap is not None and targets_overlap is not None:
        overlap = np.intersect1d(regions_overlap, targets_overlap)
    elif regions_overlap is not None:
        overlap = regions_overlap
    else:
        overlap = targets_overlap
    return overlap
