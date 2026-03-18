import re
from typing import Any

import numpy as np
import pandas as pd

try:
    # Use ruranges if installed
    from ruranges_py import overlaps, subtract

    class GenomicRanges:
        def __init__(self, contigs, starts, ends, complement=False):
            # note that ruranges groups must be unsigned
            self.contigs = np.ascontiguousarray(contigs, dtype=np.uint64)
            self.starts = np.ascontiguousarray(starts)
            self.ends = np.ascontiguousarray(ends)
            self.complement = complement

        def overlaps(self, other: "GenomicRanges"):
            if self.complement:
                overlap = subtract(
                    groups=self.contigs,
                    starts=self.starts,
                    ends=self.ends,
                    groups2=other.contigs,
                    starts2=other.starts,
                    ends2=other.ends,
                )
            else:
                overlap = overlaps(
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
                f"starts={self.starts}, ends={self.ends}, "
                f"complement={self.complement})"
            )

except ImportError:
    # Otherwise fallback to older pyranges
    import pandas as pd
    from pyranges import PyRanges

    class GenomicRanges:
        def __init__(self, contigs, starts, ends, complement=False):
            df = pd.DataFrame(
                {
                    "Chromosome": contigs,
                    "Start": starts,
                    "End": ends,
                }
            )
            df["index"] = df.index
            self.pyranges = PyRanges(df)
            self.contigs = df["Chromosome"].to_numpy()
            self.starts = df["Start"].to_numpy()
            self.ends = df["End"].to_numpy()
            self.complement = complement

        def overlaps(self, other: "GenomicRanges"):
            if self.complement:
                overlap = self.pyranges.subtract(other.pyranges)
            else:
                overlap = self.pyranges.overlap(other.pyranges)
            if overlap.empty:
                return np.empty((0,), dtype=np.int64)
            return overlap.df["index"].to_numpy()


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


def regions_to_ranges(
    regions: list[tuple[str, int | None, int | None]],
    all_contigs: list[str],
    complement: bool = False,
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

    return GenomicRanges(
        contigs=contigs, starts=starts, ends=ends, complement=complement
    )


def _parse_regions_or_targets(
    regions: GenomicRanges | list[str] | str | None,
    all_contigs: list[str],
    regions_file: str | None = None,
    allow_complement: bool = False,
) -> GenomicRanges | None:
    assert regions is None or regions_file is None  # only one is set
    if regions is None and regions_file is not None:
        return _parse_regions_or_targets_file(
            regions_file, all_contigs, allow_complement=allow_complement
        )
    elif regions is None or isinstance(regions, GenomicRanges):
        return regions
    elif isinstance(regions, list):
        regions_list = regions
        complement = False
    else:
        assert isinstance(regions, str)
        if allow_complement:
            complement = regions.startswith("^")
            if complement:
                regions = regions[1:]
        else:
            complement = False
        regions_list = regions.split(",")
    return regions_to_ranges(
        [parse_region_string(region) for region in regions_list],
        all_contigs,
        complement=complement,
    )


def _parse_regions_or_targets_file(
    regions_file: str, all_contigs: list[str], allow_complement: bool = False
) -> GenomicRanges:
    if allow_complement:
        complement = regions_file.startswith("^")
        if complement:
            regions_file = regions_file[1:]
    else:
        complement = False
    df = pd.read_csv(
        regions_file,
        sep="\t",
        names=["Chromosome", "Start", "End"],
        dtype={"Chromosome": str, "Start": np.int64, "End": np.int64},
    )
    # transform contig names to indexes, and convert intervals
    # from VCF (1-based, fully-closed) to Python (0-based, half-open)
    df["Chromosome"] = df["Chromosome"].apply(lambda c: all_contigs.index(c))
    df = df.astype({"Chromosome": np.int32})
    df["Start"] = df["Start"] - 1
    return GenomicRanges(
        df["Chromosome"].to_numpy(),
        df["Start"].to_numpy(),
        df["End"].to_numpy(),
        complement=complement,
    )


def parse_regions(
    regions: GenomicRanges | list[str] | str | None,
    all_contigs: list[str],
    regions_file: str | None = None,
) -> GenomicRanges | None:
    """Return a GenomicRanges object from a comma-separated set of region strings,
    or a list of region strings."""
    if regions is not None and regions_file is not None:
        raise ValueError(
            "Cannot specify both a regions string (-r) and a regions file (-R)"
        )
    return _parse_regions_or_targets(regions, all_contigs, regions_file=regions_file)


def parse_targets(
    targets: list[str] | str | None,
    all_contigs: list[str],
    targets_file: str | None = None,
) -> GenomicRanges | None:
    """Return a GenomicRanges object from a comma-separated set of region strings,
    optionally preceeded by a ^ character to indicate complement,
    or a list of region strings."""
    if targets is not None and targets_file is not None:
        raise ValueError(
            "Cannot specify both a target string (-t) and a targets file (-T)"
        )
    return _parse_regions_or_targets(
        targets, all_contigs, regions_file=targets_file, allow_complement=True
    )


def regions_to_chunk_indexes(regions: GenomicRanges, regions_index: Any):
    """Return chunks indexes that overlap the given regions."""

    # Create GenomicRanges for chunks using the region index.
    # For regions use max end position, for targets just end position
    chunk_index = regions_index[:, 0]
    contig_id = regions_index[:, 1]
    start_position = regions_index[:, 2]
    # end_position = regions_index[:, 3]
    max_end_position = regions_index[:, 4]
    # subtract 1 from start coordinate to convert intervals
    # from VCF (1-based, fully-closed) to Python (0-based, half-open)
    chunk_regions = GenomicRanges(contig_id, start_position - 1, max_end_position)

    overlap = chunk_regions.overlaps(regions)
    chunk_indexes = chunk_index[overlap]
    chunk_indexes = np.unique(chunk_indexes)
    return chunk_indexes


def regions_to_selection(
    regions: GenomicRanges | None,
    targets: GenomicRanges | None,
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
            variant_contig,
            variant_start,
            targets_variant_end,
            complement=targets.complement,
        )
    else:
        variant_targets = None

    if variant_regions is not None:
        regions_overlap = variant_regions.overlaps(regions)
    else:
        regions_overlap = None

    if variant_targets is not None:
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
