import dataclasses
import re
from typing import Any

import numpy as np
import pandas as pd


@dataclasses.dataclass
class Region:
    contig: str
    start: int | None = None
    end: int | None = None


def _to_int32_coords(values, name):
    """Cast a coordinate array to int32, raising if any value overflows."""
    arr = np.asarray(values)
    info = np.iinfo(np.int32)
    if arr.size > 0 and (arr.min() < info.min or arr.max() > info.max):
        raise ValueError(
            f"{name} coordinate out of range for int32 [{info.min}, {info.max}]"
        )
    return arr.astype(np.int32)


try:
    # Use ruranges if installed
    from ruranges_py import overlaps, subtract

    class GenomicRanges:
        def __init__(self, contigs, starts, ends, complement=False):
            # note that ruranges groups must be unsigned
            self.contigs = np.ascontiguousarray(contigs, dtype=np.uint64)
            self.starts = np.ascontiguousarray(_to_int32_coords(starts, "start"))
            self.ends = np.ascontiguousarray(_to_int32_coords(ends, "end"))
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
                    "Start": _to_int32_coords(starts, "start"),
                    "End": _to_int32_coords(ends, "end"),
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


def parse_region_string(region: str) -> Region:
    """Return a Region from a region string."""
    if re.search(r":\d+-\d*$", region):
        contig, start_end = region.rsplit(":", 1)
        start, end = start_end.split("-")
        return Region(contig, int(start), int(end) if len(end) > 0 else None)
    elif re.search(r":\d+$", region):
        contig, start = region.rsplit(":", 1)
        return Region(contig, int(start), int(start))
    else:
        return Region(region)


def parse_regions_file(path: str) -> list[Region]:
    """Read a bcftools-style regions/targets TSV file into Region objects."""
    regions = []
    with open(path) as f:
        for line_num, line in enumerate(f, start=1):
            parts = line.strip().split("\t")
            if len(parts) < 3:
                raise ValueError(
                    f"expected at least 3 tab-separated columns "
                    f"(chrom, start, end), got {len(parts)} at line {line_num}: "
                    f"{path}"
                )
            try:
                start = int(parts[1])
            except ValueError:
                raise ValueError(
                    f"non-numeric start position '{parts[1]}' "
                    f"at line {line_num}: {path}"
                ) from None
            try:
                end = int(parts[2])
            except ValueError:
                raise ValueError(
                    f"non-numeric end position '{parts[2]}' at line {line_num}: {path}"
                ) from None
            regions.append(Region(contig=parts[0], start=start, end=end))
    if len(regions) == 0:
        raise ValueError(f"regions file is empty: {path}")
    return regions


def regions_to_ranges(
    regions: list[Region],
    all_contigs: list[str],
    complement: bool = False,
) -> GenomicRanges:
    """Convert Region objects to a GenomicRanges object."""

    contigs = []
    starts = []
    ends = []
    for region in regions:
        if region.start is None:
            start = 0
        else:
            start = region.start - 1

        end = np.iinfo(np.int32).max if region.end is None else region.end

        contigs.append(all_contigs.index(region.contig))
        starts.append(start)
        ends.append(end)

    return GenomicRanges(
        contigs=contigs, starts=starts, ends=ends, complement=complement
    )


def _parse_regions_or_targets(
    regions: GenomicRanges | list[Region] | list[str] | str | None,
    all_contigs: list[str],
    allow_complement: bool = False,
    complement: bool = False,
) -> GenomicRanges | None:
    if regions is None or isinstance(regions, GenomicRanges):
        return regions

    if (
        isinstance(regions, list)
        and len(regions) > 0
        and isinstance(regions[0], Region)
    ):
        return regions_to_ranges(regions, all_contigs, complement=complement)

    if isinstance(regions, list):
        regions_list = regions
    else:
        assert isinstance(regions, str)
        if allow_complement:
            complement = regions.startswith("^")
            if complement:
                regions = regions[1:]
        regions_list = regions.split(",")
    return regions_to_ranges(
        [parse_region_string(region) for region in regions_list],
        all_contigs,
        complement=complement,
    )


def parse_regions(
    regions: GenomicRanges | list[Region] | list[str] | str | None,
    all_contigs: list[str],
) -> GenomicRanges | None:
    """Return a GenomicRanges object from a comma-separated set of region strings,
    a list of region strings, or a list of Region objects."""
    return _parse_regions_or_targets(regions, all_contigs)


def parse_targets(
    targets: GenomicRanges | list[Region] | list[str] | str | None,
    all_contigs: list[str],
    complement: bool = False,
) -> GenomicRanges | None:
    """Return a GenomicRanges object from a comma-separated set of region strings,
    optionally preceeded by a ^ character to indicate complement,
    or a list of region strings, or a list of Region objects."""
    return _parse_regions_or_targets(
        targets, all_contigs, allow_complement=True, complement=complement
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
