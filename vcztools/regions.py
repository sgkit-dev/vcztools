import re
from typing import Any

import numpy as np
import pandas as pd

from vcztools import utils
from vcztools.utils import _as_fixed_length_unicode

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
    """Parse a region string into a (contig, start, end) tuple.

    Region-string forms:

    - "chr1"          -> ("chr1", None, None)
    - "chr1:100"      -> ("chr1", 100, 100)
    - "chr1:100-"     -> ("chr1", 100, None)
    - "chr1:100-200"  -> ("chr1", 100, 200)

    A contig name may itself contain ``:`` characters; only the final
    ``:`` is treated as the contig/position separator.
    """
    if re.search(r":\d+-\d*$", region):
        contig, start_end = region.rsplit(":", 1)
        start_str, end_str = start_end.split("-")
        end = int(end_str) if len(end_str) > 0 else None
        return contig, int(start_str), end
    if re.search(r":\d+$", region):
        contig, start_str = region.rsplit(":", 1)
        start = int(start_str)
        return contig, start, start
    return region, None, None


def _regions_dataframe(contigs, starts, ends) -> pd.DataFrame:
    """Build a regions DataFrame with the documented schema."""
    return pd.DataFrame(
        {
            "contig": contigs,
            "start": pd.array(starts, dtype="Int64"),
            "end": pd.array(ends, dtype="Int64"),
        }
    )


def region_strings_to_dataframe(regions: list[str]) -> pd.DataFrame:
    """Parse a list of region strings into a DataFrame.

    The returned DataFrame has columns ``contig`` (str), ``start`` (nullable
    ``Int64``) and ``end`` (nullable ``Int64``); ``pd.NA`` represents an
    unbounded start (``"chr1"``) or an unbounded end (``"chr1:100-"``).
    """
    contigs = []
    starts = []
    ends = []
    for s in regions:
        contig, start, end = parse_region_string(s)
        contigs.append(contig)
        starts.append(pd.NA if start is None else start)
        ends.append(pd.NA if end is None else end)
    return _regions_dataframe(contigs, starts, ends)


def read_regions_file(path: str) -> pd.DataFrame:
    """Read a bcftools-style regions/targets TSV file into a DataFrame.

    The file must contain at least three tab-separated columns:
    ``contig``, ``start`` and ``end``. Any further columns are ignored.
    The returned DataFrame has the schema produced by
    :func:`region_strings_to_dataframe`.
    """
    contigs: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
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
            contigs.append(parts[0])
            starts.append(start)
            ends.append(end)
    if len(contigs) == 0:
        raise ValueError(f"regions file is empty: {path}")
    return _regions_dataframe(contigs, starts, ends)


def dataframe_to_ranges(
    df: pd.DataFrame | None,
    all_contigs: list[str],
    complement: bool = False,
) -> GenomicRanges | None:
    """Convert a regions/targets DataFrame to a GenomicRanges object.

    Resolves ``contig`` names against ``all_contigs`` and applies default
    sentinels for unbounded positions: ``start=NA`` becomes ``0`` (after
    1-based→0-based conversion) and ``end=NA`` becomes ``int64.max``.
    ``complement=True`` flips the sense so the returned ranges describe
    everything *outside* the listed intervals. Returns ``None`` when ``df``
    is ``None``.
    """
    if df is None:
        return None
    int64_max = np.iinfo(np.int64).max
    contig_indexes = [all_contigs.index(c) for c in df["contig"]]
    starts = [0 if pd.isna(s) else int(s) - 1 for s in df["start"]]
    ends = [int64_max if pd.isna(e) else int(e) for e in df["end"]]
    return GenomicRanges(
        contigs=contig_indexes, starts=starts, ends=ends, complement=complement
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


_REGION_DF_COLUMNS = ("contig", "start", "end")


def _regions_input_to_df(value, *, arg_name: str) -> pd.DataFrame | None:
    """Normalise a regions/targets input into the documented DataFrame
    schema used internally by :func:`build_chunk_plan`.

    Accepts ``None``, a single region string, a list of region strings,
    or a DataFrame with ``contig``, ``start`` and ``end`` columns. A
    leading ``^`` on a string input is rejected with a ``ValueError``
    pointing users at the ``targets_complement`` flag.
    """
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        missing = set(_REGION_DF_COLUMNS) - set(value.columns)
        if missing:
            raise ValueError(
                f"{arg_name} DataFrame is missing required columns: {sorted(missing)}"
            )
        return value
    if isinstance(value, str):
        if value.startswith("^"):
            raise ValueError(
                f"{arg_name} does not accept a '^' prefix in the Python API; "
                f"use targets_complement=True for complement"
            )
        if "," in value:
            raise ValueError(
                f"{arg_name} string '{value}' contains ','. "
                f"Pass a list[str] for multiple regions."
            )
        return region_strings_to_dataframe([value])
    if isinstance(value, list):
        return region_strings_to_dataframe(value)
    raise TypeError(
        f"{arg_name} must be str, list[str], pandas.DataFrame, or None; "
        f"got {type(value).__name__}"
    )


def chunk_plan_from_indexes(
    variant_indexes: np.ndarray,
    *,
    variants_chunk_size: int,
) -> list[utils.ChunkRead]:
    """Build a plan from a sorted flat array of global variant indexes.
    Buckets into chunks, producing one :class:`~vcztools.utils.ChunkRead`
    per visited chunk with a real (non-None) local selection."""
    variant_indexes = np.asarray(variant_indexes, dtype=np.int64)
    chunk_of_each = variant_indexes // variants_chunk_size
    chunk_indexes = np.unique(chunk_of_each)
    return [
        utils.ChunkRead(
            index=int(ci),
            selection=variant_indexes[chunk_of_each == ci] - (ci * variants_chunk_size),
        )
        for ci in chunk_indexes
    ]


def build_chunk_plan(
    root,
    *,
    regions=None,
    targets=None,
    targets_complement: bool = False,
) -> list[utils.ChunkRead]:
    """Build a list of :class:`~vcztools.utils.ChunkRead` from
    region/target inputs and the Zarr root. When neither ``regions``
    nor ``targets`` applies a filter, returns one
    ``ChunkRead(index=i, selection=None)`` per chunk — meaning "read
    every chunk in full".

    Uses the ``region_index`` array to prune candidate chunks (only
    when ``regions`` is set — ``targets`` alone doesn't prune), then
    scans each candidate chunk once to compute a local row selection.
    Only non-empty chunks end up in the plan.
    """
    position_arr = root["variant_position"]
    num_chunks = int(position_arr.cdata_shape[0])

    regions_df = _regions_input_to_df(regions, arg_name="regions")
    targets_df = _regions_input_to_df(targets, arg_name="targets")

    contigs_u = _as_fixed_length_unicode(root["contig_id"][:]).tolist()
    regions_gr = dataframe_to_ranges(regions_df, contigs_u)
    targets_gr = dataframe_to_ranges(
        targets_df, contigs_u, complement=targets_complement
    )

    if regions_gr is None and targets_gr is None:
        return [utils.ChunkRead(index=i) for i in range(num_chunks)]

    contig_arr = root["variant_contig"]
    length_arr = root["variant_length"]

    if regions_gr is not None:
        region_index = root["region_index"][:]
        candidate_chunks = regions_to_chunk_indexes(regions_gr, region_index)
    else:
        candidate_chunks = np.arange(num_chunks, dtype=np.int64)

    plan = []
    for chunk_idx in candidate_chunks:
        chunk_idx = int(chunk_idx)
        vc = contig_arr.blocks[chunk_idx]
        vp = position_arr.blocks[chunk_idx]
        vl = length_arr.blocks[chunk_idx]
        local_sel = regions_to_selection(regions_gr, targets_gr, vc, vp, vl)
        if local_sel is None:
            continue
        local_sel = np.asarray(local_sel, dtype=np.int64)
        if local_sel.size == 0:
            continue
        plan.append(utils.ChunkRead(index=chunk_idx, selection=local_sel))

    return plan
