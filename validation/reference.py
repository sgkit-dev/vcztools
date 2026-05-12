"""Ground-truth statistics computed directly from a VCZ store.

The downstream-tool validation suite compares each tool's reported
allele frequencies (and similar per-variant statistics) against the
numbers here. Computing them from the source VCZ rather than from the
round-tripped PLINK/BGEN avoids the obvious tautology of asking a
tool to validate against its own input.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import pandas as pd
import zarr


@dataclasses.dataclass(frozen=True)
class VariantStats:
    """Per-variant ALT statistics from the source VCZ."""

    chrom: np.ndarray  # (n_variants,) str
    pos: np.ndarray  # (n_variants,) int
    ref: np.ndarray  # (n_variants,) str
    alt: np.ndarray  # (n_variants,) str (first ALT only)
    alt_freq: np.ndarray  # (n_variants,) float — ALT allele frequency
    minor_freq: np.ndarray  # (n_variants,) float — MAF (min(alt_freq, 1-alt_freq))
    n_called: np.ndarray  # (n_variants,) int — non-missing allele count
    n_alleles: np.ndarray  # (n_variants,) int — count of non-empty alleles

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "chrom": self.chrom,
                "pos": self.pos,
                "ref": self.ref,
                "alt": self.alt,
                "alt_freq": self.alt_freq,
                "minor_freq": self.minor_freq,
                "n_called": self.n_called,
            }
        )


def compute_variant_stats(vcz_path: str | pathlib.Path) -> VariantStats:
    """Compute per-variant ALT allele frequency from a VCZ store.

    Only the first ALT is considered: validation fixtures are biallelic
    (msprime mutations are single-base substitutions), so this is the
    full picture. Missing genotype entries (negative values in
    ``call_genotype``) are excluded from both the numerator and
    denominator, matching what PLINK / qctool / BGEN tools compute.
    """
    group = zarr.open_group(str(vcz_path), mode="r")
    contig_id = group["contig_id"][...]
    variant_contig = group["variant_contig"][...]
    pos = group["variant_position"][...]
    alleles = group["variant_allele"][...]
    gt = group["call_genotype"][...]

    chrom = np.array([str(c) for c in contig_id[variant_contig]])
    ref = np.array([str(a) for a in alleles[:, 0]])
    if alleles.shape[1] >= 2:
        alt = np.array([str(a) for a in alleles[:, 1]])
    else:
        alt = np.array(["."] * len(pos))
    # n_alleles counts how many entries on the row are non-empty
    # strings. msprime fixtures generate occasional tri-allelic sites
    # via recurrent mutation; downstream tools drop those when fed a
    # BGEN/PLINK fileset built with --max-alleles 2.
    alleles_str = np.array([[str(c) for c in row] for row in alleles], dtype=object)
    n_alleles_per_row = (alleles_str != "").sum(axis=1)

    # ALT-allele dosage per (variant, sample): count of nonzero alleles,
    # excluding missing (-1) entries.
    missing = gt < 0
    alt_count_per_call = np.where(missing, 0, gt > 0).sum(axis=-1)
    n_called_per_call = (~missing).sum(axis=-1)
    n_called = n_called_per_call.sum(axis=1)
    alt_count = alt_count_per_call.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        alt_freq = np.where(n_called > 0, alt_count / n_called, np.nan)
    minor_freq = np.minimum(alt_freq, 1.0 - alt_freq)
    return VariantStats(
        chrom=chrom,
        pos=pos.astype(np.int64),
        ref=ref,
        alt=alt,
        alt_freq=alt_freq.astype(np.float64),
        minor_freq=minor_freq.astype(np.float64),
        n_called=n_called.astype(np.int64),
        n_alleles=n_alleles_per_row.astype(np.int64),
    )


def sample_ids(vcz_path: str | pathlib.Path) -> list[str]:
    group = zarr.open_group(str(vcz_path), mode="r")
    return [str(s) for s in group["sample_id"][...]]


def variant_ids(vcz_path: str | pathlib.Path) -> np.ndarray:
    """Return the per-variant ``variant_id`` array as Python ``str`` values.

    Raises ``KeyError`` if the store has no ``variant_id`` field —
    fixtures used by the round-trip checks must have one injected by
    ``generate_data.py``.
    """
    group = zarr.open_group(str(vcz_path), mode="r")
    raw = group["variant_id"][...]
    return np.array([str(v) for v in raw])
