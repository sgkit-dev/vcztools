import numpy as np

from . import _vcztools, constants

# Variant types
REF = -1  # missing value
SNP = 1 << 0
UNCLASSIFIED = 1 << 8


def get_variant_type(ref: str, alt: str) -> int:
    """Return the variant type int for the given REF, ALT combination."""
    if len(alt) == 0:
        return REF
    elif len(ref) == 1 and len(alt) == 1 and alt != "*":
        if ref == alt:
            return REF
        else:
            return SNP
    elif alt == "<*>" or alt == "<NON_REF>":
        return REF
    elif (
        len(ref) > 1
        and len(ref) == len(alt)
        and sum([r != a for r, a in zip(ref, alt)]) == 1  # one base differs
    ):
        return SNP
    else:
        return UNCLASSIFIED


def calculate_variant_type(variant_allele: np.ndarray) -> np.ndarray:
    """Calculate the variant type array from the variant_allele array."""
    ref = variant_allele[:, 0]
    alt = variant_allele[:, 1:]

    variant_type = np.zeros(alt.shape, dtype=np.int16)

    for i in range(alt.shape[0]):
        for j in range(alt.shape[1]):
            variant_type[i, j] = get_variant_type(ref[i], alt[i, j])
    return variant_type


def compute_ac_an(gt: np.ndarray, alt: np.ndarray):
    """Compute (AC, AN) per variant from a genotype chunk + ALT mask.

    gt: (V, S, P) int8 genotype array.
    alt: (V, num_alt_alleles) string ALT-allele matrix (bytes or
        Unicode). Only the empty-string mask is used, to mark padding
        positions in the AC output with INT_FILL.

    Returns (ac, an) with ac.shape == (V, num_alt_alleles), int32 and
    an.shape == (V,), int32.
    """
    gt = np.ascontiguousarray(gt, dtype=np.int8)
    num_alt = alt.shape[1]
    ac = np.zeros((gt.shape[0], num_alt), dtype=np.int32)
    an = np.zeros(gt.shape[0], dtype=np.int32)
    _vcztools.compute_ac_an(gt, ac, an)
    # alt may arrive as bytes (``S``) or Unicode (``U`` / ``T`` /
    # ``O``); pick the matching empty literal so the mask works
    # regardless of how the Zarr store represents strings.
    empty = b"" if alt.dtype.kind == "S" else ""
    ac[alt == empty] = constants.INT_FILL
    return ac, an


def compute_an(gt: np.ndarray) -> np.ndarray:
    """AN-only fast path that skips the per-allele bincount work."""
    gt = np.ascontiguousarray(gt, dtype=np.int8)
    ac = np.zeros((gt.shape[0], 0), dtype=np.int32)
    an = np.zeros(gt.shape[0], dtype=np.int32)
    _vcztools.compute_ac_an(gt, ac, an)
    return an
