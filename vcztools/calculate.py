import numpy as np

from . import _vcztools

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
    """Compute (AC, AN) per variant from a genotype chunk + ALT matrix.

    gt: (V, S, P) int8 genotype array.
    alt: (V, max_num_alt) string ALT-allele matrix (bytes or Unicode).
        Empty-string entries mark padding for variants with fewer ALTs
        than max_num_alt and define each row's allele count.

    Returns (ac, an) with ac.shape == (V, max_num_alt), int32 and
    an.shape == (V,), int32. AC cells beyond a variant's actual ALT
    count come back as constants.INT_FILL.

    Raises ValueError if any genotype value lies outside [-2,
    num_alleles[j]) for its row.
    """
    gt = np.ascontiguousarray(gt, dtype=np.int8)
    # alt may arrive as bytes (``S``) or Unicode (``U`` / ``T`` /
    # ``O``); pick the matching empty literal so the mask works
    # regardless of how the Zarr store represents strings.
    empty = b"" if alt.dtype.kind == "S" else ""
    num_alleles = 1 + (alt != empty).sum(axis=1).astype(np.int32)
    num_alleles = np.ascontiguousarray(num_alleles)
    ac = np.zeros((gt.shape[0], alt.shape[1]), dtype=np.int32)
    an = np.zeros(gt.shape[0], dtype=np.int32)
    _vcztools.compute_ac_an(gt, num_alleles, ac, an)
    return ac, an
