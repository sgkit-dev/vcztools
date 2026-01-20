import numpy as np

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


SUPPORTED_CALCULATED_VARIABLES = ["N_MISSING", "F_MISSING"]
UNSUPPORTED_CALCULATED_VARIABLES = [
    "N_ALT",
    "N_SAMPLES",
    "AC",
    "MAC",
    "AF",
    "MAF",
    "AN",
    "ILEN",
]


def _n_missing(data: dict[str, np.ndarray]) -> np.ndarray:
    gt = data.get("call_genotype", None)
    if gt is None:
        return np.zeros_like(data["variant_position"])
    else:
        return np.sum(np.all(gt < 0, axis=-1), axis=1)


def _f_missing(data: dict[str, np.ndarray]) -> np.ndarray:
    calculate_if_absent(data, "variant_N_MISSING")
    gt = data["call_genotype"]
    n_samples = gt.shape[1]
    n_missing = data["variant_N_MISSING"]
    return n_missing / n_samples


field_to_function = {
    "variant_N_MISSING": _n_missing,
    "variant_F_MISSING": _f_missing,
}


def calculate_if_absent(data: dict[str, np.ndarray], vcz_name: str) -> None:
    """Populates the array for 'vcz_name' in 'data' if not already present."""
    if vcz_name not in data:
        func = field_to_function[vcz_name]
        data[vcz_name] = func(data)
