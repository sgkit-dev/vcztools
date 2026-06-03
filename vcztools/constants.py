import numpy as np

INT_MISSING, INT_FILL = -1, -2
STR_MISSING = "."
STR_FILL = ""

FLOAT32_MISSING, FLOAT32_FILL = np.array([0x7F800001, 0x7F800002], dtype=np.int32).view(
    np.float32
)
FLOAT32_MISSING_AS_INT32, FLOAT32_FILL_AS_INT32 = np.array(
    [0x7F800001, 0x7F800002], dtype=np.int32
)

# Missing / end-of-vector sentinels for non-float32 widths, generalising the
# float32 patterns above: exponent all-ones, mantissa low bits 1 (missing) and
# 2 (fill).
FLOAT16_MISSING, FLOAT16_FILL = np.array([0x7C01, 0x7C02], dtype=np.int16).view(
    np.float16
)
FLOAT64_MISSING, FLOAT64_FILL = np.array(
    [0x7FF0000000000001, 0x7FF0000000000002], dtype=np.int64
).view(np.float64)

# From VCF fixed fields
RESERVED_VARIABLE_NAMES = [
    "variant_contig",
    "variant_position",
    "variant_length",
    "variant_id",
    "variant_id_mask",
    "variant_allele",
    "variant_quality",
    "variant_filter",
]

RESERVED_VCF_FIELDS = {
    "CHROM": "variant_contig",
    "POS": "variant_position",
    "ID": "variant_id",
    "REF": "variant_allele",
    "ALT": "variant_allele",
    "QUAL": "variant_quality",
    "FILTER": "variant_filter",
}
