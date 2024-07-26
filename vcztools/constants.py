import numpy as np

INT_MISSING, INT_FILL = -1, -2

FLOAT32_MISSING, FLOAT32_FILL = np.array([0x7F800001, 0x7F800002], dtype=np.int32).view(
    np.float32
)
FLOAT32_MISSING_AS_INT32, FLOAT32_FILL_AS_INT32 = np.array(
    [0x7F800001, 0x7F800002], dtype=np.int32
)

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
