"""
Convert VCZ to plink 1 binary format.
"""

import numpy as np
import pandas as pd
import zarr


def generate_fam(root):
    sample_id = root["sample_id"][:].astype(str)
    zeros = np.zeros(sample_id.shape, dtype=int)
    df = pd.DataFrame(
        {
            "FamilyID": sample_id,
            "IndividualID": sample_id,
            "FatherID": zeros,
            "MotherId": zeros,
            "Sex": zeros,
            "Phenotype": np.full_like(zeros, -9),
        }
    )
    return df.to_csv(sep="\t", header=False, index=False)


class Writer:
    def __init__(self, vcz_path, bed_path, fam_path, bim_path):
        root = zarr.open(vcz_path, mode="r")
        with open(fam_path, "w") as f:
            f.write(generate_fam(root))
