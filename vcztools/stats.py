import numpy as np
import zarr

from vcztools.utils import open_file_like


def nrecords(vcz, output):
    root = zarr.open(vcz, mode="r")

    with open_file_like(output) as output:
        num_variants = root["variant_position"].shape[0]
        print(num_variants, file=output)


def stats(vcz, output):
    root = zarr.open(vcz, mode="r")

    if "region_index" not in root:
        raise ValueError(
            "Could not load 'region_index' variable. "
            "Use 'vcz2zarr' to create an index."
        )

    with open_file_like(output) as output:
        contigs = root["contig_id"][:].astype("U").tolist()
        if "contig_length" in root:
            contig_lengths = root["contig_length"][:]
        else:
            contig_lengths = ["."] * len(contigs)

        region_index = root["region_index"][:]

        contig_indexes = region_index[:, 1]
        num_records = region_index[:, 5]

        num_records_per_contig = np.bincount(
            contig_indexes, weights=num_records
        ).astype(np.int64)

        for contig, contig_length, nr in zip(
            contigs, contig_lengths, num_records_per_contig
        ):
            if nr > 0:
                print(f"{contig}\t{contig_length}\t{nr}", file=output)
