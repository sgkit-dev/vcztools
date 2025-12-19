import numpy as np

from vcztools.utils import _as_fixed_length_unicode, open_file_like, open_zarr


def nrecords(vcz, output, zarr_backend_storage=None):
    root = open_zarr(vcz, mode="r", zarr_backend_storage=zarr_backend_storage)

    with open_file_like(output) as output:
        num_variants = root["variant_position"].shape[0]
        print(num_variants, file=output)


def stats(vcz, output, zarr_backend_storage=None):
    root = open_zarr(vcz, mode="r", zarr_backend_storage=zarr_backend_storage)

    if "region_index" not in root:
        raise ValueError(
            "Could not load 'region_index' variable. "
            "Use 'vcz2zarr' to create an index."
        )

    with open_file_like(output) as output:
        contigs = _as_fixed_length_unicode(root["contig_id"][:]).tolist()
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
