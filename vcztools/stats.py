import zarr

from vcztools.utils import open_file_like


def nrecords(vcz, output):
    root = zarr.open(vcz, mode="r")

    with open_file_like(output) as output:
        num_variants = root["variant_position"].shape[0]
        print(num_variants, file=output)
