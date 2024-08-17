import zarr

from vcztools.utils import open_file_like


def list_samples(vcz_path, output=None):
    root = zarr.open(vcz_path, mode="r")

    with open_file_like(output) as output:
        sample_ids = root["sample_id"][:]
        print("\n".join(sample_ids), file=output)
