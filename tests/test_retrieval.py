import pathlib

import numpy.testing as nt
import zarr

from vcztools.retrieval import variant_chunk_iter
from vcztools.samples import parse_samples

from .utils import vcz_path_cache


def test_variant_chunk_iter():
    original = pathlib.Path("tests/data/vcf") / "sample.vcf.gz"
    vcz = vcz_path_cache(original)
    root = zarr.open(vcz, mode="r")

    _, samples_selection = parse_samples("NA00002,NA00003", root["sample_id"][:])
    chunk_data = next(
        variant_chunk_iter(
            root,
            fields=["variant_contig", "variant_position", "call_DP", "call_GQ"],
            variant_regions="20:1230236-",
            include="FMT/DP>3",
            samples_selection=samples_selection,
        )
    )
    nt.assert_array_equal(chunk_data["variant_contig"], [1, 1])
    nt.assert_array_equal(chunk_data["variant_position"], [1230237, 1234567])
    nt.assert_array_equal(chunk_data["call_DP"], [[4, 2], [2, 3]])
    nt.assert_array_equal(chunk_data["call_GQ"], [[48, 61], [17, 40]])
    # note second site (at pos 1234567) is included even though both samples in mask
    # are False (NA00002 and NA00003), since sample NA00001 matched filter criteria,
    # but was then removed by samples_selection
    nt.assert_array_equal(chunk_data["call_mask"], [[True, False], [False, False]])
