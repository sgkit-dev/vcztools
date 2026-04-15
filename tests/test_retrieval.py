import numpy.testing as nt
import pytest

from vcztools.retrieval import variant_chunk_iter, variant_iter
from vcztools.samples import parse_samples


def test_variant_chunk_iter(fx_sample_vcz):
    root = fx_sample_vcz.group

    _, samples_selection = parse_samples("NA00002,NA00003", root["sample_id"][:])
    chunk_data = next(
        variant_chunk_iter(
            root,
            fields=["variant_contig", "variant_position", "call_DP", "call_GQ"],
            regions="20:1230236-",
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


def test_variant_chunk_iter_empty_fields(fx_sample_vcz):
    with pytest.raises(StopIteration):
        print(next(variant_chunk_iter(fx_sample_vcz.group, fields=[])))


@pytest.mark.parametrize(
    ("regions", "samples"),
    [("20:1230236-", "NA00002,NA00003"), (["20:1230236-"], ["NA00002", "NA00003"])],
)
def test_variant_iter(fx_sample_vcz, regions, samples):
    iter = variant_iter(
        fx_sample_vcz.group,
        fields=["variant_contig", "variant_position", "call_DP", "call_GQ"],
        regions=regions,
        include="FMT/DP>3",
        samples=samples,
    )

    variant1 = next(iter)
    assert variant1["variant_contig"] == 1
    assert variant1["variant_position"] == 1230237
    nt.assert_array_equal(variant1["call_DP"], [4, 2])
    nt.assert_array_equal(variant1["call_GQ"], [48, 61])
    nt.assert_array_equal(variant1["call_mask"], [True, False])

    variant2 = next(iter)
    assert variant2["variant_contig"] == 1
    assert variant2["variant_position"] == 1234567
    nt.assert_array_equal(variant2["call_DP"], [2, 3])
    nt.assert_array_equal(variant2["call_GQ"], [17, 40])
    nt.assert_array_equal(variant2["call_mask"], [False, False])

    with pytest.raises(StopIteration):
        next(iter)


def test_variant_iter_empty_fields(fx_sample_vcz):
    with pytest.raises(StopIteration):
        next(variant_iter(fx_sample_vcz.group, fields=[]))
