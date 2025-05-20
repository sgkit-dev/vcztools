"""
Tests for data originating from tskit format for compatibility
with various outputs.
"""

import bio2zarr.plink as p2z
import bio2zarr.tskit as ts2z
import bio2zarr.vcf as v2z
import msprime
import numpy as np
import numpy.testing as nt
import pytest
import sgkit as sg
import xarray.testing as xt

from vcztools.plink import write_plink
from vcztools.vcf_writer import write_vcf


@pytest.fixture()
def fx_msprime_sim(tmp_path):
    seed = 1234
    ts = msprime.sim_ancestry(5, sequence_length=100, random_seed=seed)
    ts = msprime.sim_mutations(ts, rate=0.5, random_seed=seed)
    assert ts.num_mutations > 0
    ts_path = tmp_path / "sim.ts"
    zarr_path = tmp_path / "sim.vcz"
    ts.dump(ts_path)
    ts2z.convert(ts_path, zarr_path)
    return zarr_path


# TODO add other fixtures here like stuff with odd mixtures of ploidy,
# and zero variants (need to address
# https://github.com/sgkit-dev/bio2zarr/issues/342 before zero variants
# handled)


class TestVcfRoundTrip:
    def assert_bio2zarr_rt(self, tmp_path, tskit_vcz):
        vcf_path = tmp_path / "out.vcf"
        write_vcf(tskit_vcz, vcf_path)
        rt_vcz_path = tmp_path / "rt.vcz"
        v2z.convert([vcf_path], rt_vcz_path)
        ds1 = sg.load_dataset(tskit_vcz)
        ds2 = sg.load_dataset(rt_vcz_path)
        drop_fields = [
            "variant_id",
            "variant_id_mask",
            "filter_id",
            "filter_description",
            "variant_filter",
            "variant_quality",
        ]
        xt.assert_equal(ds1, ds2.drop_vars(drop_fields))
        num_variants = ds2.dims["variants"]
        assert np.all(np.isnan(ds2["variant_quality"].values))
        nt.assert_array_equal(
            ds2["variant_filter"], np.ones((num_variants, 1), dtype=bool)
        )
        assert list(ds2["filter_id"].values) == ["PASS"]

    def test_msprime_sim(self, tmp_path, fx_msprime_sim):
        self.assert_bio2zarr_rt(tmp_path, fx_msprime_sim)


class TestPlinkRoundTrip:
    def assert_bio2zarr_rt(self, tmp_path, tskit_vcz):
        # import pathlib

        # tmp_path = pathlib.Path("tmp/plink")
        plink_path = tmp_path / "plink"
        print("plink_path", plink_path)
        write_plink(tskit_vcz, plink_path)
        print("Write plink done")
        rt_vcz_path = tmp_path / "rt.vcz"
        p2z.convert(plink_path.with_suffix(".bed"), rt_vcz_path)
        ds1 = sg.load_dataset(tskit_vcz)
        ds2 = sg.load_dataset(rt_vcz_path)
        print(ds1)
        print(ds2)

        drop_fields = [
            # "variant_id",
            # "variant_id_mask",
            # "filter_id",
            # "filter_description",
            # "variant_filter",
            # "variant_quality",
        ]
        xt.assert_equal(ds1, ds2.drop_vars(drop_fields))
        # num_variants = ds2.dims["variants"]
        # assert np.all(np.isnan(ds2["variant_quality"].values))
        # nt.assert_array_equal(
        #     ds2["variant_filter"], np.ones((num_variants, 1), dtype=bool)
        # )
        # assert list(ds2["filter_id"].values) == ["PASS"]

    def test_msprime_sim(self, tmp_path, fx_msprime_sim):
        self.assert_bio2zarr_rt(tmp_path, fx_msprime_sim)
