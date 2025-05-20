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
import tskit
import xarray.testing as xt

from vcztools.plink import write_plink
from vcztools.vcf_writer import write_vcf


def add_mutations(ts):
    # Add some mutation to the tree sequence. This guarantees that
    # we have variation at all sites > 0.
    tables = ts.dump_tables()
    samples = ts.samples()
    states = "ACGT"
    for j in range(1, int(ts.sequence_length) - 1):
        site = tables.sites.add_row(j, ancestral_state=states[j % 4])
        tables.mutations.add_row(
            site=site,
            derived_state=states[(j + 1) % 4],
            node=samples[j % ts.num_samples],
        )
    return tables.tree_sequence()


@pytest.fixture()
def fx_diploid_msprime_sim(tmp_path):
    seed = 1234
    ts = msprime.sim_ancestry(5, sequence_length=10_000, random_seed=seed)
    ts = msprime.sim_mutations(ts, rate=1e-4, random_seed=seed)
    assert ts.num_mutations > 0
    assert ts.num_mutations == ts.num_sites  # make sure we have biallelic sites
    zarr_path = tmp_path / "sim.vcz"
    ts2z.convert(ts, zarr_path)
    return zarr_path


@pytest.fixture()
def fx_haploid_missing_data(tmp_path):
    # 2.00┊   4     ┊
    #     ┊ ┏━┻┓    ┊
    # 1.00┊ ┃  3    ┊
    #     ┊ ┃ ┏┻┓   ┊
    # 0.00┊ 0 1 2 5 ┊
    #     0        10
    #      |      |
    #  pos 2      9
    #  anc A      T
    ts = tskit.Tree.generate_balanced(3, span=10).tree_sequence
    tables = ts.dump_tables()
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    tables.sites.add_row(2, ancestral_state="A")
    tables.sites.add_row(9, ancestral_state="T")
    tables.mutations.add_row(site=0, node=0, derived_state="G")
    tables.mutations.add_row(site=1, node=3, derived_state="C")
    zarr_path = tmp_path / "sim.vcz"
    ts2z.convert(tables.tree_sequence(), zarr_path, isolated_as_missing=True)
    return zarr_path


def test_haploid_missing_data(fx_haploid_missing_data):
    ds = sg.load_dataset(fx_haploid_missing_data)
    nt.assert_array_equal(
        ds.call_genotype.values,
        [
            [[1], [0], [0], [-1]],
            [[0], [1], [1], [-1]],
        ],
    )


@pytest.fixture()
def fx_diploid_missing_data(tmp_path):
    # 2.00┊    6       ┊
    #     ┊  ┏━┻━┓     ┊
    # 1.00┊  4   5     ┊
    #     ┊ ┏┻┓ ┏┻┓    ┊
    # 0.00┊ 0 1 2 3 7 8┊
    #     0            10
    #      |         |
    #  pos 2         9
    #  anc A         T
    ts = tskit.Tree.generate_balanced(4, span=10).tree_sequence
    tables = ts.dump_tables()
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    u = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    assert u == 8
    tables.sites.add_row(2, ancestral_state="A")
    tables.sites.add_row(9, ancestral_state="T")
    tables.mutations.add_row(site=0, node=0, derived_state="G")
    tables.mutations.add_row(site=1, node=5, derived_state="C")
    zarr_path = tmp_path / "sim.vcz"
    ts = tables.tree_sequence()
    model_map = ts.map_to_vcf_model(ploidy=2)
    ts2z.convert(ts, zarr_path, model_mapping=model_map, isolated_as_missing=True)
    return zarr_path


def test_diploid_missing_data(fx_diploid_missing_data):
    ds = sg.load_dataset(fx_diploid_missing_data)
    nt.assert_array_equal(
        ds.call_genotype.values,
        [
            [[1, 0], [0, 0], [-1, -1]],
            [[0, 0], [1, 1], [-1, -1]],
        ],
    )


@pytest.fixture()
def fx_diploid_multi_allelic(tmp_path):
    # 2.00┊    6    ┊
    #     ┊  ┏━┻━┓  ┊
    # 1.00┊  4   5  ┊
    #     ┊ ┏┻┓ ┏┻┓ ┊
    # 0.00┊ 0 1 2 3 ┊
    #     0         10
    #      |       |
    #  pos 2       9
    #  anc A       T
    ts = tskit.Tree.generate_balanced(4, span=10).tree_sequence
    tables = ts.dump_tables()
    tables.sites.add_row(2, ancestral_state="A")
    tables.sites.add_row(9, ancestral_state="T")
    tables.mutations.add_row(site=0, node=0, derived_state="G")
    tables.mutations.add_row(site=1, node=1, derived_state="G")
    tables.mutations.add_row(site=1, node=5, derived_state="C")
    zarr_path = tmp_path / "sim.vcz"
    ts = tables.tree_sequence()
    model_map = ts.map_to_vcf_model(ploidy=2)
    ts2z.convert(ts, zarr_path, model_mapping=model_map)
    return zarr_path


def test_diploid_multi_allelic(fx_diploid_multi_allelic):
    ds = sg.load_dataset(fx_diploid_multi_allelic)
    # NOTE this example is constructed so that the rarest allele is in the middle
    # of the alleles array
    nt.assert_array_equal(ds.variant_allele.values, [["A", "G", ""], ["T", "G", "C"]])
    nt.assert_array_equal(
        ds.call_genotype.values,
        [
            [[1, 0], [0, 0]],
            [[0, 1], [2, 2]],
        ],
    )


@pytest.fixture()
def fx_haploid_msprime_sim(tmp_path):
    seed = 12345
    ts = msprime.sim_ancestry(5, ploidy=1, sequence_length=100, random_seed=seed)
    ts = msprime.sim_mutations(ts, rate=0.5, random_seed=seed)
    assert ts.num_mutations > 0
    zarr_path = tmp_path / "sim.vcz"
    ts2z.convert(ts, zarr_path)
    return zarr_path


def simple_ts_tables():
    tables = tskit.TableCollection(sequence_length=100)
    for _ in range(4):
        ind = -1
        ind = tables.individuals.add_row()
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=ind)
    tables.nodes.add_row(flags=0, time=1)  # MRCA for 0,1
    tables.nodes.add_row(flags=0, time=1)  # MRCA for 2,3
    tables.edges.add_row(left=0, right=100, parent=4, child=0)
    tables.edges.add_row(left=0, right=100, parent=4, child=1)
    tables.edges.add_row(left=0, right=100, parent=5, child=2)
    tables.edges.add_row(left=0, right=100, parent=5, child=3)
    site_id = tables.sites.add_row(position=10, ancestral_state="A")
    tables.mutations.add_row(site=site_id, node=4, derived_state="TTTT")
    site_id = tables.sites.add_row(position=20, ancestral_state="CCC")
    tables.mutations.add_row(site=site_id, node=5, derived_state="G")
    site_id = tables.sites.add_row(position=30, ancestral_state="G")
    tables.mutations.add_row(site=site_id, node=0, derived_state="AA")

    tables.sort()
    return tables


@pytest.fixture()
def fx_simple_ts(tmp_path):
    ts = simple_ts_tables().tree_sequence()
    zarr_path = tmp_path / "sim.vcz"
    ts2z.convert(ts, zarr_path)
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
        num_variants = ds2.sizes["variants"]
        assert np.all(np.isnan(ds2["variant_quality"].values))
        nt.assert_array_equal(
            ds2["variant_filter"], np.ones((num_variants, 1), dtype=bool)
        )
        assert list(ds2["filter_id"].values) == ["PASS"]

    def test_diploid_msprime_sim(self, tmp_path, fx_diploid_msprime_sim):
        self.assert_bio2zarr_rt(tmp_path, fx_diploid_msprime_sim)

    def test_haploid_msprime_sim(self, tmp_path, fx_haploid_msprime_sim):
        self.assert_bio2zarr_rt(tmp_path, fx_haploid_msprime_sim)

    def test_simple_ts(self, tmp_path, fx_simple_ts):
        self.assert_bio2zarr_rt(tmp_path, fx_simple_ts)

    def test_haploid_missing_data(self, tmp_path, fx_haploid_missing_data):
        self.assert_bio2zarr_rt(tmp_path, fx_haploid_missing_data)

    def test_diploid_missing_data(self, tmp_path, fx_diploid_missing_data):
        self.assert_bio2zarr_rt(tmp_path, fx_diploid_missing_data)

    def test_diploid_multi_allelic(self, tmp_path, fx_diploid_multi_allelic):
        self.assert_bio2zarr_rt(tmp_path, fx_diploid_multi_allelic)


def recode_plink_hets(G):
    """
    Returns a copy of the specified genotype matrix in which hets are all
    in the canonical unphased plink orientation, [0, 1]
    """
    G = G.copy()
    for j in range(G.shape[0]):
        for k in range(G.shape[1]):
            if G[j, k, 0] == 1 and G[j, k, 1] == 0:
                G[j, k, 0] = 0
                G[j, k, 1] = 1
    return G


class TestPlinkRoundTrip:
    def assert_bio2zarr_rt(self, tmp_path, tskit_vcz):
        # import pathlib
        # tmp_path = pathlib.Path("tmp/plink")
        plink_path = tmp_path / "plink"
        write_plink(tskit_vcz, plink_path)
        rt_vcz_path = tmp_path / "rt.vcz"
        p2z.convert(plink_path, rt_vcz_path)
        ds1 = sg.load_dataset(tskit_vcz)
        ds2 = sg.load_dataset(rt_vcz_path)

        assert np.all(ds1["call_genotype_phased"])
        assert np.all(~ds2["call_genotype_phased"])

        nt.assert_array_equal(
            recode_plink_hets(ds1["call_genotype"].values), ds2["call_genotype"]
        )

        drop_fields = [
            "variant_id",
            "variant_id_mask",
            "call_genotype",
            "call_genotype_phased",
        ]
        xt.assert_equal(
            ds1.drop_vars(["call_genotype", "call_genotype_phased"]),
            ds2.drop_vars(drop_fields),
        )

    def test_diploid_msprime_sim(self, tmp_path, fx_diploid_msprime_sim):
        self.assert_bio2zarr_rt(tmp_path, fx_diploid_msprime_sim)

    def test_diploid_missing_data(self, tmp_path, fx_diploid_missing_data):
        self.assert_bio2zarr_rt(tmp_path, fx_diploid_missing_data)

    def test_diploid_multi_allelic(self, tmp_path, fx_diploid_multi_allelic):
        with pytest.raises(ValueError, match="Only biallelic VCFs supported"):
            self.assert_bio2zarr_rt(tmp_path, fx_diploid_multi_allelic)
