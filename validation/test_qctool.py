"""qctool v2 reads vcztools view-bgen output and reports the same
per-variant statistics we compute from the source VCZ.

For each of the three BGEN flavours (CLI lvl=-1, CLI lvl=0,
``BgenEncoder``) we run ``qctool -snp-stats``, parse the TSV, and
assert ``alleleB_frequency`` (qctool's ALT-allele frequency) matches
``reference.compute_variant_stats`` within an 8-bit BGEN probability
quantisation tolerance. A separate class checks that variant IDs
round-trip through the ``alternate_ids`` column.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from . import conftest as cfg
from . import helpers, reference


def _qctool_snp_stats(
    qctool: pathlib.Path,
    bgen: pathlib.Path,
    sample: pathlib.Path,
    out: pathlib.Path,
) -> pd.DataFrame:
    helpers.run_tool(
        [
            str(qctool),
            "-g",
            str(bgen),
            "-s",
            str(sample),
            "-snp-stats",
            "-osnp",
            str(out),
        ]
    )
    # qctool TSVs are header-commented with "#"; the column header
    # line itself starts with "alternate_ids".
    return pd.read_csv(out, sep="\t", comment="#")


class TestQctoolSnpStats:
    @pytest.mark.parametrize("level", cfg.BGEN_LEVELS)
    def test_alt_allele_frequency_matches_reference(
        self, tmp_path, qctool_bin, small_fixture, level
    ):
        bgen, sample = cfg.bgen_for_level(small_fixture, level)
        out = tmp_path / "snp_stats.tsv"
        df = _qctool_snp_stats(qctool_bin, bgen, sample, out)

        ref = reference.compute_variant_stats(small_fixture.vcz_path)
        # The fixture is biallelic-filtered (--max-alleles 2 / equivalent
        # N_ALT<=1 in the encoder path), so qctool's row order over the
        # BGEN is the same biallelic-survivor order as the reference
        # after the same filter is applied.
        biallelic = ref.n_alleles == 2
        assert len(df) == int(biallelic.sum()), (
            f"qctool emitted {len(df)} rows, reference has "
            f"{int(biallelic.sum())} biallelic variants"
        )
        # Positions line up after the filter.
        np.testing.assert_array_equal(
            df["position"].to_numpy(),
            ref.pos[biallelic],
        )
        # 8 bits/probability quantisation is 1/255 ≈ 4e-3 per sample;
        # frequency averaged over n samples shrinks roughly as
        # 1/sqrt(n). atol=5e-3 is comfortably above that for n=200.
        np.testing.assert_allclose(
            df["alleleB_frequency"].to_numpy(),
            ref.alt_freq[biallelic],
            atol=5e-3,
        )

    @pytest.mark.parametrize("level", cfg.BGEN_LEVELS)
    def test_minor_allele_frequency_matches_reference(
        self, tmp_path, qctool_bin, small_fixture, level
    ):
        bgen, sample = cfg.bgen_for_level(small_fixture, level)
        out = tmp_path / "snp_stats.tsv"
        df = _qctool_snp_stats(qctool_bin, bgen, sample, out)

        ref = reference.compute_variant_stats(small_fixture.vcz_path)
        biallelic = ref.n_alleles == 2
        np.testing.assert_allclose(
            df["minor_allele_frequency"].to_numpy(),
            ref.minor_freq[biallelic],
            atol=5e-3,
        )

    @pytest.mark.parametrize("level", cfg.BGEN_LEVELS)
    def test_sample_and_variant_counts(
        self, tmp_path, qctool_bin, small_fixture, level
    ):
        bgen, sample = cfg.bgen_for_level(small_fixture, level)
        out = tmp_path / "snp_stats.tsv"
        df = _qctool_snp_stats(qctool_bin, bgen, sample, out)
        # Sample count is implicit in the .sample file format; check
        # that the analysis succeeded by inspecting the row count.
        assert len(df) > 0
        # qctool collapses 4 expected genotype-count columns; AA + AB +
        # BB + NULL should sum to the sample count for every row.
        n_samples = len(reference.sample_ids(small_fixture.vcz_path))
        totals = (df["AA"] + df["AB"] + df["BB"] + df["NULL"]).to_numpy()
        np.testing.assert_array_equal(totals, np.full(len(df), n_samples))


class TestQctoolVariantIds:
    @pytest.mark.parametrize("level", cfg.BGEN_LEVELS)
    def test_alternate_ids_match_reference(
        self, tmp_path, qctool_bin, small_fixture, level
    ):
        bgen, sample = cfg.bgen_for_level(small_fixture, level)
        out = tmp_path / "snp_stats.tsv"
        df = _qctool_snp_stats(qctool_bin, bgen, sample, out)
        ref = reference.compute_variant_stats(small_fixture.vcz_path)
        biallelic = ref.n_alleles == 2
        ids = reference.variant_ids(small_fixture.vcz_path)[biallelic]
        np.testing.assert_array_equal(df["alternate_ids"].astype(str).to_numpy(), ids)
