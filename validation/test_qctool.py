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
        self, tmp_path, qctool_bin, small_unphased_fixture, level
    ):
        bgen, sample = cfg.bgen_for_level(small_unphased_fixture, level)
        out = tmp_path / "snp_stats.tsv"
        df = _qctool_snp_stats(qctool_bin, bgen, sample, out)

        ref = reference.compute_variant_stats(small_unphased_fixture.vcz_path)
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
        # qctool truncates ``alleleB_frequency`` to ~6 decimal places in
        # the TSV output, so a direct compare against an exact-rational
        # reference fails at fractions like 291/396. Recompute from the
        # integer ``alleleB_count`` and ``NULL`` columns — vcztools
        # writes hard calls (P=1.0 on the called genotype), so the
        # integer count round-trips exactly and the recomputed
        # frequency matches the reference at float64 precision.
        n_called = (2 * (df["total"] - df["NULL"])).to_numpy()
        alt_freq = df["alleleB_count"].to_numpy() / n_called
        np.testing.assert_allclose(
            alt_freq,
            ref.alt_freq[biallelic],
            atol=1e-10,
        )

    @pytest.mark.parametrize("level", cfg.BGEN_LEVELS)
    def test_minor_allele_frequency_matches_reference(
        self, tmp_path, qctool_bin, small_unphased_fixture, level
    ):
        bgen, sample = cfg.bgen_for_level(small_unphased_fixture, level)
        out = tmp_path / "snp_stats.tsv"
        df = _qctool_snp_stats(qctool_bin, bgen, sample, out)

        ref = reference.compute_variant_stats(small_unphased_fixture.vcz_path)
        biallelic = ref.n_alleles == 2
        # Same display-precision dodge as above — recompute MAF from
        # the integer count columns rather than the formatted output.
        n_called = (2 * (df["total"] - df["NULL"])).to_numpy()
        minor_count = np.minimum(
            df["alleleA_count"].to_numpy(), df["alleleB_count"].to_numpy()
        )
        minor_freq = minor_count / n_called
        np.testing.assert_allclose(
            minor_freq,
            ref.minor_freq[biallelic],
            atol=1e-10,
        )

    @pytest.mark.parametrize("level", cfg.BGEN_LEVELS)
    def test_sample_and_variant_counts(
        self, tmp_path, qctool_bin, small_unphased_fixture, level
    ):
        bgen, sample = cfg.bgen_for_level(small_unphased_fixture, level)
        out = tmp_path / "snp_stats.tsv"
        df = _qctool_snp_stats(qctool_bin, bgen, sample, out)
        # Sample count is implicit in the .sample file format; check
        # that the analysis succeeded by inspecting the row count.
        assert len(df) > 0
        # qctool collapses 4 expected genotype-count columns; AA + AB +
        # BB + NULL should sum to the sample count for every row.
        n_samples = len(reference.sample_ids(small_unphased_fixture.vcz_path))
        totals = (df["AA"] + df["AB"] + df["BB"] + df["NULL"]).to_numpy()
        np.testing.assert_array_equal(totals, np.full(len(df), n_samples))


class TestQctoolVariantIds:
    @pytest.mark.parametrize("level", cfg.BGEN_LEVELS)
    def test_alternate_ids_match_reference(
        self, tmp_path, qctool_bin, small_unphased_fixture, level
    ):
        bgen, sample = cfg.bgen_for_level(small_unphased_fixture, level)
        out = tmp_path / "snp_stats.tsv"
        df = _qctool_snp_stats(qctool_bin, bgen, sample, out)
        ref = reference.compute_variant_stats(small_unphased_fixture.vcz_path)
        biallelic = ref.n_alleles == 2
        ids = reference.variant_ids(small_unphased_fixture.vcz_path)[biallelic]
        np.testing.assert_array_equal(df["alternate_ids"].astype(str).to_numpy(), ids)


class TestQctoolHaploid:
    """qctool reads vcztools' uniform-haploid BGEN output and reports the
    ALT-allele frequency we compute from the source. The diploid-suite
    test recomputes ``alt_freq`` from integer columns using
    ``2 * (total - NULL)`` to dodge the formatted-precision rounding;
    that recomputation hardcodes ploidy=2 and is wrong for haploid, so
    we compare the already-formatted ``alleleB_frequency`` column
    directly with an ``atol`` that absorbs qctool's 6-decimal printing.
    """

    @pytest.mark.parametrize("level", cfg.BGEN_LEVELS)
    def test_alt_allele_frequency_matches_reference(
        self, tmp_path, qctool_bin, haploid_fixture, level
    ):
        bgen, sample = cfg.bgen_for_level(haploid_fixture, level)
        out = tmp_path / "snp_stats.tsv"
        df = _qctool_snp_stats(qctool_bin, bgen, sample, out)

        ref = reference.compute_variant_stats(haploid_fixture.vcz_path)
        biallelic = ref.n_alleles == 2
        assert len(df) == int(biallelic.sum())
        np.testing.assert_array_equal(df["position"].to_numpy(), ref.pos[biallelic])
        np.testing.assert_allclose(
            df["alleleB_frequency"].to_numpy(),
            ref.alt_freq[biallelic],
            atol=1e-6,
        )


class TestQctoolMixedPloidy:
    """qctool reads vcztools' mixed-ploidy BGEN output. Only the two
    view-bgen flavours apply: BgenEncoder is uniform-only."""

    @pytest.mark.parametrize("level", cfg.BGEN_CLI_LEVELS)
    def test_alt_allele_frequency_matches_reference(
        self, tmp_path, qctool_bin, mixed_ploidy_fixture, level
    ):
        bgen, sample = cfg.bgen_for_level(mixed_ploidy_fixture, level)
        out = tmp_path / "snp_stats.tsv"
        df = _qctool_snp_stats(qctool_bin, bgen, sample, out)

        ref = reference.compute_variant_stats(mixed_ploidy_fixture.vcz_path)
        biallelic = ref.n_alleles == 2
        assert len(df) == int(biallelic.sum())
        np.testing.assert_array_equal(df["position"].to_numpy(), ref.pos[biallelic])
        np.testing.assert_allclose(
            df["alleleB_frequency"].to_numpy(),
            ref.alt_freq[biallelic],
            atol=1e-6,
        )


class TestQctoolPhasedBgen:
    """qctool refuses vcztools' phased BGEN layout — it raises
    ``genfile::BadArgumentError`` complaining that the
    ``order_type=6, value_type=1`` combination is not supported.
    Pin the diagnostic so a future qctool release that adds phased
    support surfaces here.
    """

    def test_rejects_phased_bgen(self, tmp_path, qctool_bin, small_phased_fixture):
        result = helpers.run_tool(
            [
                str(qctool_bin),
                "-g",
                str(small_phased_fixture.bgen_minus1),
                "-s",
                str(small_phased_fixture.sample_path),
                "-snp-stats",
                "-osnp",
                str(tmp_path / "snp_stats.tsv"),
            ],
            check=False,
        )
        # qctool reports the error on stdout, then exits non-zero.
        combined = result.stdout + result.stderr
        tail = f"stdout+stderr tail:\n{combined[-2000:]}"
        assert "order_type=6" in combined, (
            f"qctool didn't print the order_type=6 marker; {tail}"
        )
        assert "value_type=1" in combined, (
            f"qctool didn't print the value_type=1 marker; {tail}"
        )
