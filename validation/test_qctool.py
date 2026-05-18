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


def _strip_padding(values: np.ndarray) -> np.ndarray:
    """Strip the encoder's padding fill from a column read back by
    qctool. See :func:`validation.test_bgenix._strip_padding` for the
    rationale.
    """
    return np.array([s.lstrip(".") for s in values], dtype=object)


class TestQctoolVariantIds:
    """Variant IDs round-trip through qctool's ``rsid`` column when the
    BGEN was encoded with ``variant_id_field="rsid"`` (the default), or
    through ``alternate_ids`` when ``variant_id_field="varid"``. The
    unused slot is the padding field — ``"."`` for ``write_bgen`` or
    ``"." + pad_byte * (slack - 1)`` for ``BgenEncoder``. Stripping
    leading ``"."`` characters collapses both encodings to the empty
    string so the same assertion applies.
    """

    @pytest.mark.parametrize("level", cfg.BGEN_LEVELS)
    def test_default_field_routes_variant_id_to_rsid(
        self, tmp_path, qctool_bin, small_unphased_fixture, level
    ):
        bgen, sample = cfg.bgen_for_level(small_unphased_fixture, level)
        out = tmp_path / "snp_stats.tsv"
        df = _qctool_snp_stats(qctool_bin, bgen, sample, out)
        ref = reference.compute_variant_stats(small_unphased_fixture.vcz_path)
        biallelic = ref.n_alleles == 2
        ids = reference.variant_ids(small_unphased_fixture.vcz_path)[biallelic]
        np.testing.assert_array_equal(df["rsid"].astype(str).to_numpy(), ids)
        stripped = _strip_padding(df["alternate_ids"].astype(str).to_numpy())
        np.testing.assert_array_equal(stripped, np.array([""] * len(ids), dtype=object))


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


class TestQctoolStringVariety:
    """Round-trip every BGEN string field through qctool on the
    ``varied_strings`` fixture, where chrom / alleles / variant_id all
    vary in byte length across the variants axis. Mirrors the bgenix
    coverage in :class:`validation.test_bgenix.TestBgenixStringVariety`.
    """

    def _carrier_column(self, variant_id_field: str) -> str:
        return "alternate_ids" if variant_id_field == "varid" else "rsid"

    def _padding_column(self, variant_id_field: str) -> str:
        return "rsid" if variant_id_field == "varid" else "alternate_ids"

    @pytest.mark.parametrize("variant_id_field", cfg.VARIANT_ID_FIELDS)
    @pytest.mark.parametrize("level", cfg.BGEN_LEVELS)
    def test_chrom_and_alleles_round_trip(
        self, tmp_path, qctool_bin, varied_strings_fixture, level, variant_id_field
    ):
        bgen, sample = cfg.bgen_for_field(
            varied_strings_fixture, level, variant_id_field
        )
        out = tmp_path / "snp_stats.tsv"
        df = _qctool_snp_stats(qctool_bin, bgen, sample, out)
        ref = reference.compute_variant_stats(varied_strings_fixture.vcz_path)
        biallelic = ref.n_alleles == 2
        assert len(df) == int(biallelic.sum())
        np.testing.assert_array_equal(
            df["chromosome"].astype(str).to_numpy(), ref.chrom[biallelic]
        )
        np.testing.assert_array_equal(
            df["alleleA"].astype(str).to_numpy(), ref.ref[biallelic]
        )
        np.testing.assert_array_equal(
            df["alleleB"].astype(str).to_numpy(), ref.alt[biallelic]
        )
        np.testing.assert_array_equal(df["position"].to_numpy(), ref.pos[biallelic])

    @pytest.mark.parametrize("variant_id_field", cfg.VARIANT_ID_FIELDS)
    @pytest.mark.parametrize("level", cfg.BGEN_LEVELS)
    def test_variant_id_round_trip(
        self, tmp_path, qctool_bin, varied_strings_fixture, level, variant_id_field
    ):
        bgen, sample = cfg.bgen_for_field(
            varied_strings_fixture, level, variant_id_field
        )
        out = tmp_path / "snp_stats.tsv"
        df = _qctool_snp_stats(qctool_bin, bgen, sample, out)
        ref = reference.compute_variant_stats(varied_strings_fixture.vcz_path)
        biallelic = ref.n_alleles == 2
        ids = reference.variant_ids(varied_strings_fixture.vcz_path)[biallelic]

        carrier = self._carrier_column(variant_id_field)
        padding = self._padding_column(variant_id_field)
        np.testing.assert_array_equal(df[carrier].astype(str).to_numpy(), ids)
        stripped = _strip_padding(df[padding].astype(str).to_numpy())
        np.testing.assert_array_equal(stripped, np.array([""] * len(ids), dtype=object))


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
