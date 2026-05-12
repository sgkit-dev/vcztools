"""PLINK 1.9 reads vcztools view-plink output and reports the same
per-variant minor allele frequency we compute from the source VCZ.

Only PLINK input is exercised: PLINK 1.9 does not support BGEN v1.2
(our layout-2 output) and refuses to open the file with a message
pointing users at qctool / PLINK 2.0. That refusal is itself a useful
"loads the file" signal, pinned by :func:`TestPlink19RejectsBgenV12`.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from . import conftest as cfg
from . import helpers, reference


def _plink_freq(
    plink_bin: pathlib.Path,
    prefix: pathlib.Path,
    out: pathlib.Path,
) -> pd.DataFrame:
    """Run ``plink --bfile <prefix> --freq`` and return the parsed
    ``.frq`` table."""
    helpers.run_tool(
        [
            str(plink_bin),
            "--bfile",
            str(prefix),
            "--freq",
            "--out",
            str(out),
        ]
    )
    # .frq is space-separated with a leading space-padded header.
    return pd.read_csv(out.with_suffix(".frq"), sep=r"\s+", engine="python")


def _plink_freq_counts(
    plink_bin: pathlib.Path,
    prefix: pathlib.Path,
    out: pathlib.Path,
) -> pd.DataFrame:
    """Run ``plink --bfile <prefix> --freq counts`` and return the
    parsed ``.frq.counts`` table. Columns: CHR, SNP, A1, A2, C1 (minor
    allele count), C2 (major allele count), G0 (missing genotype
    count). Integer-valued — bypasses PLINK's 4-decimal MAF display
    truncation in ``--freq``."""
    helpers.run_tool(
        [
            str(plink_bin),
            "--bfile",
            str(prefix),
            "--freq",
            "counts",
            "--out",
            str(out),
        ]
    )
    return pd.read_csv(str(out) + ".frq.counts", sep=r"\s+", engine="python")


class TestPlink19FromPlinkInput:
    def test_minor_allele_frequency_matches_reference(
        self, tmp_path, plink19_bin, small_unphased_fixture
    ):
        # Use ``--freq counts`` so we get integer allele counts (C1 =
        # minor, C2 = major). PLINK truncates the ``MAF`` column of
        # ``--freq`` to 4 decimals, which fails an atol=1e-10
        # comparison at fractions like 105/396.
        out = tmp_path / "freq"
        df = _plink_freq_counts(plink19_bin, small_unphased_fixture.plink_prefix, out)

        ref = reference.compute_variant_stats(small_unphased_fixture.vcz_path)
        biallelic = ref.n_alleles == 2
        assert len(df) == int(biallelic.sum()), (
            f"plink --freq counts emitted {len(df)} rows, reference has "
            f"{int(biallelic.sum())} biallelic variants"
        )
        # PLINK preserves the .bim allele order, so ``C1`` is the count
        # of the file's A1 (ALT, per view-plink) — not always the minor.
        # Take min(C1, C2) for the minor-allele count.
        n_called = (df["C1"] + df["C2"]).to_numpy()
        minor_count = np.minimum(df["C1"].to_numpy(), df["C2"].to_numpy())
        minor_freq = minor_count / n_called
        np.testing.assert_allclose(
            minor_freq,
            ref.minor_freq[biallelic],
            atol=1e-10,
        )

    def test_chromosome_observation_count(
        self, tmp_path, plink19_bin, small_unphased_fixture
    ):
        # The fixture has 1% missing calls (all full-call). NCHROBS
        # reports the count of called chromosomes per variant: every
        # missing call subtracts 2 from the row's total. Compare
        # against the per-variant non-missing-allele count we get from
        # the source VCZ.
        out = tmp_path / "freq"
        df = _plink_freq(plink19_bin, small_unphased_fixture.plink_prefix, out)
        ref = reference.compute_variant_stats(small_unphased_fixture.vcz_path)
        biallelic = ref.n_alleles == 2
        np.testing.assert_array_equal(
            df["NCHROBS"].to_numpy(),
            ref.n_called[biallelic],
        )


class TestPlink19VariantIds:
    def test_bim_snp_column_matches_reference(
        self, plink19_bin, small_unphased_fixture
    ):
        # PLINK doesn't echo the .bim SNP column in --freq output, so
        # read it directly from the .bim file written by view-plink.
        bim = pd.read_csv(
            small_unphased_fixture.plink_prefix.with_suffix(".bim"),
            sep=r"\s+",
            engine="python",
            header=None,
            names=["chrom", "snp", "cm", "pos", "a1", "a2"],
            dtype={"snp": str},
        )
        ref = reference.compute_variant_stats(small_unphased_fixture.vcz_path)
        biallelic = ref.n_alleles == 2
        ids = reference.variant_ids(small_unphased_fixture.vcz_path)[biallelic]
        assert len(bim) == len(ids)
        np.testing.assert_array_equal(bim["snp"].to_numpy(), ids)


class TestPlink19RejectsBgenV12:
    """PLINK 1.9 cannot read BGEN v1.2. Pin the diagnostic so a future
    PLINK 1.9 release that drops the restriction surfaces here.
    """

    @pytest.mark.parametrize("level", cfg.BGEN_LEVELS)
    def test_rejects_layout2(
        self, tmp_path, plink19_bin, small_unphased_fixture, level
    ):
        bgen, sample = cfg.bgen_for_level(small_unphased_fixture, level)
        result = helpers.run_tool(
            [
                str(plink19_bin),
                "--bgen",
                str(bgen),
                "--sample",
                str(sample),
                "--freq",
                "--out",
                str(tmp_path / "frq"),
            ],
            check=False,
        )
        assert result.returncode != 0
        # PLINK's exact wording: "BGEN v1.2 input requires PLINK 2.0".
        assert "BGEN v1.2" in result.stderr or "BGEN v1.2" in result.stdout
