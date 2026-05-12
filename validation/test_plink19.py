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


class TestPlink19FromPlinkInput:
    def test_minor_allele_frequency_matches_reference(
        self, tmp_path, plink19_bin, small_unphased_fixture
    ):
        out = tmp_path / "freq"
        df = _plink_freq(plink19_bin, small_unphased_fixture.plink_prefix, out)

        ref = reference.compute_variant_stats(small_unphased_fixture.vcz_path)
        biallelic = ref.n_alleles == 2
        assert len(df) == int(biallelic.sum()), (
            f"plink --freq emitted {len(df)} rows, reference has "
            f"{int(biallelic.sum())} biallelic variants"
        )
        # PLINK reports MAF on the minor allele only; compare against
        # ``minor_freq`` directly. PLINK 1.9 uses exact float; no
        # quantisation tolerance needed (PLINK reads BED, not BGEN).
        np.testing.assert_allclose(
            df["MAF"].to_numpy(),
            ref.minor_freq[biallelic],
            atol=1e-10,
        )

    def test_chromosome_observation_count(
        self, tmp_path, plink19_bin, small_unphased_fixture
    ):
        # Each biallelic row has 2 * n_samples observed chromosomes
        # when no genotypes are missing (the msprime fixture has
        # no missingness).
        out = tmp_path / "freq"
        df = _plink_freq(plink19_bin, small_unphased_fixture.plink_prefix, out)
        n_samples = len(reference.sample_ids(small_unphased_fixture.vcz_path))
        np.testing.assert_array_equal(
            df["NCHROBS"].to_numpy(),
            np.full(len(df), 2 * n_samples),
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
