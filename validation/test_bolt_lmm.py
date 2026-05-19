"""BOLT-LMM accepts vcztools view-plink and view-bgen output as valid
genotype input.

BOLT-LMM's LMM variance-component fit is fragile on small / synthetic
genomes — heritability estimation routinely returns h2g ≈ 0 and the
process segfaults during the inf-model step for any fixture small
enough to ship in a local validation suite. Real performance
benchmarks of BOLT-LMM use UK-Biobank-scale data: hundreds of
thousands of independent loci across 22 chromosomes. We can't
reproduce that here.

What we *can* validate is the file-format contract:

- BOLT-LMM parses ``view-plink`` output (.bed/.bim/.fam) and reports
  the correct sample and SNP counts in its log.
- BOLT-LMM parses ``view-bgen`` output (.bgen/.sample), at all three
  BGEN flavours (CLI lvl=-1, CLI lvl=0, encoder), and reports
  matching counts.

This pins the "downstream tool can read the file" contract, which is
the actual deliverable of the validation suite for vcztools. The
allele-frequency cross-check is left to qctool, PLINK 1.9, and
REGENIE, all of which produce reliable per-variant frequency output on
the same fixtures.

Each test passes ``--exclude`` listing every variant after the first
:data:`KEEP_N_SNPS` so BOLT only processes a handful — wall time per
test is then ~1.5 s instead of minutes. The summary lines we assert
on (``Total snps in PLINK data``, ``snpBlocks (Mbgen)``) report the
pre-exclude count, so the assertion logic is unchanged.
"""

from __future__ import annotations

import pathlib
import re

import pandas as pd
import pytest

from . import conftest as cfg
from . import helpers, reference

KEEP_N_SNPS = 20
KEEP_N_SAMPLES = 10


def _write_exclude_all_but_first_n(
    bim_path: pathlib.Path, out_path: pathlib.Path, keep_n: int = KEEP_N_SNPS
) -> pathlib.Path:
    """Write a BOLT ``--exclude`` file naming every SNP in ``bim_path``
    after the first ``keep_n``. BOLT then skips those, so the LMM-side
    work shrinks to a handful of variants per run."""
    bim = pd.read_csv(
        bim_path,
        sep=r"\s+",
        engine="python",
        header=None,
        names=["chrom", "snp", "cm", "pos", "a1", "a2"],
        dtype={"snp": str},
    )
    excluded = bim["snp"].iloc[keep_n:]
    out_path.write_text("\n".join(excluded) + "\n")
    return out_path


def _write_remove_all_but_first_n(
    fam_path: pathlib.Path,
    out_path: pathlib.Path,
    keep_n: int = KEEP_N_SAMPLES,
) -> pathlib.Path:
    """Write a BOLT ``--remove`` file naming every individual in
    ``fam_path`` after the first ``keep_n``. Combined with
    ``--exclude``, BOLT operates on a sample × variant slab small
    enough that the LMM step finishes in well under a second."""
    fam = pd.read_csv(
        fam_path,
        sep=r"\s+",
        engine="python",
        header=None,
        names=["fid", "iid", "fa", "mo", "sex", "pheno"],
        dtype={"fid": str, "iid": str},
    )
    removed = fam[["fid", "iid"]].iloc[keep_n:]
    out_path.write_text(
        "\n".join(f"{fid} {iid}" for fid, iid in removed.itertuples(index=False)) + "\n"
    )
    return out_path


def _remap_sample_fid0(
    sample_path: pathlib.Path, out_path: pathlib.Path
) -> pathlib.Path:
    """Rewrite a BGEN ``.sample`` file's ID_1 column to "0" so the
    (FID, IID) pairs match the ``.fam`` written by ``view-plink``.
    BOLT-LMM, when given both ``--bfile`` and ``--bgenFile``, cross-
    checks the two on (FID, IID) and aborts if any sample is missing
    on either side."""
    lines = sample_path.read_text().splitlines()
    # .sample format: header row, then a per-column type row
    # (0=identifier), then one row per sample.
    out = [lines[0], lines[1]]
    for row in lines[2:]:
        parts = row.split()
        parts[0] = "0"
        out.append(" ".join(parts))
    out_path.write_text("\n".join(out) + "\n")
    return out_path


def _bolt_run(
    bolt: pathlib.Path,
    args: list[str],
    *,
    expected_n_full_samples: int,
    expected_n_full_snps: int,
) -> str:
    """Run BOLT-LMM and return its log stdout. We don't assert exit 0
    because the LMM fit can fail on small synthetic data; we only need
    the genotype-data parsing to succeed, which happens before the
    fit. The ``Nbed`` / ``Mbed`` lines report the count loaded from
    PLINK *before* ``--remove`` / ``--exclude`` filtering, so the
    expected values are still the full-fixture totals.
    """
    result = helpers.run_tool([str(bolt), *args], check=False)
    log = result.stdout
    m_n_plink = re.search(r"Total indivs in PLINK data: Nbed = (\d+)", log)
    m_m_plink = re.search(r"Total snps in PLINK data: Mbed = (\d+)", log)
    assert m_n_plink is not None, (
        f"BOLT didn't print a PLINK samples-loaded summary; stdout tail:\n{log[-2000:]}"
    )
    assert m_m_plink is not None, (
        f"BOLT didn't print a PLINK SNPs-loaded summary; stdout tail:\n{log[-2000:]}"
    )
    assert int(m_n_plink.group(1)) == expected_n_full_samples
    assert int(m_m_plink.group(1)) == expected_n_full_snps
    return log


class TestBoltLmmFromPlinkInput:
    def test_loads_bed_and_runs_linreg(
        self, tmp_path, bolt_lmm_bin, large_unphased_fixture
    ):
        # Pheno is FID="0" by construction, matching view-plink's .fam.
        pheno = large_unphased_fixture.pheno_path
        exclude = _write_exclude_all_but_first_n(
            large_unphased_fixture.plink_prefix.with_suffix(".bim"),
            tmp_path / "exclude.txt",
        )
        remove = _write_remove_all_but_first_n(
            large_unphased_fixture.plink_prefix.with_suffix(".fam"),
            tmp_path / "remove.txt",
        )
        ref = reference.compute_variant_stats(large_unphased_fixture.vcz_path)
        n_samples = len(reference.sample_ids(large_unphased_fixture.vcz_path))
        n_snps_biallelic = int((ref.n_alleles == 2).sum())
        log = _bolt_run(
            bolt_lmm_bin,
            [
                "--bfile",
                str(large_unphased_fixture.plink_prefix),
                "--exclude",
                str(exclude),
                "--remove",
                str(remove),
                "--phenoFile",
                str(pheno),
                "--phenoCol",
                "Y1",
                "--lmm",
                "--LDscoresUseChip",
                "--numLeaveOutChunks",
                "2",
                "--statsFile",
                str(tmp_path / "stats.txt"),
            ],
            expected_n_full_samples=n_samples,
            expected_n_full_snps=n_snps_biallelic,
        )
        # BOLT computes LINREG stats before fitting the LMM. The
        # presence of the timing line confirms LINREG ran successfully
        # against our genotype data.
        assert "Computing linear regression (LINREG) stats" in log


class TestBoltLmmFromBgenInput:
    @pytest.mark.parametrize("level", cfg.BGEN_LEVELS)
    def test_loads_bgen(self, tmp_path, bolt_lmm_bin, large_unphased_fixture, level):
        bgen, sample_src = cfg.bgen_for_level(large_unphased_fixture, level)
        # BOLT also requires a .bed-style anchor input for --lmm even
        # when scoring BGEN variants, so we pass --bfile alongside.
        # Pheno is already FID="0" by construction; the .sample file is
        # remapped from ID_1=ID_2=sample_id to FID="0" so (FID, IID)
        # pairs line up with the .fam written by view-plink (BOLT
        # cross-checks .fam and .sample).
        pheno = large_unphased_fixture.pheno_path
        sample = _remap_sample_fid0(sample_src, tmp_path / "remap.sample")
        exclude = _write_exclude_all_but_first_n(
            large_unphased_fixture.plink_prefix.with_suffix(".bim"),
            tmp_path / "exclude.txt",
        )
        remove = _write_remove_all_but_first_n(
            large_unphased_fixture.plink_prefix.with_suffix(".fam"),
            tmp_path / "remove.txt",
        )
        ref = reference.compute_variant_stats(large_unphased_fixture.vcz_path)
        n_samples = len(reference.sample_ids(large_unphased_fixture.vcz_path))
        n_snps_biallelic = int((ref.n_alleles == 2).sum())
        log = _bolt_run(
            bolt_lmm_bin,
            [
                "--bfile",
                str(large_unphased_fixture.plink_prefix),
                "--bgenFile",
                str(bgen),
                "--sampleFile",
                str(sample),
                "--exclude",
                str(exclude),
                "--remove",
                str(remove),
                "--phenoFile",
                str(pheno),
                "--phenoCol",
                "Y1",
                "--lmm",
                "--LDscoresUseChip",
                "--numLeaveOutChunks",
                "2",
                "--statsFile",
                str(tmp_path / "stats.txt"),
                "--statsFileBgenSnps",
                str(tmp_path / "stats_bgen.txt"),
            ],
            expected_n_full_samples=n_samples,
            expected_n_full_snps=n_snps_biallelic,
        )
        # BGEN parsing prints its own summary lines:
        #   samples (Nbgen): <N>
        #   snpBlocks (Mbgen): <M>
        m_n_bgen = re.search(r"samples \(Nbgen\): (\d+)", log)
        m_m_bgen = re.search(r"snpBlocks \(Mbgen\): (\d+)", log)
        assert m_n_bgen is not None, (
            "BOLT didn't print a BGEN samples-loaded summary; "
            f"stdout tail:\n{log[-2000:]}"
        )
        assert m_m_bgen is not None, (
            f"BOLT didn't print a BGEN SNPs-loaded summary; stdout tail:\n{log[-2000:]}"
        )
        assert int(m_n_bgen.group(1)) == n_samples
        assert int(m_m_bgen.group(1)) == n_snps_biallelic


class TestBoltLmmPhasedBgen:
    """BOLT-LMM handles vcztools' phased BGEN cleanly — unlike qctool
    and REGENIE, which reject it outright. Document that and pin the
    summary lines we'd expect for either phase."""

    def test_loads_phased_bgen(self, tmp_path, bolt_lmm_bin, large_phased_fixture):
        bgen, sample_src = (
            large_phased_fixture.bgen_minus1,
            large_phased_fixture.sample_path,
        )
        pheno = large_phased_fixture.pheno_path
        sample = _remap_sample_fid0(sample_src, tmp_path / "remap.sample")
        exclude = _write_exclude_all_but_first_n(
            large_phased_fixture.plink_prefix.with_suffix(".bim"),
            tmp_path / "exclude.txt",
        )
        remove = _write_remove_all_but_first_n(
            large_phased_fixture.plink_prefix.with_suffix(".fam"),
            tmp_path / "remove.txt",
        )
        ref = reference.compute_variant_stats(large_phased_fixture.vcz_path)
        n_samples = len(reference.sample_ids(large_phased_fixture.vcz_path))
        n_snps_biallelic = int((ref.n_alleles == 2).sum())
        log = _bolt_run(
            bolt_lmm_bin,
            [
                "--bfile",
                str(large_phased_fixture.plink_prefix),
                "--bgenFile",
                str(bgen),
                "--sampleFile",
                str(sample),
                "--exclude",
                str(exclude),
                "--remove",
                str(remove),
                "--phenoFile",
                str(pheno),
                "--phenoCol",
                "Y1",
                "--lmm",
                "--LDscoresUseChip",
                "--numLeaveOutChunks",
                "2",
                "--statsFile",
                str(tmp_path / "stats.txt"),
                "--statsFileBgenSnps",
                str(tmp_path / "stats_bgen.txt"),
            ],
            expected_n_full_samples=n_samples,
            expected_n_full_snps=n_snps_biallelic,
        )
        m_n_bgen = re.search(r"samples \(Nbgen\): (\d+)", log)
        m_m_bgen = re.search(r"snpBlocks \(Mbgen\): (\d+)", log)
        assert m_n_bgen is not None
        assert m_m_bgen is not None
        assert int(m_n_bgen.group(1)) == n_samples
        assert int(m_m_bgen.group(1)) == n_snps_biallelic


# Variant ID round-trip via BOLT-LMM is not asserted: on this synthetic
# fixture BOLT's LMM fit fails before emitting a usable stats file (the
# same fragility that drives TestBoltLmmFromBgenInput to assert on log
# lines rather than output content). qctool, PLINK 1.9, bgenix, and
# REGENIE all cover variant ID round-trip on the same data; pinning it
# again through BOLT would require either a non-synthetic fixture or a
# BOLT mode that bypasses the LMM step.
