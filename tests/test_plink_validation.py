"""End-to-end validation: vcztools view-bed vs plink 2 --make-bed.

Modelled on :mod:`tests.test_bcftools_validation`. Each test runs
``plink2 --vcf <fixture> --make-bed [args]`` against a fixture's VCF
and ``vcztools view-bed <fixture>.vcz.zip [args]`` against the
matching VCZ, then compares the resulting filesets.

The guiding principle (per the implementation plan) is that PLINK 2
is canonical: when vcztools and PLINK 2 disagree on the same
variants, vcztools is wrong unless a deliberate divergence is
documented in :class:`TestKnownDivergences`. PLINK 1.9's default
behaviour (minor-allele relabelling on load) is *not* the spec —
PLINK 1.9 is treated as a downstream consumer that must be invoked
with ``--keep-allele-order`` to read our output faithfully.

The .bed payload is byte-compared. The .bim and .fam files are
text TSVs and compared via :func:`pandas.testing.assert_frame_equal`
on parsed DataFrames; this gives clearer failure messages than a
raw ``open(...).read()`` diff.

Multi-allelic corner cases are the headline target of this suite —
every fixture except ``sample-split-alleles.vcf.gz`` and
``1kg_2020_chr20_annotations.bcf`` contains some multi-allelic
variants. The validation strategy is to pass ``--max-alleles 2`` to
both tools and lock down the surviving-subset .bed bytes.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import click.testing as ct
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
import zarr
import zarr.storage

import vcztools.cli as cli
from tests import utils as test_utils
from tests import vcz_builder

PLINK2 = shutil.which("plink2")

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="Not supported on Windows"),
    pytest.mark.skipif(PLINK2 is None, reason="plink2 not on PATH"),
]


# ---------------------------------------------------------------------------
# Helpers — parsing, invocation, fileset comparison.
# ---------------------------------------------------------------------------


def _parse_fam(path: Path) -> pd.DataFrame:
    text = path.read_text()
    if text == "":
        return pd.DataFrame(columns=["FID", "IID", "Father", "Mother", "Sex", "Pheno"])
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["FID", "IID", "Father", "Mother", "Sex", "Pheno"],
        dtype=str,
    )


def _parse_bim(path: Path) -> pd.DataFrame:
    text = path.read_text()
    if text == "":
        return pd.DataFrame(columns=["Chrom", "ID", "CM", "Pos", "A1", "A2"])
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["Chrom", "ID", "CM", "Pos", "A1", "A2"],
        dtype={"Chrom": str, "ID": str, "CM": int, "Pos": int, "A1": str, "A2": str},
    )


def _input_flag(vcf_path: Path) -> str:
    """plink 2 takes ``--bcf`` for BCF input and ``--vcf`` for VCF."""
    if vcf_path.suffix == ".bcf":
        return "--bcf"
    return "--vcf"


def run_plink2(args: str, vcf_path: Path, out_prefix: Path) -> Path:
    """Run plink2 against a VCF/BCF fixture and return ``out_prefix``.

    ``args`` is the plink2 argument string (without --vcf/--bcf,
    --out, or --make-bed, which the helper supplies). Asserts that
    plink2 exited 0; on non-zero exit, dumps stderr into the
    AssertionError so the failure is debuggable.
    """
    cmd = (
        f"{PLINK2} {_input_flag(vcf_path)} {vcf_path} --make-bed "
        f"--out {out_prefix} {args}"
    )
    completed = subprocess.run(cmd, capture_output=True, check=False, shell=True)
    if completed.returncode != 0:
        raise AssertionError(
            f"plink2 exited with code {completed.returncode}\n"
            f"command: {cmd}\n"
            f"stderr:\n{completed.stderr.decode('utf-8', errors='replace')}"
        )
    return out_prefix


def run_plink2_expect_error(args: str, vcf_path: Path, out_prefix: Path) -> str:
    """Run plink2; assert it failed; return combined stderr/stdout."""
    cmd = (
        f"{PLINK2} {_input_flag(vcf_path)} {vcf_path} --make-bed "
        f"--out {out_prefix} {args}"
    )
    completed = subprocess.run(cmd, capture_output=True, check=False, shell=True)
    assert completed.returncode != 0, f"plink2 unexpectedly exited 0\ncommand: {cmd}"
    return completed.stderr.decode("utf-8", errors="replace") + completed.stdout.decode(
        "utf-8", errors="replace"
    )


def run_view_bed(args: str, vcz_path: Path, out_prefix: Path) -> Path:
    """Run ``vcztools view-bed`` and return the output prefix."""
    cmd = f"view-bed {vcz_path.as_posix()} --out {out_prefix.as_posix()} {args}"
    runner = ct.CliRunner()
    result = runner.invoke(cli.vcztools_main, cmd, catch_exceptions=False)
    if result.exit_code != 0:
        raise AssertionError(
            f"view-bed exited with code {result.exit_code}\n"
            f"command: {cmd}\n"
            f"stderr: {result.stderr}"
        )
    return out_prefix


def run_view_bed_expect_error(args: str, vcz_path: Path, out_prefix: Path) -> str:
    cmd = f"view-bed {vcz_path.as_posix()} --out {out_prefix.as_posix()} {args}"
    runner = ct.CliRunner()
    result = runner.invoke(cli.vcztools_main, cmd, catch_exceptions=False)
    assert result.exit_code != 0
    return result.stderr


def assert_filesets_equal(
    prefix1: Path,
    prefix2: Path,
    *,
    label1: str = "expected",
    label2: str = "actual",
):
    """Assert .bed/.bim/.fam under both prefixes are equivalent.

    .bed is byte-compared. .bim/.fam are parsed as TSV DataFrames and
    compared with :func:`pandas.testing.assert_frame_equal`. Failure
    messages name which file diverged.
    """
    bed1 = (prefix1.with_suffix(".bed")).read_bytes()
    bed2 = (prefix2.with_suffix(".bed")).read_bytes()
    if bed1 != bed2:
        raise AssertionError(
            f".bed bytes differ: {label1}={len(bed1)} bytes, "
            f"{label2}={len(bed2)} bytes; first differing byte at "
            f"{next((i for i, (a, b) in enumerate(zip(bed1, bed2)) if a != b), 'EOF')}"
        )
    bim1 = _parse_bim(prefix1.with_suffix(".bim"))
    bim2 = _parse_bim(prefix2.with_suffix(".bim"))
    pdt.assert_frame_equal(bim1, bim2, check_like=False)
    fam1 = _parse_fam(prefix1.with_suffix(".fam"))
    fam2 = _parse_fam(prefix2.with_suffix(".fam"))
    pdt.assert_frame_equal(fam1, fam2, check_like=False)


# ---------------------------------------------------------------------------
# Fixture discovery — fixtures that work end-to-end with plink 2.
# ---------------------------------------------------------------------------


# (vcf_filename, plink2_extra_args, vcztools_extra_args, expected_n_variants)
#
# - plink2_extra_args: extras beyond `--max-alleles 2` and `--allow-extra-chr`
#   (--make-bed needs --allow-extra-chr for non-standard contigs).
# - vcztools_extra_args: matching extras to drop the same variants on the
#   vcztools side. Currently we use `-e 'CHROM=="X"'` to skip chrX (plink 2
#   needs sex info for chrX rows under --make-bed).
# - expected_n_variants: the number of biallelic-or-monomorphic variants
#   surviving both filters; included so the suite catches a regression where
#   a filter accidentally drops a row that should survive.
ROUND_TRIP_FIXTURES = [
    # 13 variants, 13 biallelic+monomorphic, 3 on chrX → 10 survive.
    pytest.param(
        "sample-split-alleles.vcf.gz",
        "--not-chr X",
        "-e 'CHROM==\"X\"'",
        10,
        id="sample-split-alleles",
    ),
    # 100 variants, 4 multi-allelic, no chrX → 96 survive.
    pytest.param(
        "chr22.vcf.gz",
        "",
        "",
        96,
        id="chr22",
    ),
    # 5 variants, 1 multi-allelic, no chrX → 4 survive.
    pytest.param(
        "msprime_diploid.vcf.gz",
        "",
        "",
        4,
        id="msprime_diploid",
    ),
]


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


class TestBiallelicByteIdentical:
    """For each non-pathological fixture, plink 2 and vcztools produce
    byte-identical .bed and content-equal .bim/.fam under
    ``--max-alleles 2``.
    """

    @pytest.mark.parametrize(
        ("vcf_file", "plink2_args", "vcztools_args", "expected_n_variants"),
        ROUND_TRIP_FIXTURES,
    )
    def test_round_trip(
        self,
        tmp_path,
        fx_all_vcz,
        vcf_file,
        plink2_args,
        vcztools_args,
        expected_n_variants,
    ):
        fx = fx_all_vcz[vcf_file]
        p2_prefix = tmp_path / "p2"
        vcz_prefix = tmp_path / "vcz"

        run_plink2(
            f"--max-alleles 2 --allow-extra-chr {plink2_args}",
            fx.vcf_path,
            p2_prefix,
        )
        run_view_bed(
            f"--max-alleles 2 {vcztools_args}",
            fx.zip_path,
            vcz_prefix,
        )

        # Variant-count check first — gives the clearest message if a
        # filter is wrong.
        bim = _parse_bim(vcz_prefix.with_suffix(".bim"))
        assert len(bim) == expected_n_variants, (
            f"vcztools BIM has {len(bim)} variants, expected "
            f"{expected_n_variants} after the test's filter set"
        )

        assert_filesets_equal(p2_prefix, vcz_prefix)


class TestMultiallelicHandling:
    """Behaviour when multi-allelic variants are present.

    Both tools reject by default; both accept ``--max-alleles 2`` and
    drop multi-allelic rows. The error messages each tool prints
    mention multi-allelic-ness, so the user can tell what to do.
    """

    def test_default_rejection_vcztools(self, tmp_path, fx_chr22_vcz):
        # chr22 fixture has 4 multi-allelic variants. Without
        # --max-alleles 2 the vcztools writer raises.
        err = run_view_bed_expect_error("", fx_chr22_vcz.zip_path, tmp_path / "p")
        assert "Multi-allelic" in err

    def test_default_rejection_plink2(self, tmp_path, fx_chr22_vcz):
        err = run_plink2_expect_error("", fx_chr22_vcz.vcf_path, tmp_path / "p")
        # plink 2's exact wording: "cannot contain multiallelic variants".
        assert "multiallelic" in err.lower()

    def test_max_alleles_skips_in_both(self, tmp_path, fx_chr22_vcz):
        # With --max-alleles 2 both succeed and produce the same fileset.
        p2_prefix = tmp_path / "p2"
        vcz_prefix = tmp_path / "vcz"
        run_plink2(
            "--max-alleles 2 --allow-extra-chr",
            fx_chr22_vcz.vcf_path,
            p2_prefix,
        )
        run_view_bed(
            "--max-alleles 2",
            fx_chr22_vcz.zip_path,
            vcz_prefix,
        )
        assert_filesets_equal(p2_prefix, vcz_prefix)

    def test_all_multiallelic_yields_empty_fileset(self, tmp_path):
        # Synthetic store with two multi-allelic variants and no
        # biallelic ones. --max-alleles 2 drops both; the resulting
        # .bed has only the 3-byte magic header, the .bim is empty,
        # and the .fam still lists every sample.
        root = vcz_builder.make_vcz(
            num_samples=3,
            variant_contig=[0, 0],
            variant_position=[100, 200],
            alleles=[("A", "T", "C"), ("G", "C", "T", "A")],
            sample_id=["s1", "s2", "s3"],
            call_genotype=np.zeros((2, 3, 2), dtype=np.int8),
        )
        # Persist the in-memory VCZ to a directory so the CLI can open it.
        vcz_dir = tmp_path / "synth.vcz"
        dest = zarr.storage.LocalStore(vcz_dir)
        test_utils.copy_store(root.store, dest)

        vcz_prefix = tmp_path / "vcz"
        run_view_bed("--max-alleles 2", vcz_dir, vcz_prefix)
        assert vcz_prefix.with_suffix(".bed").read_bytes() == b"\x6c\x1b\x01"
        assert vcz_prefix.with_suffix(".bim").read_text() == ""
        fam = _parse_fam(vcz_prefix.with_suffix(".fam"))
        assert list(fam["IID"]) == ["s1", "s2", "s3"]


class TestSampleSubset:
    """Sample-subset selection vs plink 2 ``--keep`` / ``--remove``.

    plink 2 expects a file (FID + IID columns); we generate one in
    ``tmp_path`` per test so we can compare apples-to-apples.
    """

    @staticmethod
    def _write_keep_file(path: Path, iids: list[str]) -> Path:
        # FID column is "0" to match what vcztools writes; IIDs follow.
        path.write_text("\n".join(f"0\t{iid}" for iid in iids) + "\n")
        return path

    def test_keep_subset(self, tmp_path, fx_chr22_vcz):
        # chr22 has 100 samples HG00096..HG00146. Keep 3 of them.
        iids = ["HG00096", "HG00100", "HG00120"]
        keep_file = self._write_keep_file(tmp_path / "keep.txt", iids)

        p2_prefix = tmp_path / "p2"
        vcz_prefix = tmp_path / "vcz"
        run_plink2(
            f"--max-alleles 2 --allow-extra-chr --keep {keep_file}",
            fx_chr22_vcz.vcf_path,
            p2_prefix,
        )
        run_view_bed(
            f"--max-alleles 2 -s {','.join(iids)}",
            fx_chr22_vcz.zip_path,
            vcz_prefix,
        )
        assert_filesets_equal(p2_prefix, vcz_prefix)

    def test_keep_subset_via_samples_file(self, tmp_path, fx_chr22_vcz):
        # vcztools -S accepts a one-IID-per-line file (no FID column).
        iids = ["HG00096", "HG00100", "HG00120"]
        vcz_samples_file = tmp_path / "samples.txt"
        vcz_samples_file.write_text("\n".join(iids) + "\n")

        plink_keep_file = self._write_keep_file(tmp_path / "keep.txt", iids)

        p2_prefix = tmp_path / "p2"
        vcz_prefix = tmp_path / "vcz"
        run_plink2(
            f"--max-alleles 2 --allow-extra-chr --keep {plink_keep_file}",
            fx_chr22_vcz.vcf_path,
            p2_prefix,
        )
        run_view_bed(
            f"--max-alleles 2 -S {vcz_samples_file}",
            fx_chr22_vcz.zip_path,
            vcz_prefix,
        )
        assert_filesets_equal(p2_prefix, vcz_prefix)

    def test_remove_subset(self, tmp_path, fx_chr22_vcz):
        # vcztools' -s ^X negation vs plink 2's --remove file.
        iids_to_drop = ["HG00099", "HG00100"]
        remove_file = self._write_keep_file(tmp_path / "remove.txt", iids_to_drop)

        p2_prefix = tmp_path / "p2"
        vcz_prefix = tmp_path / "vcz"
        run_plink2(
            f"--max-alleles 2 --allow-extra-chr --remove {remove_file}",
            fx_chr22_vcz.vcf_path,
            p2_prefix,
        )
        run_view_bed(
            f"--max-alleles 2 -s ^{','.join(iids_to_drop)}",
            fx_chr22_vcz.zip_path,
            vcz_prefix,
        )
        assert_filesets_equal(p2_prefix, vcz_prefix)


class TestRegionSelection:
    """Region/target selection vs plink 2 ``--chr`` / ``--from-bp`` /
    ``--to-bp`` and ``--extract range`` / ``--exclude range``.

    Region semantics differ in subtle ways between tools; these tests
    pin the cases where they agree (positional intervals) and rely on
    plink 2's behaviour as canonical.
    """

    def test_chr_and_position_range(self, tmp_path, fx_chr22_vcz):
        # vcztools '-r chr22:10510000-10511000' vs plink 2
        # '--chr 22 --from-bp 10510000 --to-bp 10511000'.
        # NB: plink 2 strips the "chr" prefix, so it needs "22" not
        # "chr22"; the same .bed bytes result either way.
        p2_prefix = tmp_path / "p2"
        vcz_prefix = tmp_path / "vcz"
        run_plink2(
            "--max-alleles 2 --allow-extra-chr --chr 22 "
            "--from-bp 10510000 --to-bp 10511000",
            fx_chr22_vcz.vcf_path,
            p2_prefix,
        )
        run_view_bed(
            "--max-alleles 2 -r chr22:10510000-10511000",
            fx_chr22_vcz.zip_path,
            vcz_prefix,
        )
        assert_filesets_equal(p2_prefix, vcz_prefix)

    def test_chr_only(self, tmp_path, fx_chr22_vcz):
        # Whole-contig select. Both should yield the full --max-alleles 2 set.
        p2_prefix = tmp_path / "p2"
        vcz_prefix = tmp_path / "vcz"
        run_plink2(
            "--max-alleles 2 --allow-extra-chr --chr 22",
            fx_chr22_vcz.vcf_path,
            p2_prefix,
        )
        run_view_bed(
            "--max-alleles 2 -r chr22",
            fx_chr22_vcz.zip_path,
            vcz_prefix,
        )
        assert_filesets_equal(p2_prefix, vcz_prefix)


class TestMonomorphicSites:
    """Pin the monomorphic encoding: A1='.', A2=REF, all genotypes
    encode as MISSING in the .bed.
    """

    def test_monomorphic_a1_is_dot_in_both(self, tmp_path, fx_sample_split_alleles_vcz):
        # sample-split-alleles has two monomorphic variants on chr20
        # (positions 1230237 and 1235237). Both tools should write
        # A1="." for those rows, preserving REF in A2.
        p2_prefix = tmp_path / "p2"
        vcz_prefix = tmp_path / "vcz"
        run_plink2(
            "--max-alleles 2 --allow-extra-chr --not-chr X",
            fx_sample_split_alleles_vcz.vcf_path,
            p2_prefix,
        )
        run_view_bed(
            "--max-alleles 2 -e 'CHROM==\"X\"'",
            fx_sample_split_alleles_vcz.zip_path,
            vcz_prefix,
        )
        bim = _parse_bim(vcz_prefix.with_suffix(".bim"))
        # The two known monomorphic rows must have A1=".", A2 the REF.
        mono = bim[bim["A1"] == "."]
        assert list(mono["Pos"]) == [1230237, 1235237]
        assert list(mono["A2"]) == ["T", "T"]
        # And the same byte-level output as plink 2.
        assert_filesets_equal(p2_prefix, vcz_prefix)


class TestKnownDivergences:
    """Cases where vcztools deliberately doesn't match plink 2.

    Each case is marked ``xfail(strict=True)`` so a future fix that
    *closes* the divergence will be flagged in CI for the comment to
    be updated.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "plink 2 errors on chrX without sex info (--update-sex / "
            "--split-par); vcztools writes chrX rows pass-through "
            "because we don't track sex. This is by design — "
            "see the 'PLINK 1 binary output' docs."
        ),
    )
    def test_chrx_without_sex_info(self, tmp_path, fx_sample_split_alleles_vcz):
        # plink 2 errors out; vcztools succeeds. We assert "the
        # filesets agree" to make the divergence the failing axis.
        p2_prefix = tmp_path / "p2"
        vcz_prefix = tmp_path / "vcz"
        # plink 2 will error here; the assertion below is what we
        # want to mark as deliberately divergent.
        run_plink2(
            "--max-alleles 2 --allow-extra-chr",
            fx_sample_split_alleles_vcz.vcf_path,
            p2_prefix,
        )
        run_view_bed(
            "--max-alleles 2",
            fx_sample_split_alleles_vcz.zip_path,
            vcz_prefix,
        )
        assert_filesets_equal(p2_prefix, vcz_prefix)
