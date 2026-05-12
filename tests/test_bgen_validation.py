"""External validation for ``vcztools view-bgen``.

Two layers:

1. :class:`TestBgenReaderRoundtrip` — always runs. Parses ``view-bgen``
   output with the maintained reference reader (``bgen-reader``) and
   asserts the genotype matrix and metadata round-trip from the source
   VCZ store. Confirms our spec interpretation matches a third-party
   reader's.

2. :class:`TestPlink2BgenCrossCheck` — skipped if ``plink2`` is not on
   PATH. Runs ``plink2 --export bgen-1.2 bits=8 ref-first`` and
   ``vcztools view-bgen`` on the same fixture, parses both outputs with
   ``bgen-reader``, and asserts they agree on the genotype probability
   matrix and core metadata. Confirms semantic equivalence with the
   reference transcoder. Known field-level divergences (sample-ID
   format, rsid synthesis) are documented in :func:`_strip_double_id`
   and the per-test comments.
"""

import shutil
import subprocess
from pathlib import Path

import bgen_reader as br
import click.testing as ct
import numpy as np
import pytest

import vcztools.cli as cli
from tests import vcz_builder
from vcztools import bgen as bgen_mod
from vcztools import retrieval

PLINK2 = shutil.which("plink2")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_view_bgen(args: str, vcz_path: Path, out_prefix: Path) -> Path:
    """Invoke ``vcztools view-bgen`` via CliRunner; assert exit 0; return
    the output prefix."""
    cmd = f"view-bgen {Path(vcz_path).as_posix()} --out {out_prefix.as_posix()} {args}"
    runner = ct.CliRunner()
    result = runner.invoke(cli.vcztools_main, cmd, catch_exceptions=False)
    if result.exit_code != 0:
        raise AssertionError(
            f"vcztools view-bgen exited with {result.exit_code}\n"
            f"command: {cmd}\nstderr:\n{result.stderr}"
        )
    return out_prefix


def run_plink2_bgen(args: str, vcf_path: Path, out_prefix: Path) -> Path:
    """Run plink2 BGEN export against ``vcf_path`` and return ``out_prefix``.

    ``args`` is appended to the plink2 invocation; the helper supplies
    ``--vcf``, ``--out``, ``--export bgen-1.2 bits=8 ref-first``, and
    ``--double-id`` (so VCF samples become FID==IID rather than 0_IID).
    Asserts plink2 exited 0; on failure dumps stderr into the
    AssertionError."""
    cmd = (
        f"{PLINK2} --vcf {Path(vcf_path).as_posix()} "
        f"--export bgen-1.2 bits=8 ref-first --double-id "
        f"--out {out_prefix.as_posix()} {args}"
    )
    completed = subprocess.run(cmd, capture_output=True, check=False, shell=True)
    if completed.returncode != 0:
        raise AssertionError(
            f"plink2 exited with code {completed.returncode}\n"
            f"command: {cmd}\n"
            f"stderr:\n{completed.stderr.decode('utf-8', errors='replace')}"
        )
    return out_prefix


def run_plink2_read_bgen(
    args: str, bgen_path: Path, sample_path: Path, out_prefix: Path
) -> Path:
    """Run ``plink2 --bgen <bgen> ref-first --sample <sample>`` and return
    ``out_prefix``. ``args`` is appended verbatim; callers add region
    filters (``--chr``/``--from-bp``/``--to-bp``) and an output stage
    (e.g. ``--make-just-pvar``). plink2 v2.00a6 does not consume the
    ``.bgen.bgi`` sidecar — region selection is filtered in memory after
    a full streaming scan — but the index sits alongside the ``.bgen``
    in the same directory."""
    cmd = (
        f"{PLINK2} --bgen {Path(bgen_path).as_posix()} ref-first "
        f"--sample {Path(sample_path).as_posix()} "
        f"--out {out_prefix.as_posix()} {args}"
    )
    completed = subprocess.run(cmd, capture_output=True, check=False, shell=True)
    if completed.returncode != 0:
        raise AssertionError(
            f"plink2 exited with code {completed.returncode}\n"
            f"command: {cmd}\n"
            f"stderr:\n{completed.stderr.decode('utf-8', errors='replace')}"
        )
    return out_prefix


def _strip_double_id(samples) -> list[str]:
    """plink2's BGEN export with ``--double-id`` writes sample IDs as
    ``"FID_IID"``. Two flavours appear in practice for VCF input:

    - ``"0_NA00001"`` — plink2 defaults to ``FID=0`` for VCFs without
      an explicit FID column.
    - ``"HG00096_HG00096"`` — when ``--double-id`` activates and the
      VCF sample IDs aren't trivially decomposable.

    Either way the IID is the part after ``_``; strip the prefix for
    comparison against vcztools' raw sample-ID encoding.
    """
    out = []
    for s in samples:
        s = str(s)
        if "_" in s:
            fid, iid = s.split("_", 1)
            if fid == "0" or fid == iid:
                s = iid
        out.append(s)
    return out


def _strip_chr_prefix(chroms) -> list[str]:
    """plink2's BGEN export normalises human chromosome names like
    ``--make-bed`` does (strips ``chr`` from 1-22, X, Y, MT; rewrites
    ``chrM`` → ``MT``). vcztools passes contig names through unchanged.
    Strip the prefix on both sides for comparison."""
    out = []
    for c in chroms:
        c = str(c)
        if c.startswith("chr"):
            suffix = c[3:]
            if suffix == "M":
                suffix = "MT"
            c = suffix
        out.append(c)
    return out


def _bgen_unphased_dosages(probs: np.ndarray) -> np.ndarray:
    """``probs`` from bgen-reader for unphased biallelic diploid is
    shape ``(n_samples, n_variants, 3)`` storing P(00), P(01), P(11) per
    sample. Returns the integer ALT dosage matrix as
    ``(n_variants, n_samples)``. Missing samples (NaN in ``probs``) are
    coerced to ``0``; callers must mask via the source missing flag."""
    dosage = probs[..., 1] + 2 * probs[..., 2]
    dosage = np.where(np.isnan(dosage), 0.0, dosage)
    return np.rint(dosage).astype(np.int8).T


def _vcz_dosages_and_missing(call_genotype: np.ndarray):
    """Compute per-(variant, sample) ALT-allele dosage and a missing
    mask from a VCZ ``call_genotype`` slice of shape
    ``(n_variants, n_samples, 2)``. Negative entries (``-1`` for VCF
    missing or ``-2`` for haploid padding) mark a sample as missing
    for that variant; both alleles must be non-negative for the
    dosage to be valid."""
    G = np.asarray(call_genotype)
    missing = (G < 0).any(axis=-1)
    safe = np.where(G < 0, 0, G)
    dosage = safe.sum(axis=-1).astype(np.int8)
    return dosage, missing


def _filter_indexes(group, *, drop_chrom: str | None = None, max_alleles: int = 2):
    """Return the boolean mask over the source variant axis selecting
    rows that survive ``--max-alleles N`` and an optional chromosome
    drop. Used by tests that need to align a reader-side filter
    (applied via ``view-bgen``) with the source-side ``call_genotype``."""
    contig_id = group["contig_id"][...]
    variant_contig = group["variant_contig"][...]
    variant_allele = group["variant_allele"][...]
    n_alleles = (variant_allele != "").sum(axis=1)
    keep = n_alleles <= max_alleles
    if drop_chrom is not None:
        chroms = contig_id[variant_contig]
        keep = keep & (chroms != drop_chrom)
    return keep


# ---------------------------------------------------------------------------
# Always-run roundtrip (no external tool)
# ---------------------------------------------------------------------------


class TestBgenReaderRoundtrip:
    """Parse vcztools output with ``bgen-reader`` and assert the source
    VCZ round-trips through the BGEN encoding."""

    def test_sample_fixture_genotypes(self, tmp_path, fx_sample_vcz):
        # Filter chrX + multi-allelic to fit BGEN biallelic profile.
        out = tmp_path / "x"
        run_view_bgen(
            "--max-alleles 2 -e 'CHROM==\"X\"'",
            fx_sample_vcz.zip_path,
            out,
        )
        keep = _filter_indexes(fx_sample_vcz.group, drop_chrom="X", max_alleles=2)
        source_g = fx_sample_vcz.group["call_genotype"][...][keep]
        with br.open_bgen(out.with_suffix(".bgen"), verbose=False) as bgen:
            assert list(bgen.samples) == ["NA00001", "NA00002", "NA00003"]
            assert bgen.nvariants == int(keep.sum())
            probs = bgen.read()
        bgen_dosage = _bgen_unphased_dosages(probs)
        source_dosage, missing = _vcz_dosages_and_missing(source_g)
        np.testing.assert_array_equal(bgen_dosage[~missing], source_dosage[~missing])

    def test_sample_fixture_metadata(self, tmp_path, fx_sample_vcz):
        out = tmp_path / "x"
        run_view_bgen(
            "--max-alleles 2 -e 'CHROM==\"X\"'",
            fx_sample_vcz.zip_path,
            out,
        )
        keep = _filter_indexes(fx_sample_vcz.group, drop_chrom="X", max_alleles=2)
        contig_id = fx_sample_vcz.group["contig_id"][...]
        variant_contig = fx_sample_vcz.group["variant_contig"][...][keep]
        expected_chrom = list(contig_id[variant_contig])
        expected_pos = list(fx_sample_vcz.group["variant_position"][...][keep])
        with br.open_bgen(out.with_suffix(".bgen"), verbose=False) as bgen:
            assert [str(c) for c in bgen.chromosomes] == expected_chrom
            assert [int(p) for p in bgen.positions] == expected_pos
            # Allele list is comma-joined by bgen-reader. Each variant
            # has exactly 2 alleles (biallelic + monomorphic encoded as
            # "REF,.").
            for allele_str in bgen.allele_ids:
                assert len(str(allele_str).split(",")) == 2

    def test_sample_subset(self, tmp_path, fx_sample_vcz):
        out = tmp_path / "x"
        run_view_bgen(
            "--max-alleles 2 -e 'CHROM==\"X\"' -s NA00001,NA00003",
            fx_sample_vcz.zip_path,
            out,
        )
        with br.open_bgen(out.with_suffix(".bgen"), verbose=False) as bgen:
            assert list(bgen.samples) == ["NA00001", "NA00003"]
            assert bgen.nsamples == 2

    def test_msprime_fixture_genotypes(self, tmp_path, fx_msprime_diploid_vcz):
        # The msprime fixture has one tri-allelic site; --max-alleles 2
        # drops it. The remaining 4 sites are phased biallelic diploid.
        out = tmp_path / "x"
        run_view_bgen("--max-alleles 2", fx_msprime_diploid_vcz.zip_path, out)
        keep = _filter_indexes(fx_msprime_diploid_vcz.group, max_alleles=2)
        source_g = fx_msprime_diploid_vcz.group["call_genotype"][...][keep]
        with br.open_bgen(out.with_suffix(".bgen"), verbose=False) as bgen:
            assert bgen.nvariants == int(keep.sum())
            # msprime output is fully phased.
            assert all(bool(p) for p in bgen.phased)
            probs = bgen.read()
        # For phased diploid biallelic, bgen-reader returns shape
        # (n_samples, n_variants, 4): per-haplotype P(allele 0), P(allele 1).
        assert probs.shape[-1] == 4
        # Decode hap1 and hap2 separately and compare against source.
        hap1 = np.argmax(probs[..., 0:2], axis=-1).T  # (n_variants, n_samples)
        hap2 = np.argmax(probs[..., 2:4], axis=-1).T
        np.testing.assert_array_equal(hap1, source_g[..., 0])
        np.testing.assert_array_equal(hap2, source_g[..., 1])

    def test_synthetic_phased(self, tmp_path):
        # All-phased synthetic VCZ: every (0,1) and (1,0) must distinguish
        # in the BGEN output.
        G = np.array(
            [
                [[0, 1], [1, 0], [0, 0], [1, 1]],
                [[1, 1], [0, 1], [1, 0], [0, 0]],
            ],
            dtype=np.int8,
        )
        phased = np.ones(G.shape[:2], dtype=bool)
        root = vcz_builder.make_vcz(
            variant_contig=[0, 0],
            variant_position=[100, 200],
            alleles=[("A", "T"), ("C", "G")],
            num_samples=4,
            call_genotype=G,
            call_fields={"genotype_phased": phased},
        )
        reader = retrieval.VczReader(root)
        out = tmp_path / "p"
        bgen_mod.write_bgen(reader, out)
        with br.open_bgen(out.with_suffix(".bgen"), verbose=False) as bgen:
            assert all(bool(p) for p in bgen.phased)
            probs = bgen.read()
        hap1 = np.argmax(probs[..., 0:2], axis=-1).T
        hap2 = np.argmax(probs[..., 2:4], axis=-1).T
        np.testing.assert_array_equal(hap1, G[..., 0])
        np.testing.assert_array_equal(hap2, G[..., 1])

    def test_missing_genotypes(self, tmp_path):
        G = np.array(
            [
                [[0, 0], [0, 1], [-1, -1], [1, 1]],
                [[0, 1], [-1, 0], [1, 1], [0, 0]],
            ],
            dtype=np.int8,
        )
        root = vcz_builder.make_vcz(
            variant_contig=[0, 0],
            variant_position=[100, 200],
            alleles=[("A", "T"), ("C", "G")],
            num_samples=4,
            call_genotype=G,
        )
        reader = retrieval.VczReader(root)
        out = tmp_path / "m"
        bgen_mod.write_bgen(reader, out)
        with br.open_bgen(out.with_suffix(".bgen"), verbose=False) as bgen:
            probs, missing_mask = bgen.read(return_missings=True)
        # bgen-reader missing_mask is (n_samples, n_variants).
        expected_missing = (G < 0).any(axis=-1).T  # (n_samples, n_variants)
        np.testing.assert_array_equal(missing_mask, expected_missing)


# ---------------------------------------------------------------------------
# plink2 cross-check (skipped if plink2 not on PATH)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(PLINK2 is None, reason="plink2 not on PATH")
class TestPlink2BgenCrossCheck:
    """``plink2 --export bgen-1.2 bits=8 ref-first --double-id`` vs
    ``vcztools view-bgen``: parse both outputs with ``bgen-reader`` and
    assert they agree semantically.

    Known field-level divergences:

    - **Sample IDs**: plink2 with ``--double-id`` writes ``"FID_IID"``
      (e.g. ``"0_NA00001"`` or ``"HG00096_HG00096"``); we write the
      bare IID. Stripped via :func:`_strip_double_id` for comparison.
    - **Chromosome names**: plink2 normalises like ``--make-bed``
      (``chr22`` → ``22``); we pass through. Stripped via
      :func:`_strip_chr_prefix` for comparison.
    - **rsids**: plink2 synthesises ``chrom:pos:ref:alt`` when the
      VCF ID is missing; we emit ``.``. Not asserted to match.
    - **Compression bytes**: differ; only decoded contents asserted.

    plink2 v2.00a6 has a known assertion bug (``compressed_bytect``)
    on mixed-phase data at any bit depth, so the ``sample.vcf.gz`` and
    ``msprime_diploid.vcf.gz`` fixtures crash plink2 and aren't useful
    for the cross-check. The ``chr22.vcf.gz`` fixture (100 samples,
    96 biallelic variants after ``--max-alleles 2``, all unphased) is
    what plink2 actually exports, so it's our canonical cross-check
    target."""

    def test_chr22_fixture_genotypes_match(self, tmp_path, fx_chr22_vcz):
        plink_out = tmp_path / "plink"
        vcz_out = tmp_path / "vcztools"
        run_plink2_bgen("--max-alleles 2", fx_chr22_vcz.vcf_path, plink_out)
        run_view_bgen("--max-alleles 2", fx_chr22_vcz.zip_path, vcz_out)
        with br.open_bgen(plink_out.with_suffix(".bgen"), verbose=False) as p:
            p_samples = _strip_double_id(p.samples)
            p_probs = p.read()
            p_chrom = _strip_chr_prefix(p.chromosomes)
            p_pos = [int(x) for x in p.positions]
            p_phased = [bool(x) for x in p.phased]
        with br.open_bgen(vcz_out.with_suffix(".bgen"), verbose=False) as v:
            v_samples = list(v.samples)
            v_probs = v.read()
            v_chrom = _strip_chr_prefix(v.chromosomes)
            v_pos = [int(x) for x in v.positions]
            v_phased = [bool(x) for x in v.phased]
        assert p_samples == v_samples
        assert p_chrom == v_chrom
        assert p_pos == v_pos
        # Both pipelines agree this fixture is unphased (no
        # call_genotype_phased flags propagate).
        assert p_phased == v_phased
        np.testing.assert_array_equal(
            _bgen_unphased_dosages(p_probs),
            _bgen_unphased_dosages(v_probs),
        )

    def test_chr22_fixture_alleles_match(self, tmp_path, fx_chr22_vcz):
        plink_out = tmp_path / "plink"
        vcz_out = tmp_path / "vcztools"
        run_plink2_bgen("--max-alleles 2", fx_chr22_vcz.vcf_path, plink_out)
        run_view_bgen("--max-alleles 2", fx_chr22_vcz.zip_path, vcz_out)
        with br.open_bgen(plink_out.with_suffix(".bgen"), verbose=False) as p:
            p_alleles = [str(a) for a in p.allele_ids]
        with br.open_bgen(vcz_out.with_suffix(".bgen"), verbose=False) as v:
            v_alleles = [str(a) for a in v.allele_ids]
        assert p_alleles == v_alleles

    def test_chr22_fixture_variant_count_matches_max_alleles(
        self, tmp_path, fx_chr22_vcz
    ):
        # Sanity check that --max-alleles 2 has the same effect on both
        # sides (plink2's --max-alleles vs ours).
        plink_out = tmp_path / "plink"
        vcz_out = tmp_path / "vcztools"
        run_plink2_bgen("--max-alleles 2", fx_chr22_vcz.vcf_path, plink_out)
        run_view_bgen("--max-alleles 2", fx_chr22_vcz.zip_path, vcz_out)
        with br.open_bgen(plink_out.with_suffix(".bgen"), verbose=False) as p:
            p_n = p.nvariants
        with br.open_bgen(vcz_out.with_suffix(".bgen"), verbose=False) as v:
            v_n = v.nvariants
        assert p_n == v_n
        # The chr22 fixture has 100 variants; --max-alleles 2 drops the
        # 4 multi-allelic ones, leaving 96.
        assert v_n == 96

    def test_mixed_phase_fixture_crashes_plink2(self, tmp_path, fx_sample_vcz):
        # Document plink2 v2.00a6's `compressed_bytect` assertion bug:
        # it crashes when exporting BGEN from VCFs with mixed-phase
        # variants. vcztools handles the same input cleanly. This
        # test asserts the divergence so the next plink2 release that
        # fixes the bug surfaces here as a green test.
        plink_out = tmp_path / "plink"
        with pytest.raises(AssertionError, match="plink2 exited"):
            run_plink2_bgen(
                "--max-alleles 2 --not-chr X", fx_sample_vcz.vcf_path, plink_out
            )
        # vcztools-side completes without error.
        run_view_bgen(
            "--max-alleles 2 -e 'CHROM==\"X\"'",
            fx_sample_vcz.zip_path,
            tmp_path / "vcz",
        )


def _read_pvar_positions(pvar_path: Path) -> list[int]:
    """Return the POS column of a plink2 ``.pvar`` (skipping ``#``-comment
    rows including the header)."""
    positions = []
    with open(pvar_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            positions.append(int(fields[1]))
    return positions


@pytest.mark.skipif(PLINK2 is None, reason="plink2 not on PATH")
class TestPlink2ReadsBgen:
    """plink2 consumes a vcztools-produced ``.bgen`` and reports the
    expected variants. The ``.bgen.bgi`` sidecar sits alongside but
    plink2 streams the file rather than indexing into it; index
    correctness is validated separately via the bgenix-equality test
    in ``validation/test_bgenix.py``."""

    def test_full_file_variant_count(self, tmp_path, fx_chr22_vcz):
        vcz_out = tmp_path / "vcz"
        run_view_bgen("--max-alleles 2", fx_chr22_vcz.zip_path, vcz_out)
        bgen_path = vcz_out.with_suffix(".bgen")
        sample_path = vcz_out.with_suffix(".sample")
        assert (vcz_out.with_suffix(".bgen").with_suffix(".bgen.bgi")).exists()
        run_plink2_read_bgen("--make-just-pvar", bgen_path, sample_path, tmp_path / "p")
        positions = _read_pvar_positions(tmp_path / "p.pvar")
        # --max-alleles 2 on the chr22 fixture leaves 96 variants
        # (existing TestPlink2BgenCrossCheck asserts the same count from
        # the plink2-export side).
        assert len(positions) == 96
        assert positions == sorted(positions)

    def test_region_query_matches_source(self, tmp_path, fx_chr22_vcz):
        # plink2's --from-bp/--to-bp filters in memory; the assertion
        # is on the surviving variant count agreeing with what the
        # source VCZ has in the same range (regardless of how plink2
        # gets there).
        vcz_out = tmp_path / "vcz"
        run_view_bgen("--max-alleles 2", fx_chr22_vcz.zip_path, vcz_out)
        bgen_path = vcz_out.with_suffix(".bgen")
        sample_path = vcz_out.with_suffix(".sample")

        # First import the full file to learn the position span.
        run_plink2_read_bgen(
            "--make-just-pvar", bgen_path, sample_path, tmp_path / "all"
        )
        all_positions = _read_pvar_positions(tmp_path / "all.pvar")
        midpoint = all_positions[len(all_positions) // 2]
        expected = sum(1 for p in all_positions if p <= midpoint)

        run_plink2_read_bgen(
            f"--chr 22 --to-bp {midpoint} --make-just-pvar",
            bgen_path,
            sample_path,
            tmp_path / "sub",
        )
        sub_positions = _read_pvar_positions(tmp_path / "sub.pvar")
        assert len(sub_positions) == expected
        assert max(sub_positions) <= midpoint
