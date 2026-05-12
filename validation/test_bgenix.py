"""bgenix reads vcztools view-bgen output and lists the same variants
we wrote.

Three checks per compression level:

1. ``bgenix -list`` emits a metadata row per variant. The variant
   positions match the source VCZ after the biallelic filter.
2. ``bgenix -index`` builds a ``.bgen.bgi`` index from scratch
   (overwriting the one ``view-bgen`` produces). The reindexed file
   then lists the same set of variants.
3. ``bgenix -incl-range`` returns a per-range subset whose variant
   count matches the count we'd compute from the source VCZ for the
   same range.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from . import helpers, reference


def _bgenix_list(bgenix: pathlib.Path, bgen: pathlib.Path) -> pd.DataFrame:
    result = helpers.run_tool([str(bgenix), "-g", str(bgen), "-list"])
    # Strip the per-line `#`-commented bgenix header and the
    # `# bgenix: success ...` trailer; what remains is a TSV with a
    # header row starting "alternate_ids".
    lines = [
        line for line in result.stdout.splitlines() if line and not line.startswith("#")
    ]
    text = "\n".join(lines) + "\n"
    return pd.read_csv(pd.io.common.StringIO(text), sep="\t")


class TestBgenixList:
    @pytest.mark.parametrize("level", [-1, 0], ids=["lvl=-1", "lvl=0"])
    def test_variant_positions_match_reference(self, bgenix_bin, small_fixture, level):
        bgen = small_fixture.bgen_minus1 if level == -1 else small_fixture.bgen_stored
        df = _bgenix_list(bgenix_bin, bgen)
        ref = reference.compute_variant_stats(small_fixture.vcz_path)
        biallelic = ref.n_alleles == 2
        assert len(df) == int(biallelic.sum())
        np.testing.assert_array_equal(df["position"].to_numpy(), ref.pos[biallelic])

    @pytest.mark.parametrize("level", [-1, 0], ids=["lvl=-1", "lvl=0"])
    def test_alleles_match_reference(self, bgenix_bin, small_fixture, level):
        bgen = small_fixture.bgen_minus1 if level == -1 else small_fixture.bgen_stored
        df = _bgenix_list(bgenix_bin, bgen)
        ref = reference.compute_variant_stats(small_fixture.vcz_path)
        biallelic = ref.n_alleles == 2
        # vcztools writes alleles ref-first: first_allele = REF,
        # alternative_alleles = ALT (comma-separated; biallelic so just
        # one entry per row).
        np.testing.assert_array_equal(
            df["first_allele"].astype(str).to_numpy(),
            ref.ref[biallelic],
        )
        np.testing.assert_array_equal(
            df["alternative_alleles"].astype(str).to_numpy(),
            ref.alt[biallelic],
        )


class TestBgenixIndex:
    @pytest.mark.parametrize("level", [-1, 0], ids=["lvl=-1", "lvl=0"])
    def test_reindex_then_list_agrees(self, tmp_path, bgenix_bin, small_fixture, level):
        # Copy the BGEN into tmp so we don't clobber the .bgen.bgi
        # written by view-bgen for other tests in the session.
        bgen_src = (
            small_fixture.bgen_minus1 if level == -1 else small_fixture.bgen_stored
        )
        bgen = tmp_path / "x.bgen"
        bgen.write_bytes(bgen_src.read_bytes())
        # `-clobber` lets bgenix overwrite a stale index.
        helpers.run_tool([str(bgenix_bin), "-g", str(bgen), "-index", "-clobber"])
        bgi = bgen.with_suffix(".bgen.bgi")
        assert bgi.exists(), f"bgenix did not write {bgi}"
        df = _bgenix_list(bgenix_bin, bgen)
        ref = reference.compute_variant_stats(small_fixture.vcz_path)
        biallelic = ref.n_alleles == 2
        assert len(df) == int(biallelic.sum())


class TestBgenixIncludeRange:
    @pytest.mark.parametrize("level", [-1, 0], ids=["lvl=-1", "lvl=0"])
    def test_range_subset_count_matches(
        self, tmp_path, bgenix_bin, small_fixture, level
    ):
        # Pick a position interval covering the first half of the
        # variants and count the survivors.
        ref = reference.compute_variant_stats(small_fixture.vcz_path)
        biallelic = ref.n_alleles == 2
        positions = ref.pos[biallelic]
        midpoint = int(positions[len(positions) // 2])

        bgen = small_fixture.bgen_minus1 if level == -1 else small_fixture.bgen_stored
        # bgenix -incl-range emits a subset BGEN to stdout; capture
        # the raw bytes directly into a file and re-list to count.
        out = tmp_path / "sub.bgen"
        helpers.run_tool_capture_binary(
            [
                str(bgenix_bin),
                "-g",
                str(bgen),
                "-incl-range",
                f"1:1-{midpoint}",
            ],
            out,
        )
        # bgenix needs an index to read its own output again.
        helpers.run_tool([str(bgenix_bin), "-g", str(out), "-index", "-clobber"])
        df = _bgenix_list(bgenix_bin, out)
        # Reference: biallelic variants on contig "1" with pos <= midpoint.
        expected = int(((ref.chrom == "1") & (ref.pos <= midpoint) & biallelic).sum())
        assert len(df) == expected
