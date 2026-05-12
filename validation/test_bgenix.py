"""bgenix reads vcztools view-bgen / BgenEncoder output and lists the
same variants we wrote.

Three checks per BGEN flavour (CLI lvl=-1, CLI lvl=0, encoder):

1. ``bgenix -list`` emits a metadata row per variant. Positions and
   alleles match the source VCZ after the biallelic filter.
2. ``bgenix -index`` builds a ``.bgen.bgi`` index from scratch
   (overwriting any pre-existing one). The reindexed file then lists
   the same set of variants.
3. ``bgenix -incl-range`` returns a per-range subset whose variant
   count matches the count we'd compute from the source VCZ for the
   same range.

A separate class checks variant IDs round-trip through bgenix's
``rsid`` and ``alternate_ids`` columns. Another asserts that the
``Variant`` table of the ``.bgi`` we produce is row-equal to the one
bgenix builds from the same ``.bgen``.

All tests stage the BGEN into ``tmp_path`` and rebuild the ``.bgi``
index there. Reindexing the source outputs in-place would clobber the
index for other tests in the session.
"""

from __future__ import annotations

import pathlib
import sqlite3

import numpy as np
import pandas as pd
import pytest

from . import conftest as cfg
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


def _stage_with_index(
    bgenix: pathlib.Path, src: pathlib.Path, dst: pathlib.Path
) -> pathlib.Path:
    """Copy ``src`` to ``dst`` and build a fresh .bgen.bgi alongside it."""
    dst.write_bytes(src.read_bytes())
    helpers.run_tool([str(bgenix), "-g", str(dst), "-index", "-clobber"])
    return dst


class TestBgenixList:
    @pytest.mark.parametrize("level", cfg.BGEN_LEVELS)
    def test_variant_positions_match_reference(
        self, tmp_path, bgenix_bin, small_unphased_fixture, level
    ):
        src, _ = cfg.bgen_for_level(small_unphased_fixture, level)
        bgen = _stage_with_index(bgenix_bin, src, tmp_path / "x.bgen")
        df = _bgenix_list(bgenix_bin, bgen)
        ref = reference.compute_variant_stats(small_unphased_fixture.vcz_path)
        biallelic = ref.n_alleles == 2
        assert len(df) == int(biallelic.sum())
        np.testing.assert_array_equal(df["position"].to_numpy(), ref.pos[biallelic])

    @pytest.mark.parametrize("level", cfg.BGEN_LEVELS)
    def test_alleles_match_reference(
        self, tmp_path, bgenix_bin, small_unphased_fixture, level
    ):
        src, _ = cfg.bgen_for_level(small_unphased_fixture, level)
        bgen = _stage_with_index(bgenix_bin, src, tmp_path / "x.bgen")
        df = _bgenix_list(bgenix_bin, bgen)
        ref = reference.compute_variant_stats(small_unphased_fixture.vcz_path)
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
    @pytest.mark.parametrize("level", cfg.BGEN_LEVELS)
    def test_reindex_then_list_agrees(
        self, tmp_path, bgenix_bin, small_unphased_fixture, level
    ):
        src, _ = cfg.bgen_for_level(small_unphased_fixture, level)
        bgen = _stage_with_index(bgenix_bin, src, tmp_path / "x.bgen")
        bgi = bgen.with_suffix(".bgen.bgi")
        assert bgi.exists(), f"bgenix did not write {bgi}"
        df = _bgenix_list(bgenix_bin, bgen)
        ref = reference.compute_variant_stats(small_unphased_fixture.vcz_path)
        biallelic = ref.n_alleles == 2
        assert len(df) == int(biallelic.sum())


class TestBgenixIncludeRange:
    @pytest.mark.parametrize("level", cfg.BGEN_LEVELS)
    def test_range_subset_count_matches(
        self, tmp_path, bgenix_bin, small_unphased_fixture, level
    ):
        # Pick a position interval covering the first half of the
        # variants and count the survivors.
        ref = reference.compute_variant_stats(small_unphased_fixture.vcz_path)
        biallelic = ref.n_alleles == 2
        positions = ref.pos[biallelic]
        midpoint = int(positions[len(positions) // 2])

        src, _ = cfg.bgen_for_level(small_unphased_fixture, level)
        bgen = _stage_with_index(bgenix_bin, src, tmp_path / "x.bgen")
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


class TestBgenixVariantIds:
    @pytest.mark.parametrize("level", cfg.BGEN_LEVELS)
    def test_rsid_and_alternate_ids_match_reference(
        self, tmp_path, bgenix_bin, small_unphased_fixture, level
    ):
        src, _ = cfg.bgen_for_level(small_unphased_fixture, level)
        bgen = _stage_with_index(bgenix_bin, src, tmp_path / "x.bgen")
        df = _bgenix_list(bgenix_bin, bgen)
        ref = reference.compute_variant_stats(small_unphased_fixture.vcz_path)
        biallelic = ref.n_alleles == 2
        ids = reference.variant_ids(small_unphased_fixture.vcz_path)[biallelic]
        # vcztools (both write_bgen and BgenEncoder) sets BGEN's varid
        # and rsid fields to the same value from variant_id; bgenix
        # exposes them as alternate_ids and rsid respectively.
        np.testing.assert_array_equal(df["rsid"].astype(str).to_numpy(), ids)
        np.testing.assert_array_equal(df["alternate_ids"].astype(str).to_numpy(), ids)


class TestBgenixPhasedBgen:
    """bgenix is phase-agnostic — its ``-list`` operation reads metadata
    only and never decodes probability data. The phased fixture should
    list the same biallelic-survivor count as the unphased one."""

    def test_lists_phased_variants(self, tmp_path, bgenix_bin, small_phased_fixture):
        bgen = _stage_with_index(
            bgenix_bin, small_phased_fixture.bgen_minus1, tmp_path / "phased.bgen"
        )
        df = _bgenix_list(bgenix_bin, bgen)
        ref = reference.compute_variant_stats(small_phased_fixture.vcz_path)
        biallelic = ref.n_alleles == 2
        assert len(df) == int(biallelic.sum())
        np.testing.assert_array_equal(df["position"].to_numpy(), ref.pos[biallelic])


def _read_variant_table(bgi_path: pathlib.Path) -> list[tuple]:
    """Read the Variant table, stripping trailing NULs from every TEXT
    column. ``BgenEncoder`` NUL-pads variable-length BGEN string fields
    (varid, rsid, chromosome, alleles) to a fixed width so the variant
    block is a constant size. bgenix indexes the on-disk bytes verbatim
    and so records the padded form; we record canonical values in our
    own ``.bgi``. The padding is BGEN-spec-permitted (the reference
    bgen-reader strips it on read) but it would otherwise make a
    byte-exact comparison spurious. Stripping on both sides aligns
    canonical and on-disk-byte representations for the equality
    assertion."""
    conn = sqlite3.connect(str(bgi_path))
    try:
        rows = conn.execute(
            "SELECT chromosome, position, rsid, number_of_alleles, "
            "allele1, allele2, file_start_position, size_in_bytes "
            "FROM Variant ORDER BY file_start_position"
        ).fetchall()
    finally:
        conn.close()
    return [
        (
            chrom.rstrip("\x00"),
            pos,
            rsid.rstrip("\x00"),
            n_alleles,
            a1.rstrip("\x00"),
            a2.rstrip("\x00") if a2 is not None else a2,
            start,
            size,
        )
        for chrom, pos, rsid, n_alleles, a1, a2, start, size in rows
    ]


class TestBgenixIndexEquality:
    """The ``.bgi`` we produce alongside each BGEN flavour is row-equal
    to the one bgenix builds from the same ``.bgen``.

    For each flavour: the vcztools-written ``.bgi`` lives next to the
    fixture's ``.bgen``. A separate copy of the ``.bgen`` is staged
    into ``tmp_path`` and reindexed by bgenix; the two ``Variant``
    tables are then compared row-by-row including
    ``file_start_position`` and ``size_in_bytes`` (the BGEN file is
    byte-identical, so byte offsets must match exactly)."""

    @pytest.mark.parametrize("level", cfg.BGEN_LEVELS)
    def test_variant_table_matches_bgenix(
        self, tmp_path, bgenix_bin, small_unphased_fixture, level
    ):
        bgen_src, _ = cfg.bgen_for_level(small_unphased_fixture, level)
        vcztools_bgi = pathlib.Path(str(bgen_src) + ".bgi")
        assert vcztools_bgi.exists(), (
            f"expected vcztools-produced .bgi alongside {bgen_src}"
        )

        bgenix_bgen = _stage_with_index(bgenix_bin, bgen_src, tmp_path / "ref.bgen")
        bgenix_bgi = pathlib.Path(str(bgenix_bgen) + ".bgi")

        ours = _read_variant_table(vcztools_bgi)
        theirs = _read_variant_table(bgenix_bgi)
        assert ours == theirs
