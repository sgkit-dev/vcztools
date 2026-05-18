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


def _strip_padding(values: np.ndarray) -> np.ndarray:
    """Strip the encoder's padding fill from a column read back by
    bgenix / qctool.

    ``BgenEncoder`` writes the padding slot as ``b"." + pad_byte *
    (slack - 1)``, where ``slack`` varies per variant because the four
    actual-length fields do. Stripping leading ``"."`` characters
    collapses any such run to the empty string so a padding-slot
    column can be compared against the canonical empty value. The
    ``write_bgen`` path emits a single ``b"."`` which strips to the
    same canonical empty value.
    """
    return np.array([s.lstrip(".") for s in values], dtype=object)


class TestBgenixVariantIds:
    """Variant IDs round-trip through one of bgenix's ``rsid`` /
    ``alternate_ids`` columns, depending on which slot was selected at
    encode time. The other slot is the padding field — ``"."`` for the
    ``write_bgen`` path, or ``"." + pad_byte * (slack - 1)`` for
    ``BgenEncoder``. Stripping leading ``"."`` characters collapses
    both encodings to the empty string so the same assertion applies.
    """

    def _column_for_id(self, variant_id_field: str) -> str:
        # BGEN spec: a variant has two id slots, ``varid`` (BGEN's
        # primary id) and ``rsid``. bgenix surfaces them as
        # ``alternate_ids`` and ``rsid`` respectively.
        return "alternate_ids" if variant_id_field == "varid" else "rsid"

    @pytest.mark.parametrize("level", cfg.BGEN_LEVELS)
    def test_default_field_routes_variant_id_to_rsid(
        self, tmp_path, bgenix_bin, small_unphased_fixture, level
    ):
        # The session fixture builds BGEN with the default
        # variant_id_field=rsid, so rsid carries variant_id and
        # alternate_ids is the padding slot.
        src, _ = cfg.bgen_for_level(small_unphased_fixture, level)
        bgen = _stage_with_index(bgenix_bin, src, tmp_path / "x.bgen")
        df = _bgenix_list(bgenix_bin, bgen)
        ref = reference.compute_variant_stats(small_unphased_fixture.vcz_path)
        biallelic = ref.n_alleles == 2
        ids = reference.variant_ids(small_unphased_fixture.vcz_path)[biallelic]
        np.testing.assert_array_equal(df["rsid"].astype(str).to_numpy(), ids)
        stripped = _strip_padding(df["alternate_ids"].astype(str).to_numpy())
        np.testing.assert_array_equal(stripped, np.array([""] * len(ids), dtype=object))


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


class TestBgenixHaploid:
    """bgenix reads vcztools' uniform-haploid BGEN output. All three
    BGEN flavours are exercised (view-bgen lvl=-1, lvl=0, BgenEncoder).
    Allele set and ALT-allele frequency are unchanged from the diploid
    source — `compute_variant_stats` treats slot-0-only haploid stores
    the same way it treats diploid (negative values are missing)."""

    @pytest.mark.parametrize("level", cfg.BGEN_LEVELS)
    def test_variant_positions_match_reference(
        self, tmp_path, bgenix_bin, haploid_fixture, level
    ):
        src, _ = cfg.bgen_for_level(haploid_fixture, level)
        bgen = _stage_with_index(bgenix_bin, src, tmp_path / "x.bgen")
        df = _bgenix_list(bgenix_bin, bgen)
        ref = reference.compute_variant_stats(haploid_fixture.vcz_path)
        biallelic = ref.n_alleles == 2
        assert len(df) == int(biallelic.sum())
        np.testing.assert_array_equal(df["position"].to_numpy(), ref.pos[biallelic])

    @pytest.mark.parametrize("level", cfg.BGEN_LEVELS)
    def test_alleles_match_reference(
        self, tmp_path, bgenix_bin, haploid_fixture, level
    ):
        src, _ = cfg.bgen_for_level(haploid_fixture, level)
        bgen = _stage_with_index(bgenix_bin, src, tmp_path / "x.bgen")
        df = _bgenix_list(bgenix_bin, bgen)
        ref = reference.compute_variant_stats(haploid_fixture.vcz_path)
        biallelic = ref.n_alleles == 2
        np.testing.assert_array_equal(
            df["first_allele"].astype(str).to_numpy(), ref.ref[biallelic]
        )
        np.testing.assert_array_equal(
            df["alternative_alleles"].astype(str).to_numpy(), ref.alt[biallelic]
        )

    @pytest.mark.parametrize("level", cfg.BGEN_LEVELS)
    def test_variant_table_matches_bgenix(
        self, tmp_path, bgenix_bin, haploid_fixture, level
    ):
        bgen_src, _ = cfg.bgen_for_level(haploid_fixture, level)
        vcztools_bgi = pathlib.Path(str(bgen_src) + ".bgi")
        assert vcztools_bgi.exists(), (
            f"expected vcztools-produced .bgi alongside {bgen_src}"
        )

        bgenix_bgen = _stage_with_index(bgenix_bin, bgen_src, tmp_path / "ref.bgen")
        bgenix_bgi = pathlib.Path(str(bgenix_bgen) + ".bgi")

        ours = _read_variant_table(vcztools_bgi)
        theirs = _read_variant_table(bgenix_bgi)
        assert ours == theirs


class TestBgenixMixedPloidy:
    """bgenix reads vcztools' mixed-ploidy BGEN output. Only the two
    view-bgen flavours are exercised: BgenEncoder is uniform-only and
    has no mixed-ploidy build (see ``_build_bgen_only_outputs`` /
    ``vcztools.bgen.BgenEncoder``)."""

    @pytest.mark.parametrize("level", cfg.BGEN_CLI_LEVELS)
    def test_variant_positions_match_reference(
        self, tmp_path, bgenix_bin, mixed_ploidy_fixture, level
    ):
        src, _ = cfg.bgen_for_level(mixed_ploidy_fixture, level)
        bgen = _stage_with_index(bgenix_bin, src, tmp_path / "x.bgen")
        df = _bgenix_list(bgenix_bin, bgen)
        ref = reference.compute_variant_stats(mixed_ploidy_fixture.vcz_path)
        biallelic = ref.n_alleles == 2
        assert len(df) == int(biallelic.sum())
        np.testing.assert_array_equal(df["position"].to_numpy(), ref.pos[biallelic])

    @pytest.mark.parametrize("level", cfg.BGEN_CLI_LEVELS)
    def test_alleles_match_reference(
        self, tmp_path, bgenix_bin, mixed_ploidy_fixture, level
    ):
        src, _ = cfg.bgen_for_level(mixed_ploidy_fixture, level)
        bgen = _stage_with_index(bgenix_bin, src, tmp_path / "x.bgen")
        df = _bgenix_list(bgenix_bin, bgen)
        ref = reference.compute_variant_stats(mixed_ploidy_fixture.vcz_path)
        biallelic = ref.n_alleles == 2
        np.testing.assert_array_equal(
            df["first_allele"].astype(str).to_numpy(), ref.ref[biallelic]
        )
        np.testing.assert_array_equal(
            df["alternative_alleles"].astype(str).to_numpy(), ref.alt[biallelic]
        )

    @pytest.mark.parametrize("level", cfg.BGEN_CLI_LEVELS)
    def test_variant_table_matches_bgenix(
        self, tmp_path, bgenix_bin, mixed_ploidy_fixture, level
    ):
        bgen_src, _ = cfg.bgen_for_level(mixed_ploidy_fixture, level)
        vcztools_bgi = pathlib.Path(str(bgen_src) + ".bgi")
        assert vcztools_bgi.exists(), (
            f"expected vcztools-produced .bgi alongside {bgen_src}"
        )

        bgenix_bgen = _stage_with_index(bgenix_bin, bgen_src, tmp_path / "ref.bgen")
        bgenix_bgi = pathlib.Path(str(bgenix_bgen) + ".bgi")

        ours = _read_variant_table(vcztools_bgi)
        theirs = _read_variant_table(bgenix_bgi)
        assert ours == theirs


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


def _sort_by_chrom_pos(
    chrom: np.ndarray, pos: np.ndarray, *extras: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Return ``(chrom, pos, *extras)`` reordered into bgenix's output
    order: ASCII-lexicographic by ``chrom``, then ascending ``pos``
    within each chrom.

    bgenix's ``-list`` walks the .bgi in ``(chromosome, position)`` sort
    order regardless of the BGEN file's variant ordering, so any
    multi-contig fixture's reference needs reordering before a row-wise
    comparison against the listed output.
    """
    order = np.lexsort((pos, chrom))
    return (chrom[order], pos[order], *(arr[order] for arr in extras))


class TestBgenixStringVariety:
    """Round-trip every BGEN string field through bgenix on the
    ``varied_strings`` fixture, where:

    - chrom byte length varies across 4 contigs (1, 6, 1, 21 bytes).
    - allele1 / allele2 byte length varies through 6 indel / SNP
      recipes (1–10 bytes).
    - variant_id byte length varies through three template buckets
      (~3, 15, 46 bytes).

    Parametrising over ``(level, variant_id_field)`` exercises both
    BGEN generation paths (``write_bgen`` at lvl=-1 / lvl=0,
    ``BgenEncoder``) and both id-routing modes, so each of the five
    BGEN string slots is the carrier of either ``variant_id`` or the
    padding field in at least one parametrisation.

    ``bgenix -list`` emits variants in ``(chromosome, position)``
    ASCII-sorted order, not BGEN file order, so the reference is
    reordered the same way before per-row comparisons.
    """

    def _carrier_column(self, variant_id_field: str) -> str:
        return "alternate_ids" if variant_id_field == "varid" else "rsid"

    def _padding_column(self, variant_id_field: str) -> str:
        return "rsid" if variant_id_field == "varid" else "alternate_ids"

    @pytest.mark.parametrize("variant_id_field", cfg.VARIANT_ID_FIELDS)
    @pytest.mark.parametrize("level", cfg.BGEN_LEVELS)
    def test_chrom_and_alleles_round_trip(
        self, tmp_path, bgenix_bin, varied_strings_fixture, level, variant_id_field
    ):
        src, _ = cfg.bgen_for_field(varied_strings_fixture, level, variant_id_field)
        bgen = _stage_with_index(bgenix_bin, src, tmp_path / "x.bgen")
        df = _bgenix_list(bgenix_bin, bgen)
        ref = reference.compute_variant_stats(varied_strings_fixture.vcz_path)
        biallelic = ref.n_alleles == 2
        assert len(df) == int(biallelic.sum())

        chrom_ref, pos_ref, ref_a1, ref_a2 = _sort_by_chrom_pos(
            ref.chrom[biallelic],
            ref.pos[biallelic],
            ref.ref[biallelic],
            ref.alt[biallelic],
        )
        np.testing.assert_array_equal(
            df["chromosome"].astype(str).to_numpy(), chrom_ref
        )
        np.testing.assert_array_equal(df["position"].to_numpy(), pos_ref)
        np.testing.assert_array_equal(df["first_allele"].astype(str).to_numpy(), ref_a1)
        np.testing.assert_array_equal(
            df["alternative_alleles"].astype(str).to_numpy(), ref_a2
        )

    @pytest.mark.parametrize("variant_id_field", cfg.VARIANT_ID_FIELDS)
    @pytest.mark.parametrize("level", cfg.BGEN_LEVELS)
    def test_variant_id_round_trip(
        self, tmp_path, bgenix_bin, varied_strings_fixture, level, variant_id_field
    ):
        src, _ = cfg.bgen_for_field(varied_strings_fixture, level, variant_id_field)
        bgen = _stage_with_index(bgenix_bin, src, tmp_path / "x.bgen")
        df = _bgenix_list(bgenix_bin, bgen)
        ref = reference.compute_variant_stats(varied_strings_fixture.vcz_path)
        biallelic = ref.n_alleles == 2
        ids = reference.variant_ids(varied_strings_fixture.vcz_path)[biallelic]
        _, _, ids_sorted = _sort_by_chrom_pos(
            ref.chrom[biallelic], ref.pos[biallelic], ids
        )

        carrier = self._carrier_column(variant_id_field)
        padding = self._padding_column(variant_id_field)
        np.testing.assert_array_equal(df[carrier].astype(str).to_numpy(), ids_sorted)
        stripped = _strip_padding(df[padding].astype(str).to_numpy())
        np.testing.assert_array_equal(
            stripped, np.array([""] * len(ids_sorted), dtype=object)
        )

    @pytest.mark.parametrize("variant_id_field", cfg.VARIANT_ID_FIELDS)
    @pytest.mark.parametrize("level", cfg.BGEN_LEVELS)
    def test_variant_table_matches_bgenix(
        self, tmp_path, bgenix_bin, varied_strings_fixture, level, variant_id_field
    ):
        # The vcztools-produced .bgi must match bgenix's own reindex
        # byte-for-byte in both id-routing modes. When
        # variant_id_field="varid", the BGEN rsid slot holds the
        # padding field; write_bgi reconstructs that pattern so the
        # .bgi's rsid column matches what bgenix records.
        bgen_src, _ = cfg.bgen_for_field(
            varied_strings_fixture, level, variant_id_field
        )
        vcztools_bgi = pathlib.Path(str(bgen_src) + ".bgi")
        assert vcztools_bgi.exists(), (
            f"expected vcztools-produced .bgi alongside {bgen_src}"
        )
        bgenix_bgen = _stage_with_index(bgenix_bin, bgen_src, tmp_path / "ref.bgen")
        bgenix_bgi = pathlib.Path(str(bgenix_bgen) + ".bgi")
        ours = _read_variant_table(vcztools_bgi)
        theirs = _read_variant_table(bgenix_bgi)
        assert ours == theirs
