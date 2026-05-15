import copy
import pathlib
import sqlite3
import sys
from unittest import mock

import bgen_reader as br
import click
import click.testing as ct
import numpy as np
import pytest

import vcztools.cli as cli
from tests.test_bcftools_validation import run_vcztools
from vcztools import bcftools_filter, provenance

IS_WINDOWS = sys.platform == "win32"


@pytest.fixture
def fx_vcz_path(fx_sample_vcz):
    # Return a posix-form string so f-string interpolation into CLI
    # arg strings is safe on Windows (CliRunner shlex-splits with
    # posix=True, which treats backslashes as escape characters).
    return fx_sample_vcz.zip_path.as_posix()


def test_version_header(fx_vcz_path):
    output, _ = run_vcztools(f"view {fx_vcz_path}")
    assert output.find("##vcztools_viewCommand=") >= 0
    assert output.find("Date=") >= 0


class TestOutput:
    def test_view_unsupported_output(self, tmp_path, fx_vcz_path):
        bad_output = tmp_path / "output.vcf.gz"

        _, vcztools_error = run_vcztools(
            f"view --no-version {fx_vcz_path} -o {bad_output.as_posix()}",
            expect_error=True,
        )
        assert (
            "Only uncompressed VCF output supported, suffix .gz not allowed"
            in vcztools_error
        )

    @pytest.mark.parametrize("suffix", ["gz", "bgz", "bcf"])
    def test_view_unsupported_output_suffix(self, tmp_path, fx_vcz_path, suffix):
        bad_output = tmp_path / f"output.vcf.{suffix}"

        _, vcztools_error = run_vcztools(
            f"view --no-version {fx_vcz_path} -o {bad_output.as_posix()}",
            expect_error=True,
        )
        assert f".{suffix} not allowed" in vcztools_error

    def test_view_good_path(self, tmp_path, fx_vcz_path):
        output_path = tmp_path / "tmp.vcf"
        runner = ct.CliRunner()
        result = runner.invoke(
            cli.vcztools_main,
            f"view --no-version {fx_vcz_path} -o {output_path.as_posix()}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 0
        assert output_path.exists()

    def test_view_write_directory(self, tmp_path, fx_vcz_path):
        runner = ct.CliRunner()
        result = runner.invoke(
            cli.vcztools_main,
            f"view --no-version {fx_vcz_path} -o {tmp_path.as_posix()}",
            catch_exceptions=False,
        )
        assert result.exit_code == 1
        assert len(result.stdout) == 0
        expected = "Permission denied" if IS_WINDOWS else "Is a directory"
        assert expected in result.stderr

    def test_view_write_pipe(self, tmp_path, fx_vcz_path):
        runner = ct.CliRunner()
        result = runner.invoke(
            cli.vcztools_main,
            f"view --no-version {fx_vcz_path} -o {tmp_path.as_posix()}",
            catch_exceptions=False,
        )
        assert result.exit_code == 1
        assert len(result.stdout) == 0
        expected = "Permission denied" if IS_WINDOWS else "Is a directory"
        assert expected in result.stderr


def test_samples_and_drop_genotypes(fx_vcz_path):
    _, vcztools_error = run_vcztools(
        f"view -s NA00001 -G {fx_vcz_path}",
        expect_error=True,
    )
    assert "Cannot select samples and drop genotypes" in vcztools_error


def test_drop_genotypes_rejects_sample_scope_filter(fx_vcz_path):
    # --drop-genotypes with a sample-scope filter (FMT/DP) is
    # incompatible: the filter needs per-sample genotype data which
    # --drop-genotypes says we don't want.
    with pytest.raises(ValueError, match="sample-scope variant_filter is incompatible"):
        cli.make_reader(
            fx_vcz_path,
            include="FMT/DP>3",
            drop_genotypes=True,
        )


class TestFilterErrors:
    """Pins the user-visible contract for filter expression errors:
    exit code 1 plus a specific error-message substring in stderr.
    Covers every raise site in :mod:`vcztools.bcftools_filter`
    reachable from the CLI, and confirms both ``view`` and ``query``
    route errors through the same ``handle_exception`` decorator
    (``vcztools/cli.py:35-49``) that converts ``ValueError`` into a
    ``click.ClickException``.
    """

    # fmt: off
    @pytest.mark.parametrize(
        ("args", "expected"),
        [
            # Parse-grammar failure (pyparsing → ParseError).
            ('view -i "| |"', "parse error"),
            # Unknown identifier (bare name not in VCZ store).
            ("view -i 'BLAH>0'", 'the tag "BLAH" is not defined'),
            # Ambiguous bare identifier — sample.vcf has both
            # INFO/DP and FORMAT/DP.
            ("view -i 'DP>0'", "ambiguous filtering expression"),
            # Include + exclude both set.
            ("view -i 'POS>0' -e 'POS<1000'", "both an include"),
            # Filter value not present in header.
            ("""view -i 'FILTER="NOPE"'""",
             'The filter "NOPE" is not present'),
            # Each UnsupportedFilteringFeatureError subclass reachable
            # at parse time. The identifiers here are all real fields
            # in sample.vcz so the error we hit is the intended one
            # rather than "tag not defined".
            ("view -i 'GT==0'", "Genotype values"),
            ("""view -i 'ID="."'""", "Missing data"),
            ("""view -i 'TYPE="bnd"'""", "TYPE field"),
            ("view -i 'INFO/AC[0]==1'", "Array subscripts"),
            ("view -i 'POS~0'", "Regular expressions"),
            ("view -i 'ID!=@file'", "File references"),
            ("view -i 'binom(POS)'", "Function evaluation"),
            # Cross-command: ``query`` must surface the same error
            # through the same handler.
            ("query -f '%POS\\n' -i 'BLAH>0'",
             'the tag "BLAH" is not defined'),
        ],
    )
    # fmt: on
    def test_filter_error(self, fx_vcz_path, args, expected):
        runner = ct.CliRunner()
        result = runner.invoke(
            cli.vcztools_main,
            f"{args} {fx_vcz_path}",
            catch_exceptions=False,
        )
        assert result.exit_code == 1
        assert expected in result.stderr


class TestMakeReader:
    """Translation of bcftools-style ``^`` prefixes into ``targets_complement``."""

    @staticmethod
    def _positions(reader):
        chunks = list(reader.variant_chunks(fields=["variant_position"]))
        if len(chunks) == 0:
            return []

        return list(np.concatenate([c["variant_position"] for c in chunks]))

    def test_targets_string_with_caret(self, fx_vcz_path):
        # ^19:112 excludes exactly one variant (19:112).
        reader = cli.make_reader(fx_vcz_path, targets="^19:112")
        assert 112 not in self._positions(reader)

    def test_targets_string_without_caret(self, fx_vcz_path):
        # 19:112 selects exactly that one variant.
        reader = cli.make_reader(fx_vcz_path, targets="19:112")
        assert self._positions(reader) == [112]

    def test_targets_file_with_caret(self, fx_vcz_path):
        reader = cli.make_reader(
            fx_vcz_path,
            targets_file="^tests/data/txt/regions-3col.tsv",
        )
        # Complement applied → at least one chunk has a real row
        # selection rather than the full-chunk sentinel.
        assert any(cr.selection is not None for cr in reader.variant_chunk_plan)

    def test_targets_file_without_caret(self, fx_vcz_path):
        reader = cli.make_reader(
            fx_vcz_path,
            targets_file="tests/data/txt/regions-3col.tsv",
        )
        assert any(cr.selection is not None for cr in reader.variant_chunk_plan)

    def test_regions_file(self, fx_vcz_path):
        reader = cli.make_reader(
            fx_vcz_path,
            regions_file="tests/data/txt/regions-3col.tsv",
        )
        assert any(cr.selection is not None for cr in reader.variant_chunk_plan)

    def test_regions_string_splits_on_commas(self, fx_vcz_path):
        # Both contig 19 (positions 111, 112) and X (position 10) selected.
        reader = cli.make_reader(fx_vcz_path, regions="19,X")
        assert sorted(self._positions(reader)) == [10, 111, 112]

    def test_targets_string_splits_on_commas(self, fx_vcz_path):
        reader = cli.make_reader(fx_vcz_path, targets="19,X")
        assert sorted(self._positions(reader)) == [10, 111, 112]

    def test_targets_caret_and_comma(self, fx_vcz_path):
        # Complement of {19 ∪ X} = everything on chromosome 20.
        reader = cli.make_reader(fx_vcz_path, targets="^19,X")
        positions = self._positions(reader)
        # None of the 19 or X variants remain.
        assert 111 not in positions
        assert 112 not in positions
        assert 10 not in positions
        # At least one 20:... variant survives.
        assert any(p > 1000 for p in positions)

    def test_samples_string_splits_on_commas(self, fx_vcz_path):
        reader = cli.make_reader(fx_vcz_path, samples="NA00001,NA00003")
        assert list(reader.sample_ids) == ["NA00001", "NA00003"]

    def test_samples_caret_translates_to_complement(self, fx_vcz_path):
        reader = cli.make_reader(fx_vcz_path, samples="^NA00002")
        assert list(reader.sample_ids) == ["NA00001", "NA00003"]

    def test_samples_file_with_caret(self, fx_vcz_path):
        reader = cli.make_reader(
            fx_vcz_path,
            samples_file="^tests/data/txt/samples.txt",
        )
        # samples.txt lists NA00001, NA00003 -> complement leaves NA00002
        assert list(reader.sample_ids) == ["NA00002"]

    def test_samples_file_without_caret(self, fx_vcz_path):
        reader = cli.make_reader(
            fx_vcz_path,
            samples_file="tests/data/txt/samples.txt",
        )
        assert list(reader.sample_ids) == ["NA00001", "NA00003"]

    def test_regions_and_regions_file_mutually_exclusive(self, fx_vcz_path):
        with pytest.raises(ValueError, match="Cannot specify both"):
            cli.make_reader(
                fx_vcz_path,
                regions="19",
                regions_file="tests/data/txt/regions-3col.tsv",
            )

    def test_targets_and_targets_file_mutually_exclusive(self, fx_vcz_path):
        with pytest.raises(ValueError, match="Cannot specify both"):
            cli.make_reader(
                fx_vcz_path,
                targets="19",
                targets_file="tests/data/txt/regions-3col.tsv",
            )

    def test_min_alleles_excludes_monomorphic(self, fx_vcz_path):
        # sample.vcf has two REF-only variants at 20:1230237 and
        # 20:1235237 (T → .). -m 2 (>= 2 alleles) drops them.
        reader = cli.make_reader(fx_vcz_path, min_alleles=2)
        positions = self._positions(reader)
        assert 1230237 not in positions
        assert 1235237 not in positions

    def test_max_alleles_keeps_biallelic(self, fx_vcz_path):
        # -M 2 (<= 2 alleles) drops the multiallelics at
        # 20:1110696, 20:1234567 and X:10.
        reader = cli.make_reader(fx_vcz_path, max_alleles=2)
        positions = self._positions(reader)
        assert 1110696 not in positions
        assert 1234567 not in positions
        assert 10 not in positions

    def test_min_and_max_alleles_biallelic_only(self, fx_vcz_path):
        # -m 2 -M 2 keeps strictly biallelic sites.
        reader = cli.make_reader(fx_vcz_path, min_alleles=2, max_alleles=2)
        positions = self._positions(reader)
        for excluded in (1230237, 1235237, 1110696, 1234567, 10):
            assert excluded not in positions

    def test_types_snps_keeps_snp_sites(self, fx_vcz_path):
        # 20:1234567 is microsat (G→GA,GAC) and X:10 is AC→A,ATG,C —
        # neither has any SNP allele, so both drop with -v snps.
        reader = cli.make_reader(fx_vcz_path, types="snps")
        positions = self._positions(reader)
        assert 1234567 not in positions
        assert 10 not in positions

    def test_exclude_types_snps(self, fx_vcz_path):
        # The biallelic SNP at 19:111 has TYPE=snp, so it drops with -V snps.
        reader = cli.make_reader(fx_vcz_path, exclude_types="snps")
        positions = self._positions(reader)
        assert 111 not in positions

    def test_unsupported_type_keyword_surfaces_parser_error(self, fx_vcz_path):
        # ``indels`` is a bcftools-valid keyword but the TYPE operator
        # only handles 'ref' and 'snp' today (issue #166), so the
        # filter parser raises UnsupportedTypeFieldError.
        with pytest.raises(bcftools_filter.UnsupportedTypeFieldError):
            cli.make_reader(fx_vcz_path, types="indels")

    def test_min_alleles_below_one_rejected(self, fx_vcz_path):
        with pytest.raises(ValueError, match="--min-alleles must be >= 1"):
            cli.make_reader(fx_vcz_path, min_alleles=0)

    def test_invalid_type_keyword_rejected(self, fx_vcz_path):
        with pytest.raises(ValueError, match="Invalid type"):
            cli.make_reader(fx_vcz_path, types="garbage")

    def test_view_options_compose_with_sample_scope_filter(self, fx_vcz_path):
        # The variant-scope synthetic mask (here from -m 2) broadcasts
        # against a sample-scope user filter, so a site is included
        # only if it has >=2 alleles AND at least one sample passes
        # FMT/DP>3.
        reader = cli.make_reader(
            fx_vcz_path,
            include="FMT/DP>3",
            min_alleles=2,
        )
        # 20:1230237 is REF-only (1 allele) so -m 2 drops it regardless
        # of whether any sample passes the FMT/DP filter.
        assert 1230237 not in self._positions(reader)


def test_types_and_exclude_types_mutually_exclusive(fx_vcz_path):
    _, vcztools_error = run_vcztools(
        f"view -v snps -V refs {fx_vcz_path}",
        expect_error=True,
    )
    assert "Cannot use --types and --exclude-types together" in vcztools_error


def test_excluding_and_including_samples(fx_vcz_path):
    samples_file_path = pathlib.Path("tests/data/txt/samples.txt")
    error_message = "Cannot specify both a samples string (-s) and a samples file (-S)"

    _, vcztools_error = run_vcztools(
        f"view {fx_vcz_path} -s NA00001 -S ^{samples_file_path.as_posix()}",
        expect_error=True,
    )
    assert error_message in vcztools_error
    _, vcztools_error = run_vcztools(
        f"view {fx_vcz_path} -s ^NA00001 -S {samples_file_path.as_posix()}",
        expect_error=True,
    )
    assert error_message in vcztools_error


@mock.patch("sys.exit")
@mock.patch("os.dup2")
def test_broken_pipe(mocked_dup2, mocked_exit, tmp_path):
    with open(tmp_path / "tmp.txt", "w") as output:
        with cli.handle_broken_pipe(output):
            raise BrokenPipeError()
        mocked_dup2.assert_called_once()
        mocked_exit.assert_called_once_with(1)


class TestQuery:
    def test_format_required(self, fx_vcz_path):
        runner = ct.CliRunner()
        result = runner.invoke(
            cli.vcztools_main,
            f"query {fx_vcz_path} ",
            catch_exceptions=False,
        )
        assert result.exit_code != 0
        assert len(result.stdout) == 0
        assert len(result.stderr) > 0

    def test_path_required(self):
        runner = ct.CliRunner()
        result = runner.invoke(
            cli.vcztools_main,
            "query --format=POS ",
            catch_exceptions=False,
        )
        assert result.exit_code != 0
        assert len(result.stdout) == 0
        assert len(result.stderr) > 0

    def test_list(self, fx_vcz_path):
        result, _ = run_vcztools(f"query -l {fx_vcz_path}")
        assert list(result.splitlines()) == ["NA00001", "NA00002", "NA00003"]

    def test_list_ignores_output(self, fx_vcz_path, tmp_path):
        output = tmp_path / "tmp.txt"
        result, _ = run_vcztools(f"query -l {fx_vcz_path} -o {output.as_posix()}")
        assert list(result.splitlines()) == ["NA00001", "NA00002", "NA00003"]
        assert not output.exists()

    def test_output(self, fx_vcz_path, tmp_path):
        output = tmp_path / "tmp.txt"
        result, _ = run_vcztools(
            f"query -f '%POS\n' {fx_vcz_path} -o {output.as_posix()}"
        )
        assert list(result.splitlines()) == []
        assert output.exists()


class TestViewPlink:
    """CLI surface for ``view-plink``. Exercises only the wiring
    (option parsing, reader assembly, output prefix handling, error
    surfacing). Byte-level parity with ``plink2 --make-bed`` lives in
    ``tests/test_plink_validation.py``.
    """

    @staticmethod
    def _read_bim(path):
        return path.read_text().splitlines()

    @staticmethod
    def _read_fam(path):
        return path.read_text().splitlines()

    def test_minimum_invocation(self, tmp_path, fx_vcz_path):
        # Smallest viable invocation needs --max-alleles 2 because the
        # sample fixture contains multi-allelic variants.
        out = tmp_path / "p"
        result, _ = run_vcztools(
            f"view-plink {fx_vcz_path} --max-alleles 2 "
            f"-e 'CHROM==\"X\"' -o {out.as_posix()}"
        )
        assert (tmp_path / "p.bed").exists()
        assert (tmp_path / "p.bim").exists()
        assert (tmp_path / "p.fam").exists()

    def test_output_required(self, fx_vcz_path):
        # -o is required; missing it is a Click usage error.
        _, err = run_vcztools(
            f"view-plink {fx_vcz_path} --max-alleles 2",
            expect_error=True,
        )
        assert "-o" in err or "--output" in err

    def test_path_required(self):
        runner = ct.CliRunner()
        result = runner.invoke(
            cli.vcztools_main,
            "view-plink -o /tmp/p",
            catch_exceptions=False,
        )
        assert result.exit_code != 0

    def test_multiallelic_without_max_alleles_errors(self, tmp_path, fx_vcz_path):
        # No --max-alleles: vcztools encounters a 3-ALT site and raises.
        out = tmp_path / "p"
        _, err = run_vcztools(
            f"view-plink {fx_vcz_path} -o {out.as_posix()}",
            expect_error=True,
        )
        assert "Multi-allelic" in err

    def test_max_alleles_skips_multiallelic(self, tmp_path, fx_vcz_path):
        out = tmp_path / "p"
        result, _ = run_vcztools(
            f"view-plink {fx_vcz_path} --max-alleles 2 "
            f"-e 'CHROM==\"X\"' -o {out.as_posix()}"
        )
        bim_lines = self._read_bim(tmp_path / "p.bim")
        # sample.vcf.gz has 9 variants. The 1 chrX variant is multi-
        # allelic; -e 'CHROM=="X"' drops it. Of the remaining 8,
        # 20:1110696 and 20:1234567 are multi-allelic — --max-alleles 2
        # drops them, leaving 6 (incl. two monomorphic sites).
        assert len(bim_lines) == 6

    def test_sample_subset(self, tmp_path, fx_vcz_path):
        out = tmp_path / "p"
        run_vcztools(
            f"view-plink {fx_vcz_path} --max-alleles 2 "
            f"-e 'CHROM==\"X\"' -s NA00001,NA00003 -o {out.as_posix()}"
        )
        fam = self._read_fam(tmp_path / "p.fam")
        assert len(fam) == 2
        # FID column is "0", IID is the second column.
        iids = [line.split("\t")[1] for line in fam]
        assert iids == ["NA00001", "NA00003"]

    def test_sample_complement(self, tmp_path, fx_vcz_path):
        out = tmp_path / "p"
        run_vcztools(
            f"view-plink {fx_vcz_path} --max-alleles 2 "
            f"-e 'CHROM==\"X\"' -s ^NA00002 -o {out.as_posix()}"
        )
        fam = self._read_fam(tmp_path / "p.fam")
        iids = [line.split("\t")[1] for line in fam]
        assert iids == ["NA00001", "NA00003"]

    def test_samples_file(self, tmp_path, fx_vcz_path):
        out = tmp_path / "p"
        run_vcztools(
            f"view-plink {fx_vcz_path} --max-alleles 2 "
            f"-e 'CHROM==\"X\"' -S tests/data/txt/samples.txt "
            f"-o {out.as_posix()}"
        )
        fam = self._read_fam(tmp_path / "p.fam")
        iids = [line.split("\t")[1] for line in fam]
        assert iids == ["NA00001", "NA00003"]

    def test_regions(self, tmp_path, fx_vcz_path):
        out = tmp_path / "p"
        run_vcztools(
            f"view-plink {fx_vcz_path} --max-alleles 2 "
            f"-r '20:1230237-' -o {out.as_posix()}"
        )
        bim_lines = self._read_bim(tmp_path / "p.bim")
        # All bim rows must be on contig 20 with pos >= 1230237.
        for line in bim_lines:
            chrom, _id, _cm, pos, _a1, _a2 = line.split("\t")
            assert chrom == "20"
            assert int(pos) >= 1230237

    def test_targets(self, tmp_path, fx_vcz_path):
        # Targets uses exact-position semantics (vs regions overlap).
        out = tmp_path / "p"
        run_vcztools(
            f"view-plink {fx_vcz_path} --max-alleles 2 "
            f"-t '20:1230237-' -o {out.as_posix()}"
        )
        bim_lines = self._read_bim(tmp_path / "p.bim")
        for line in bim_lines:
            chrom, _id, _cm, pos, _a1, _a2 = line.split("\t")
            assert chrom == "20"
            assert int(pos) >= 1230237

    def test_targets_complement(self, tmp_path, fx_vcz_path):
        out = tmp_path / "p"
        run_vcztools(
            f"view-plink {fx_vcz_path} --max-alleles 2 -t '^20' -o {out.as_posix()}"
        )
        bim_lines = self._read_bim(tmp_path / "p.bim")
        for line in bim_lines:
            chrom = line.split("\t")[0]
            assert chrom != "20"

    def test_include_filter(self, tmp_path, fx_vcz_path):
        out = tmp_path / "p"
        run_vcztools(
            f"view-plink {fx_vcz_path} --max-alleles 2 "
            f"-i 'POS>1000000' -o {out.as_posix()}"
        )
        bim_lines = self._read_bim(tmp_path / "p.bim")
        for line in bim_lines:
            pos = int(line.split("\t")[3])
            assert pos > 1000000

    def test_max_alleles_combines_with_include(self, tmp_path, fx_vcz_path):
        # AND-composition path through variant_filter.AndFilter.
        out = tmp_path / "p"
        run_vcztools(
            f"view-plink {fx_vcz_path} --max-alleles 2 "
            f"-i 'POS>1000000' -o {out.as_posix()}"
        )
        bim_lines = self._read_bim(tmp_path / "p.bim")
        # All rows must satisfy both POS>1000000 AND ≤2 alleles.
        for line in bim_lines:
            pos = int(line.split("\t")[3])
            assert pos > 1000000

    def test_sample_scope_filter_rejected(self, tmp_path, fx_vcz_path):
        # plink output is fixed-width per variant, so a sample-scope
        # ``-i 'FMT/...'`` filter has no per-cell channel to land in;
        # the writer's ``materialise_variant_filter`` rejects it. This
        # is independent of ``--max-alleles`` (the rejection lives in
        # the reader, not the AND composition).
        out = tmp_path / "p"
        _, err = run_vcztools(
            f"view-plink {fx_vcz_path} --max-alleles 2 "
            f"-i 'FMT/DP>3' -o {out.as_posix()}",
            expect_error=True,
        )
        assert "Sample-scope variant filters" in err

    def test_min_alleles_drops_monomorphics(self, tmp_path, fx_vcz_path):
        # Without -m, --max-alleles 2 alone keeps the two REF-only
        # sites at 20:1230237 and 20:1235237; -m 2 drops them.
        out = tmp_path / "p"
        run_vcztools(f"view-plink {fx_vcz_path} -m 2 -M 2 -o {out.as_posix()}")
        bim_lines = self._read_bim(tmp_path / "p.bim")
        positions = [int(line.split("\t")[3]) for line in bim_lines]
        assert 1230237 not in positions
        assert 1235237 not in positions

    def test_types_snps_only(self, tmp_path, fx_vcz_path):
        # -v snps + -M 2 keeps only the biallelic SNP rows. The two
        # REF-only sites also drop (no SNP allele).
        out = tmp_path / "p"
        run_vcztools(f"view-plink {fx_vcz_path} -v snps -M 2 -o {out.as_posix()}")
        bim_lines = self._read_bim(tmp_path / "p.bim")
        positions = [int(line.split("\t")[3]) for line in bim_lines]
        # 4 biallelic SNP sites in the fixture: 19:111, 19:112, 20:14370, 20:17330.
        assert sorted(positions) == [111, 112, 14370, 17330]

    def test_exclude_types_snps(self, tmp_path, fx_vcz_path):
        # -V snps drops SNP sites. With -M 2 to skip the multiallelics,
        # only the two monomorphic-REF sites survive.
        out = tmp_path / "p"
        run_vcztools(f"view-plink {fx_vcz_path} -V snps -M 2 -o {out.as_posix()}")
        bim_lines = self._read_bim(tmp_path / "p.bim")
        positions = [int(line.split("\t")[3]) for line in bim_lines]
        assert sorted(positions) == [1230237, 1235237]

    def test_unsupported_type_keyword(self, tmp_path, fx_vcz_path):
        out = tmp_path / "p"
        _, err = run_vcztools(
            f"view-plink {fx_vcz_path} -v indels -M 2 -o {out.as_posix()}",
            expect_error=True,
        )
        assert "TYPE field" in err

    def test_types_and_exclude_types_mutually_exclusive(self, tmp_path, fx_vcz_path):
        out = tmp_path / "p"
        _, err = run_vcztools(
            f"view-plink {fx_vcz_path} -v snps -V refs -o {out.as_posix()}",
            expect_error=True,
        )
        assert "Cannot use --types and --exclude-types together" in err

    def test_stem_taken_verbatim(self, tmp_path, fx_vcz_path):
        # `-o p.bed` is a literal stem now — the writer appends .bed/.bim
        # /.fam to it, so files land at p.bed.bed / p.bed.bim / p.bed.fam.
        out = tmp_path / "p.bed"
        run_vcztools(
            f"view-plink {fx_vcz_path} --max-alleles 2 "
            f"-e 'CHROM==\"X\"' -o {out.as_posix()}"
        )
        assert (tmp_path / "p.bed.bed").exists()
        assert (tmp_path / "p.bed.bim").exists()
        assert (tmp_path / "p.bed.fam").exists()

    def test_no_bim(self, tmp_path, fx_vcz_path):
        out = tmp_path / "p"
        run_vcztools(
            f"view-plink {fx_vcz_path} --max-alleles 2 "
            f"-e 'CHROM==\"X\"' -o {out.as_posix()} --no-bim"
        )
        assert (tmp_path / "p.bed").exists()
        assert not (tmp_path / "p.bim").exists()
        assert (tmp_path / "p.fam").exists()

    def test_no_fam(self, tmp_path, fx_vcz_path):
        out = tmp_path / "p"
        run_vcztools(
            f"view-plink {fx_vcz_path} --max-alleles 2 "
            f"-e 'CHROM==\"X\"' -o {out.as_posix()} --no-fam"
        )
        assert (tmp_path / "p.bed").exists()
        assert (tmp_path / "p.bim").exists()
        assert not (tmp_path / "p.fam").exists()

    def test_no_bim_no_fam(self, tmp_path, fx_vcz_path):
        out = tmp_path / "p"
        run_vcztools(
            f"view-plink {fx_vcz_path} --max-alleles 2 "
            f"-e 'CHROM==\"X\"' -o {out.as_posix()} --no-bim --no-fam"
        )
        assert (tmp_path / "p.bed").exists()
        assert not (tmp_path / "p.bim").exists()
        assert not (tmp_path / "p.fam").exists()


class TestViewBgen:
    """CLI surface for ``view-bgen``. Exercises wiring (option parsing,
    reader assembly, output prefix handling, error surfacing). Byte-level
    encoder checks live in ``tests/test_bgen.py``.
    """

    @staticmethod
    def _read_sample(path):
        return path.read_text().splitlines()

    @staticmethod
    def _bgi_variant_rows(path):
        conn = sqlite3.connect(str(path))
        try:
            return conn.execute(
                "SELECT chromosome, position, rsid, allele1, allele2 "
                "FROM Variant ORDER BY position"
            ).fetchall()
        finally:
            conn.close()

    def test_minimum_invocation(self, tmp_path, fx_vcz_path):
        # The sample fixture has multi-allelic variants — need --max-alleles 2.
        out = tmp_path / "b"
        run_vcztools(
            f"view-bgen {fx_vcz_path} --max-alleles 2 "
            f"-e 'CHROM==\"X\"' -o {out.as_posix()}"
        )
        assert (tmp_path / "b.bgen").exists()
        assert (tmp_path / "b.sample").exists()
        assert (tmp_path / "b.bgen.bgi").exists()

    def test_default_streams_to_stdout(self, tmp_path, fx_vcz_path):
        # Without -o the .bgen payload streams to stdout; no sidecars.
        runner = ct.CliRunner()
        with runner.isolated_filesystem(tmp_path) as cwd:
            result = runner.invoke(
                cli.vcztools_main,
                f"view-bgen {fx_vcz_path} --max-alleles 2 -e 'CHROM==\"X\"'",
                catch_exceptions=False,
            )
            assert result.exit_code == 0
            stdout_bytes = result.stdout_bytes
            # Bytes 8-19 are: num_variants (uint32) + num_samples (uint32)
            # + BGEN_MAGIC. The magic confirms we wrote a BGEN payload.
            assert stdout_bytes[16:20] == b"bgen"
            cwd_path = pathlib.Path(cwd)
            assert not any(cwd_path.glob("*.bgen"))
            assert not any(cwd_path.glob("*.bgi"))
            assert not any(cwd_path.glob("*.sample"))

    def test_path_required(self):
        runner = ct.CliRunner()
        result = runner.invoke(
            cli.vcztools_main,
            "view-bgen",
            catch_exceptions=False,
        )
        assert result.exit_code != 0

    def test_multiallelic_without_max_alleles_errors(self, tmp_path, fx_vcz_path):
        out = tmp_path / "b"
        _, err = run_vcztools(
            f"view-bgen {fx_vcz_path} -o {out.as_posix()}",
            expect_error=True,
        )
        assert "Multi-allelic" in err

    def test_max_alleles_skips_multiallelic(self, tmp_path, fx_vcz_path):
        # Same fixture, same expectations as view-plink: 6 surviving
        # variants after -M 2 + -e 'CHROM=="X"'.
        out = tmp_path / "b"
        run_vcztools(
            f"view-bgen {fx_vcz_path} --max-alleles 2 "
            f"-e 'CHROM==\"X\"' -o {out.as_posix()}"
        )
        rows = self._bgi_variant_rows(tmp_path / "b.bgen.bgi")
        assert len(rows) == 6

    def test_sample_subset(self, tmp_path, fx_vcz_path):
        out = tmp_path / "b"
        run_vcztools(
            f"view-bgen {fx_vcz_path} --max-alleles 2 "
            f"-e 'CHROM==\"X\"' -s NA00001,NA00003 -o {out.as_posix()}"
        )
        sample_lines = self._read_sample(tmp_path / "b.sample")
        # Drop the two header rows.
        ids = [line.split()[0] for line in sample_lines[2:]]
        assert ids == ["NA00001", "NA00003"]

    def test_sample_complement(self, tmp_path, fx_vcz_path):
        out = tmp_path / "b"
        run_vcztools(
            f"view-bgen {fx_vcz_path} --max-alleles 2 "
            f"-e 'CHROM==\"X\"' -s ^NA00002 -o {out.as_posix()}"
        )
        sample_lines = self._read_sample(tmp_path / "b.sample")
        ids = [line.split()[0] for line in sample_lines[2:]]
        assert ids == ["NA00001", "NA00003"]

    def test_regions(self, tmp_path, fx_vcz_path):
        out = tmp_path / "b"
        run_vcztools(
            f"view-bgen {fx_vcz_path} --max-alleles 2 "
            f"-r '20:1230237-' -o {out.as_posix()}"
        )
        rows = self._bgi_variant_rows(tmp_path / "b.bgen.bgi")
        for chrom, pos, *_ in rows:
            assert chrom == "20"
            assert pos >= 1230237

    def test_targets_complement(self, tmp_path, fx_vcz_path):
        out = tmp_path / "b"
        run_vcztools(
            f"view-bgen {fx_vcz_path} --max-alleles 2 -t '^20' -o {out.as_posix()}"
        )
        rows = self._bgi_variant_rows(tmp_path / "b.bgen.bgi")
        for chrom, *_ in rows:
            assert chrom != "20"

    def test_include_filter(self, tmp_path, fx_vcz_path):
        out = tmp_path / "b"
        run_vcztools(
            f"view-bgen {fx_vcz_path} --max-alleles 2 "
            f"-i 'POS>1000000' -o {out.as_posix()}"
        )
        rows = self._bgi_variant_rows(tmp_path / "b.bgen.bgi")
        for _chrom, pos, *_ in rows:
            assert pos > 1000000

    def test_types_snps_only(self, tmp_path, fx_vcz_path):
        out = tmp_path / "b"
        run_vcztools(f"view-bgen {fx_vcz_path} -v snps -M 2 -o {out.as_posix()}")
        rows = self._bgi_variant_rows(tmp_path / "b.bgen.bgi")
        positions = sorted(pos for _chrom, pos, *_ in rows)
        # Same 4 biallelic SNP sites the view-plink tests rely on.
        assert positions == [111, 112, 14370, 17330]

    def test_stem_taken_verbatim(self, tmp_path, fx_vcz_path):
        # `-o p.bgen` is a literal stem — appended files land at
        # p.bgen.bgen / p.bgen.sample / p.bgen.bgen.bgi.
        out = tmp_path / "p.bgen"
        run_vcztools(
            f"view-bgen {fx_vcz_path} --max-alleles 2 "
            f"-e 'CHROM==\"X\"' -o {out.as_posix()}"
        )
        assert (tmp_path / "p.bgen.bgen").exists()
        assert (tmp_path / "p.bgen.sample").exists()
        assert (tmp_path / "p.bgen.bgen.bgi").exists()

    def test_no_bgi(self, tmp_path, fx_vcz_path):
        out = tmp_path / "b"
        run_vcztools(
            f"view-bgen {fx_vcz_path} --max-alleles 2 "
            f"-e 'CHROM==\"X\"' -o {out.as_posix()} --no-bgi"
        )
        assert (tmp_path / "b.bgen").exists()
        assert (tmp_path / "b.sample").exists()
        assert not (tmp_path / "b.bgen.bgi").exists()

    def test_no_sample_file(self, tmp_path, fx_vcz_path):
        out = tmp_path / "b"
        run_vcztools(
            f"view-bgen {fx_vcz_path} --max-alleles 2 "
            f"-e 'CHROM==\"X\"' -o {out.as_posix()} --no-sample-file"
        )
        assert (tmp_path / "b.bgen").exists()
        assert (tmp_path / "b.bgen.bgi").exists()
        assert not (tmp_path / "b.sample").exists()

    def test_no_bgi_no_sample_file(self, tmp_path, fx_vcz_path):
        out = tmp_path / "b"
        run_vcztools(
            f"view-bgen {fx_vcz_path} --max-alleles 2 "
            f"-e 'CHROM==\"X\"' -o {out.as_posix()} --no-bgi --no-sample-file"
        )
        assert (tmp_path / "b.bgen").exists()
        assert not (tmp_path / "b.bgen.bgi").exists()
        assert not (tmp_path / "b.sample").exists()

    def test_no_header_samples_clears_flag(self, tmp_path, fx_vcz_path):
        # SAMPLE_IDS_PRESENT bit (1 << 31) must be 0 in the BGEN flag
        # word; the sample-id block is omitted (offset = HEADER_LENGTH).
        out = tmp_path / "b"
        run_vcztools(
            f"view-bgen {fx_vcz_path} --max-alleles 2 "
            f"-e 'CHROM==\"X\"' -o {out.as_posix()} --no-header-samples"
        )
        data = (tmp_path / "b.bgen").read_bytes()
        # Offset is little-endian uint32 at bytes [0:4]; equals
        # HEADER_LENGTH (20) when no sample-id block is embedded.
        offset = int.from_bytes(data[0:4], "little")
        assert offset == 20
        # Flag word: bytes [16:20] are BGEN_MAGIC, [20:24] is flags.
        flags = int.from_bytes(data[20:24], "little")
        assert (flags & (1 << 31)) == 0

    def test_unphased_forces_unphased_output(self, tmp_path, fx_vcz_path):
        # Wiring check: --unphased reaches write_bgen and clears every
        # variant's phased flag in the output payload. Byte-level
        # correctness lives in tests/test_bgen.py.
        out = tmp_path / "b"
        run_vcztools(
            f"view-bgen {fx_vcz_path} --max-alleles 2 "
            f"-e 'CHROM==\"X\"' -o {out.as_posix()} --unphased"
        )
        with br.open_bgen(tmp_path / "b.bgen", verbose=False) as bg:
            assert not bg.phased.any()

    def test_no_header_samples_no_sample_file_warns(self, tmp_path, fx_vcz_path):
        # Both off → sample IDs nowhere; expect a logger.warning.
        # vcztools' setup_logging force-replaces handlers, so caplog won't
        # see the warning; route it to a --log-file instead.
        out = tmp_path / "b"
        log = tmp_path / "warn.log"
        run_vcztools(
            f"view-bgen {fx_vcz_path} --max-alleles 2 "
            f"-e 'CHROM==\"X\"' -o {out.as_posix()} "
            f"--no-header-samples --no-sample-file "
            f"--log-file {log.as_posix()}"
        )
        log_text = log.read_text()
        assert "WARNING" in log_text
        assert "sample IDs nowhere" in log_text


class TestIndex:
    def test_stats(self, fx_vcz_path):
        result, _ = run_vcztools(f"index -s {fx_vcz_path}")
        assert list(result.splitlines()) == ["19\t.\t2", "20\t.\t6", "X\t.\t1"]

    def test_nrecords(self, fx_vcz_path):
        result, _ = run_vcztools(f"index -n {fx_vcz_path}")
        assert list(result.splitlines()) == ["9"]

    def test_stats_and_nrecords(self, fx_vcz_path):
        runner = ct.CliRunner()
        result = runner.invoke(
            cli.vcztools_main,
            f"index -ns {fx_vcz_path}",
            catch_exceptions=False,
        )
        assert result.exit_code != 0
        assert len(result.stdout) == 0
        assert len(result.stderr) > 0
        assert "Expected only one of --stats or --nrecords options" in result.stderr

    def test_no_stats_or_nrecords(self, fx_vcz_path):
        runner = ct.CliRunner()
        result = runner.invoke(
            cli.vcztools_main,
            f"index {fx_vcz_path}",
            catch_exceptions=False,
        )
        assert result.exit_code != 0
        assert len(result.stdout) == 0
        assert len(result.stderr) > 0
        assert "Error: Building region indexes is not supported" in result.stderr


def test_top_level():
    runner = ct.CliRunner()
    result = runner.invoke(
        cli.vcztools_main,
        catch_exceptions=False,
    )
    assert result.exit_code != 0
    assert len(result.stdout) == 0
    assert len(result.stderr) > 0


def test_version():
    runner = ct.CliRunner()
    result = runner.invoke(cli.vcztools_main, ["--version"], catch_exceptions=False)
    s = f"version {provenance.__version__}\n"
    assert result.stdout.endswith(s)


@pytest.mark.parametrize("backend", ["zip", "directory"])
class TestBackends:
    """Smoke tests verifying each CLI command works with both storage backends."""

    def test_view(self, fx_sample_vcz, backend):
        vcz = fx_sample_vcz.path(backend).as_posix()
        output, _ = run_vcztools(f"view --no-version {vcz}")
        assert output.startswith("##fileformat=VCF")
        assert "NA00001" in output

    def test_query(self, fx_sample_vcz, backend):
        vcz = fx_sample_vcz.path(backend).as_posix()
        output, _ = run_vcztools(f"query -f '%POS\\n' {vcz}")
        positions = output.strip().splitlines()
        assert len(positions) == 9
        assert positions[0] == "111"

    def test_index_nrecords(self, fx_sample_vcz, backend):
        vcz = fx_sample_vcz.path(backend).as_posix()
        output, _ = run_vcztools(f"index -n {vcz}")
        assert output.strip() == "9"

    def test_index_stats(self, fx_sample_vcz, backend):
        vcz = fx_sample_vcz.path(backend).as_posix()
        output, _ = run_vcztools(f"index -s {vcz}")
        assert output.strip().splitlines() == ["19\t.\t2", "20\t.\t6", "X\t.\t1"]


class TestParseStorageOptions:
    """``cli._parse_storage_options`` decodes ``KEY=VALUE`` pairs into a
    dict, parsing each ``VALUE`` as JSON when possible and falling back
    to the raw string."""

    def test_empty_returns_none(self):
        assert cli._parse_storage_options(()) is None

    def test_json_int(self):
        assert cli._parse_storage_options(("timeout=30",)) == {"timeout": 30}

    def test_json_string_fallback(self):
        # "us-east-1" is not valid JSON; falls through to raw string.
        assert cli._parse_storage_options(("region=us-east-1",)) == {
            "region": "us-east-1"
        }

    def test_json_object(self):
        result = cli._parse_storage_options(('client_kwargs={"verify": false}',))
        assert result == {"client_kwargs": {"verify": False}}

    def test_multiple(self):
        result = cli._parse_storage_options(
            ("timeout=30", "region=us-east-1", 'tags=["a", "b"]')
        )
        assert result == {
            "timeout": 30,
            "region": "us-east-1",
            "tags": ["a", "b"],
        }

    def test_value_with_equals_kept_intact(self):
        # Only the first '=' splits; the rest is part of VALUE.
        assert cli._parse_storage_options(("expr=a=b",)) == {"expr": "a=b"}

    def test_invalid_format_raises(self):
        with pytest.raises(click.BadParameter, match="must be KEY=VALUE"):
            cli._parse_storage_options(("no-equals",))


class TestStorageOptionCli:
    """End-to-end: ``--storage-option`` flows through the CLI into
    ``open_zarr`` for every command that accepts a backend."""

    def test_view_passes_storage_options(self, fx_vcz_path, monkeypatch):
        # Spy on open_zarr to capture the kwargs the CLI passes; abort
        # immediately so the test doesn't have to navigate the rest of
        # the view pipeline.
        captured = {}

        def spy(path, **kwargs):
            captured.update(kwargs)
            # ValueError is converted to a ClickException by the
            # handle_exception decorator; the runner sees a clean exit.
            raise ValueError("captured")

        monkeypatch.setattr(cli, "open_zarr", spy)
        _, err = run_vcztools(
            f"view --no-version {fx_vcz_path} "
            "--storage-option foo=42 --storage-option bar=baz",
            expect_error=True,
        )
        assert captured["storage_options"] == {"foo": 42, "bar": "baz"}

    def test_invalid_pair_raises_bad_parameter(self, fx_vcz_path):
        _, err = run_vcztools(
            f"view --no-version {fx_vcz_path} --storage-option no-equals",
            expect_error=True,
        )
        assert "must be KEY=VALUE" in err


class TestDeprecatedZarrBackendStorageAlias:
    """``--zarr-backend-storage`` is accepted as a hidden, deprecated alias
    for ``--backend-storage``: it forwards the value to ``open_zarr`` and
    emits a ``DeprecationWarning``."""

    def test_alias_forwards_and_warns(self, fx_vcz_path, monkeypatch):
        captured = {}

        def spy(path, **kwargs):
            captured.update(kwargs)
            raise ValueError("captured")

        monkeypatch.setattr(cli, "open_zarr", spy)
        with pytest.warns(DeprecationWarning, match="--zarr-backend-storage"):
            run_vcztools(
                f"view --no-version {fx_vcz_path} --zarr-backend-storage fsspec",
                expect_error=True,
            )
        assert captured["backend_storage"] == "fsspec"

    def test_new_flag_does_not_warn(self, fx_vcz_path, monkeypatch, recwarn):
        captured = {}

        def spy(path, **kwargs):
            captured.update(kwargs)
            raise ValueError("captured")

        monkeypatch.setattr(cli, "open_zarr", spy)
        run_vcztools(
            f"view --no-version {fx_vcz_path} --backend-storage fsspec",
            expect_error=True,
        )
        assert captured["backend_storage"] == "fsspec"
        deprecation_warnings = [
            w for w in recwarn.list if issubclass(w.category, DeprecationWarning)
        ]
        assert deprecation_warnings == []


class TestSelectionOptions:
    """Unit tests for the bcftools-shaped SelectionOptions bundle (category 4
    reusable surface; consumed by view / view-plink / view-bgen / biofuse)."""

    def test_from_click_kwargs_populates_fields(self):
        kwargs = {
            "regions": "1:100-200",
            "samples": "s1,s2",
            "force_samples": True,
            "min_alleles": 2,
            "max_alleles": 4,
            "out": "plink",
            "log_level": "INFO",
        }
        snapshot = copy.deepcopy(kwargs)
        selection = cli.SelectionOptions.from_click_kwargs(kwargs)
        assert selection.regions == "1:100-200"
        assert selection.samples == "s1,s2"
        assert selection.force_samples is True
        assert selection.min_alleles == 2
        assert selection.max_alleles == 4
        assert kwargs == snapshot

    def test_from_click_kwargs_defaults_when_unset(self):
        kwargs = {}
        selection = cli.SelectionOptions.from_click_kwargs(kwargs)
        assert selection == cli.SelectionOptions()

    def test_from_click_kwargs_ignores_unrelated_fields(self):
        kwargs = {
            "regions": "1:100-200",
            "backend_storage": "fsspec",
            "storage_options": (),
            "readahead_workers": 4,
            "log_level": "INFO",
        }
        snapshot = copy.deepcopy(kwargs)
        selection = cli.SelectionOptions.from_click_kwargs(kwargs)
        assert selection.regions == "1:100-200"
        assert kwargs == snapshot


class TestZarrStoreOptions:
    """Unit tests for the ZarrStoreOptions bundle (category 1)."""

    def test_from_click_kwargs_populates_fields(self):
        kwargs = {
            "backend_storage": "fsspec",
            "storage_options": ("k1=42", "k2=foo"),
            "regions": "1:100-200",
        }
        snapshot = copy.deepcopy(kwargs)
        zarr_store = cli.ZarrStoreOptions.from_click_kwargs(kwargs)
        assert zarr_store.backend_storage == "fsspec"
        assert zarr_store.storage_options == {"k1": 42, "k2": "foo"}
        assert kwargs == snapshot

    def test_from_click_kwargs_defaults_when_unset(self):
        kwargs = {}
        zarr_store = cli.ZarrStoreOptions.from_click_kwargs(kwargs)
        assert zarr_store == cli.ZarrStoreOptions()
        assert zarr_store.storage_options is None

    def test_from_click_kwargs_empty_storage_options_tuple_parses_to_none(self):
        kwargs = {"storage_options": ()}
        zarr_store = cli.ZarrStoreOptions.from_click_kwargs(kwargs)
        assert zarr_store.storage_options is None


class TestReaderOptions:
    """Unit tests for the ReaderOptions bundle (category 2)."""

    def test_from_click_kwargs_populates_fields(self):
        kwargs = {
            "readahead_workers": 8,
            "readahead_bytes": 1024 * 1024,
            "regions": "1:100-200",
        }
        snapshot = copy.deepcopy(kwargs)
        reader_opts = cli.ReaderOptions.from_click_kwargs(kwargs)
        assert reader_opts.readahead_workers == 8
        assert reader_opts.readahead_bytes == 1024 * 1024
        assert kwargs == snapshot

    def test_from_click_kwargs_defaults_when_unset(self):
        kwargs = {}
        reader_opts = cli.ReaderOptions.from_click_kwargs(kwargs)
        assert reader_opts == cli.ReaderOptions()


class TestLogOptions:
    """Unit tests for the LogOptions bundle (category 3)."""

    def test_from_click_kwargs_populates_fields(self):
        kwargs = {"log_level": "DEBUG", "log_file": "/tmp/log", "out": "plink"}
        snapshot = copy.deepcopy(kwargs)
        config = cli.LogOptions.from_click_kwargs(kwargs)
        assert config.log_level == "DEBUG"
        assert config.log_file == "/tmp/log"
        assert kwargs == snapshot

    def test_from_click_kwargs_defaults(self):
        kwargs = {}
        config = cli.LogOptions.from_click_kwargs(kwargs)
        assert config == cli.LogOptions()


class TestViewPlinkOptions:
    """Unit tests for the ViewPlinkOptions command-level bundle."""

    def test_from_click_kwargs_populates_every_section(self):
        kwargs = {
            "regions": "1:100-200",
            "force_samples": True,
            "backend_storage": "fsspec",
            "storage_options": ("k=1",),
            "readahead_workers": 4,
            "readahead_bytes": 2048,
            "log_level": "DEBUG",
            "log_file": "/tmp/log",
            "no_bim": True,
            "no_fam": True,
        }
        opts = cli.ViewPlinkOptions.from_click_kwargs(kwargs)
        assert opts.selection.regions == "1:100-200"
        assert opts.selection.force_samples is True
        assert opts.zarr_store.backend_storage == "fsspec"
        assert opts.zarr_store.storage_options == {"k": 1}
        assert opts.reader.readahead_workers == 4
        assert opts.reader.readahead_bytes == 2048
        assert opts.log.log_level == "DEBUG"
        assert opts.log.log_file == "/tmp/log"
        assert opts.no_bim is True
        assert opts.no_fam is True

    def test_from_click_kwargs_defaults_when_unset(self):
        kwargs = {}
        opts = cli.ViewPlinkOptions.from_click_kwargs(kwargs)
        assert opts == cli.ViewPlinkOptions()

    def test_from_click_kwargs_does_not_mutate_input(self):
        kwargs = {
            "regions": "1:100-200",
            "force_samples": True,
            "backend_storage": "fsspec",
            "storage_options": ("k=1",),
            "readahead_workers": 4,
            "readahead_bytes": 2048,
            "log_level": "DEBUG",
            "log_file": "/tmp/log",
            "no_bim": True,
            "no_fam": True,
        }
        snapshot = copy.deepcopy(kwargs)
        cli.ViewPlinkOptions.from_click_kwargs(kwargs)
        assert kwargs == snapshot


class TestViewBgenOptions:
    """Unit tests for the ViewBgenOptions command-level bundle."""

    def test_from_click_kwargs_populates_every_section(self):
        kwargs = {
            "regions": "1:100-200",
            "max_alleles": 2,
            "backend_storage": "fsspec",
            "storage_options": ("k=1",),
            "readahead_workers": 4,
            "readahead_bytes": 2048,
            "log_level": "DEBUG",
            "log_file": "/tmp/log",
            "no_bgi": True,
            "no_sample_file": True,
            "no_header_samples": True,
        }
        opts = cli.ViewBgenOptions.from_click_kwargs(kwargs)
        assert opts.selection.regions == "1:100-200"
        assert opts.selection.max_alleles == 2
        assert opts.zarr_store.backend_storage == "fsspec"
        assert opts.zarr_store.storage_options == {"k": 1}
        assert opts.reader.readahead_workers == 4
        assert opts.reader.readahead_bytes == 2048
        assert opts.log.log_level == "DEBUG"
        assert opts.log.log_file == "/tmp/log"
        assert opts.no_bgi is True
        assert opts.no_sample_file is True
        assert opts.no_header_samples is True

    def test_from_click_kwargs_defaults_when_unset(self):
        kwargs = {}
        opts = cli.ViewBgenOptions.from_click_kwargs(kwargs)
        assert opts == cli.ViewBgenOptions()

    def test_from_click_kwargs_does_not_mutate_input(self):
        kwargs = {
            "regions": "1:100-200",
            "max_alleles": 2,
            "backend_storage": "fsspec",
            "storage_options": ("k=1",),
            "readahead_workers": 4,
            "readahead_bytes": 2048,
            "log_level": "DEBUG",
            "log_file": "/tmp/log",
            "no_bgi": True,
            "no_sample_file": True,
            "no_header_samples": True,
        }
        snapshot = copy.deepcopy(kwargs)
        cli.ViewBgenOptions.from_click_kwargs(kwargs)
        assert kwargs == snapshot


class TestMakeReaderFromGroups:
    """``make_reader_from_groups`` is the seam between the option bundles and
    :func:`make_reader`. Verify every field flows through."""

    def test_forwards_every_field(self, monkeypatch):
        captured = []

        def spy(path, **kwargs):
            captured.append((path, kwargs))
            return object()

        monkeypatch.setattr(cli, "make_reader", spy)
        selection = cli.SelectionOptions(
            regions="1:100-200",
            samples="s1,s2",
            force_samples=True,
            min_alleles=2,
        )
        zarr_store = cli.ZarrStoreOptions(
            backend_storage="fsspec",
            storage_options={"k": 1},
        )
        reader_opts = cli.ReaderOptions(
            readahead_workers=4,
            readahead_bytes=2048,
        )
        cli.make_reader_from_groups(
            "/tmp/x.vcz",
            selection=selection,
            zarr_store=zarr_store,
            reader=reader_opts,
        )
        assert len(captured) == 1
        path, kwargs = captured[0]
        assert path == "/tmp/x.vcz"
        for field, expected in [
            ("regions", "1:100-200"),
            ("samples", "s1,s2"),
            ("force_samples", True),
            ("min_alleles", 2),
            ("backend_storage", "fsspec"),
            ("storage_options", {"k": 1}),
            ("readahead_workers", 4),
            ("readahead_bytes", 2048),
        ]:
            assert kwargs[field] == expected

    def test_defaults_when_no_bundles_passed(self, monkeypatch):
        captured = []

        def spy(path, **kwargs):
            captured.append((path, kwargs))
            return object()

        monkeypatch.setattr(cli, "make_reader", spy)
        cli.make_reader_from_groups("/tmp/x.vcz")
        path, kwargs = captured[0]
        assert path == "/tmp/x.vcz"
        # Every field from every bundle resolves to its dataclass default.
        assert kwargs["regions"] is None
        assert kwargs["backend_storage"] is None
        assert kwargs["storage_options"] is None
        assert kwargs["readahead_workers"] is None
        assert kwargs["force_samples"] is False


class TestHelpGroups:
    """Smoke tests that the four category headers render in ``--help`` for
    each command that exposes options in that category."""

    @pytest.mark.parametrize(
        ("subcommand", "expected_groups"),
        [
            ("index", [cli.OptionGroup.ZARR_STORE, cli.OptionGroup.LOGGING]),
            (
                "query",
                [
                    cli.OptionGroup.SELECTION,
                    cli.OptionGroup.ZARR_STORE,
                    cli.OptionGroup.READER,
                    cli.OptionGroup.LOGGING,
                ],
            ),
            (
                "view",
                [
                    cli.OptionGroup.SELECTION,
                    cli.OptionGroup.ZARR_STORE,
                    cli.OptionGroup.READER,
                    cli.OptionGroup.LOGGING,
                ],
            ),
            (
                "view-plink",
                [
                    cli.OptionGroup.SELECTION,
                    cli.OptionGroup.ZARR_STORE,
                    cli.OptionGroup.READER,
                    cli.OptionGroup.LOGGING,
                ],
            ),
            (
                "view-bgen",
                [
                    cli.OptionGroup.SELECTION,
                    cli.OptionGroup.ZARR_STORE,
                    cli.OptionGroup.READER,
                    cli.OptionGroup.LOGGING,
                ],
            ),
        ],
    )
    def test_help_includes_group_headers(self, subcommand, expected_groups):
        runner = ct.CliRunner()
        result = runner.invoke(cli.vcztools_main, [subcommand, "--help"])
        assert result.exit_code == 0
        # The default ``Options:`` section always appears.
        assert "Options:" in result.output
        for group in expected_groups:
            assert group in result.output, (
                f"missing group {group!r} in {subcommand} --help:\n{result.output}"
            )


class TestReadaheadOptions:
    """``--readahead-workers`` / ``--readahead-buffer-size`` wiring for
    every ``make_reader`` consumer: ``query``, ``view``, ``view-plink``.
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("0", 0),
            ("1024", 1024),
            ("100M", 100 * 1024 * 1024),
            ("256MiB", 256 * 1024 * 1024),
            ("2G", 2 * 1024**3),
            ("1T", 1024**4),
        ],
    )
    def test_size_param_parses_suffixes(self, text, expected):
        assert cli.SIZE.convert(text, None, None) == expected

    def test_size_param_passes_int_through(self):
        assert cli.SIZE.convert(4096, None, None) == 4096

    def test_size_param_passes_none_through(self):
        assert cli.SIZE.convert(None, None, None) is None

    def test_size_param_rejects_invalid(self):
        with pytest.raises(click.UsageError):
            cli.SIZE.convert("not-a-size", None, None)

    def test_make_reader_forwards_workers(self, fx_vcz_path):
        with cli.make_reader(fx_vcz_path, readahead_workers=4) as reader:
            assert reader._readahead_workers == 4

    def test_make_reader_forwards_bytes(self, fx_vcz_path):
        with cli.make_reader(fx_vcz_path, readahead_bytes=1024) as reader:
            assert reader.readahead_bytes == 1024

    def test_make_reader_default_passes_none(self, fx_vcz_path):
        # ``None`` is what flows from the CLI when the flags are
        # omitted. ``VczReader`` then applies its own defaults.
        with cli.make_reader(fx_vcz_path) as reader:
            assert reader.readahead_bytes is None

    @staticmethod
    def _spy_vcz_reader_init(monkeypatch):
        captured = {}
        real_init = cli.retrieval.VczReader.__init__

        def spy(self, root, **kw):
            captured.update(kw)
            real_init(self, root, **kw)

        monkeypatch.setattr(cli.retrieval.VczReader, "__init__", spy)
        return captured

    def test_view_forwards_flags(self, monkeypatch, tmp_path, fx_vcz_path):
        captured = self._spy_vcz_reader_init(monkeypatch)
        output_path = tmp_path / "tmp.vcf"
        runner = ct.CliRunner()
        result = runner.invoke(
            cli.vcztools_main,
            f"view --no-version --readahead-workers 4 "
            f"--readahead-buffer-size 100M {fx_vcz_path} "
            f"-o {output_path.as_posix()}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert captured == {
            "readahead_workers": 4,
            "readahead_bytes": 100 * 1024 * 1024,
        }

    def test_query_forwards_flags(self, monkeypatch, tmp_path, fx_vcz_path):
        captured = self._spy_vcz_reader_init(monkeypatch)
        output_path = tmp_path / "tmp.txt"
        runner = ct.CliRunner()
        result = runner.invoke(
            cli.vcztools_main,
            f"query -f '%POS\n' --readahead-workers 2 "
            f"--readahead-buffer-size 1024 {fx_vcz_path} "
            f"-o {output_path.as_posix()}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert captured == {"readahead_workers": 2, "readahead_bytes": 1024}

    def test_view_plink_forwards_flags(self, monkeypatch, tmp_path, fx_vcz_path):
        captured = self._spy_vcz_reader_init(monkeypatch)
        out = tmp_path / "p"
        runner = ct.CliRunner()
        result = runner.invoke(
            cli.vcztools_main,
            f"view-plink --max-alleles 2 -e 'CHROM==\"X\"' "
            f"--readahead-workers 8 --readahead-buffer-size 2M "
            f"{fx_vcz_path} -o {out.as_posix()}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert captured == {
            "readahead_workers": 8,
            "readahead_bytes": 2 * 1024 * 1024,
        }

    def test_view_invalid_buffer_size(self, tmp_path, fx_vcz_path):
        output_path = tmp_path / "tmp.vcf"
        runner = ct.CliRunner()
        result = runner.invoke(
            cli.vcztools_main,
            f"view --readahead-buffer-size garbage {fx_vcz_path} "
            f"-o {output_path.as_posix()}",
            catch_exceptions=False,
        )
        assert result.exit_code != 0
        assert "garbage" in result.stderr
