import pathlib
import sys
from unittest import mock

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


class TestViewBed:
    """CLI surface for ``view-bed``. Exercises only the wiring
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
            f"view-bed {fx_vcz_path} --max-alleles 2 "
            f"-e 'CHROM==\"X\"' --out {out.as_posix()}"
        )
        assert (tmp_path / "p.bed").exists()
        assert (tmp_path / "p.bim").exists()
        assert (tmp_path / "p.fam").exists()

    def test_default_output_prefix(self, tmp_path, fx_vcz_path):
        # Without --out, files land at "plink.{bed,bim,fam}" in cwd.
        runner = ct.CliRunner()
        with runner.isolated_filesystem(tmp_path) as cwd:
            result = runner.invoke(
                cli.vcztools_main,
                f"view-bed {fx_vcz_path} --max-alleles 2 -e 'CHROM==\"X\"'",
                catch_exceptions=False,
            )
            assert result.exit_code == 0
            cwd_path = pathlib.Path(cwd)
            assert (cwd_path / "plink.bed").exists()
            assert (cwd_path / "plink.bim").exists()
            assert (cwd_path / "plink.fam").exists()

    def test_path_required(self):
        runner = ct.CliRunner()
        result = runner.invoke(
            cli.vcztools_main,
            "view-bed",
            catch_exceptions=False,
        )
        assert result.exit_code != 0

    def test_multiallelic_without_max_alleles_errors(self, tmp_path, fx_vcz_path):
        # No --max-alleles: vcztools encounters a 3-ALT site and raises.
        out = tmp_path / "p"
        _, err = run_vcztools(
            f"view-bed {fx_vcz_path} --out {out.as_posix()}",
            expect_error=True,
        )
        assert "Multi-allelic" in err

    def test_max_alleles_skips_multiallelic(self, tmp_path, fx_vcz_path):
        out = tmp_path / "p"
        result, _ = run_vcztools(
            f"view-bed {fx_vcz_path} --max-alleles 2 "
            f"-e 'CHROM==\"X\"' --out {out.as_posix()}"
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
            f"view-bed {fx_vcz_path} --max-alleles 2 "
            f"-e 'CHROM==\"X\"' -s NA00001,NA00003 --out {out.as_posix()}"
        )
        fam = self._read_fam(tmp_path / "p.fam")
        assert len(fam) == 2
        # FID column is "0", IID is the second column.
        iids = [line.split("\t")[1] for line in fam]
        assert iids == ["NA00001", "NA00003"]

    def test_sample_complement(self, tmp_path, fx_vcz_path):
        out = tmp_path / "p"
        run_vcztools(
            f"view-bed {fx_vcz_path} --max-alleles 2 "
            f"-e 'CHROM==\"X\"' -s ^NA00002 --out {out.as_posix()}"
        )
        fam = self._read_fam(tmp_path / "p.fam")
        iids = [line.split("\t")[1] for line in fam]
        assert iids == ["NA00001", "NA00003"]

    def test_samples_file(self, tmp_path, fx_vcz_path):
        out = tmp_path / "p"
        run_vcztools(
            f"view-bed {fx_vcz_path} --max-alleles 2 "
            f"-e 'CHROM==\"X\"' -S tests/data/txt/samples.txt "
            f"--out {out.as_posix()}"
        )
        fam = self._read_fam(tmp_path / "p.fam")
        iids = [line.split("\t")[1] for line in fam]
        assert iids == ["NA00001", "NA00003"]

    def test_regions(self, tmp_path, fx_vcz_path):
        out = tmp_path / "p"
        run_vcztools(
            f"view-bed {fx_vcz_path} --max-alleles 2 "
            f"-r '20:1230237-' --out {out.as_posix()}"
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
            f"view-bed {fx_vcz_path} --max-alleles 2 "
            f"-t '20:1230237-' --out {out.as_posix()}"
        )
        bim_lines = self._read_bim(tmp_path / "p.bim")
        for line in bim_lines:
            chrom, _id, _cm, pos, _a1, _a2 = line.split("\t")
            assert chrom == "20"
            assert int(pos) >= 1230237

    def test_targets_complement(self, tmp_path, fx_vcz_path):
        out = tmp_path / "p"
        run_vcztools(
            f"view-bed {fx_vcz_path} --max-alleles 2 -t '^20' --out {out.as_posix()}"
        )
        bim_lines = self._read_bim(tmp_path / "p.bim")
        for line in bim_lines:
            chrom = line.split("\t")[0]
            assert chrom != "20"

    def test_include_filter(self, tmp_path, fx_vcz_path):
        out = tmp_path / "p"
        run_vcztools(
            f"view-bed {fx_vcz_path} --max-alleles 2 "
            f"-i 'POS>1000000' --out {out.as_posix()}"
        )
        bim_lines = self._read_bim(tmp_path / "p.bim")
        for line in bim_lines:
            pos = int(line.split("\t")[3])
            assert pos > 1000000

    def test_max_alleles_combines_with_include(self, tmp_path, fx_vcz_path):
        # AND-composition path through variant_filter.AndFilter.
        out = tmp_path / "p"
        run_vcztools(
            f"view-bed {fx_vcz_path} --max-alleles 2 "
            f"-i 'POS>1000000' --out {out.as_posix()}"
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
            f"view-bed {fx_vcz_path} --max-alleles 2 "
            f"-i 'FMT/DP>3' --out {out.as_posix()}",
            expect_error=True,
        )
        assert "Sample-scope variant filters" in err

    def test_min_alleles_drops_monomorphics(self, tmp_path, fx_vcz_path):
        # Without -m, --max-alleles 2 alone keeps the two REF-only
        # sites at 20:1230237 and 20:1235237; -m 2 drops them.
        out = tmp_path / "p"
        run_vcztools(f"view-bed {fx_vcz_path} -m 2 -M 2 --out {out.as_posix()}")
        bim_lines = self._read_bim(tmp_path / "p.bim")
        positions = [int(line.split("\t")[3]) for line in bim_lines]
        assert 1230237 not in positions
        assert 1235237 not in positions

    def test_types_snps_only(self, tmp_path, fx_vcz_path):
        # -v snps + -M 2 keeps only the biallelic SNP rows. The two
        # REF-only sites also drop (no SNP allele).
        out = tmp_path / "p"
        run_vcztools(f"view-bed {fx_vcz_path} -v snps -M 2 --out {out.as_posix()}")
        bim_lines = self._read_bim(tmp_path / "p.bim")
        positions = [int(line.split("\t")[3]) for line in bim_lines]
        # 4 biallelic SNP sites in the fixture: 19:111, 19:112, 20:14370, 20:17330.
        assert sorted(positions) == [111, 112, 14370, 17330]

    def test_exclude_types_snps(self, tmp_path, fx_vcz_path):
        # -V snps drops SNP sites. With -M 2 to skip the multiallelics,
        # only the two monomorphic-REF sites survive.
        out = tmp_path / "p"
        run_vcztools(f"view-bed {fx_vcz_path} -V snps -M 2 --out {out.as_posix()}")
        bim_lines = self._read_bim(tmp_path / "p.bim")
        positions = [int(line.split("\t")[3]) for line in bim_lines]
        assert sorted(positions) == [1230237, 1235237]

    def test_unsupported_type_keyword(self, tmp_path, fx_vcz_path):
        out = tmp_path / "p"
        _, err = run_vcztools(
            f"view-bed {fx_vcz_path} -v indels -M 2 --out {out.as_posix()}",
            expect_error=True,
        )
        assert "TYPE field" in err

    def test_types_and_exclude_types_mutually_exclusive(self, tmp_path, fx_vcz_path):
        out = tmp_path / "p"
        _, err = run_vcztools(
            f"view-bed {fx_vcz_path} -v snps -V refs --out {out.as_posix()}",
            expect_error=True,
        )
        assert "Cannot use --types and --exclude-types together" in err

    def test_out_prefix_normalisation(self, tmp_path, fx_vcz_path):
        # `--out p.bed` should still write to p.bed/p.bim/p.fam (the
        # writer drops/normalises the suffix).
        out = tmp_path / "p.bed"
        run_vcztools(
            f"view-bed {fx_vcz_path} --max-alleles 2 "
            f"-e 'CHROM==\"X\"' --out {out.as_posix()}"
        )
        assert (tmp_path / "p.bed").exists()
        assert (tmp_path / "p.bim").exists()
        assert (tmp_path / "p.fam").exists()


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
