import functools
import shutil
import subprocess

import click.testing as ct
import numpy as np
import pandas as pd
import pytest

import vcztools.cli as cli
import vcztools.constants as constants_module

from .utils import assert_vcfs_close

pytestmark = pytest.mark.skipif(
    shutil.which("bcftools") is None, reason="bcftools not on PATH"
)


@functools.cache
def _run_bcftools_cached(args: str, expect_error: bool) -> tuple[str, str]:
    completed = subprocess.run(
        f"bcftools {args}", capture_output=True, check=False, shell=True
    )
    if expect_error:
        assert completed.returncode != 0
    else:
        assert completed.returncode == 0
    return completed.stdout.decode("utf-8"), completed.stderr.decode("utf-8")


def run_bcftools(args: str, expect_error=False) -> tuple[str, str]:
    """
    Run bcftools (which must be on the PATH) and return stdout and stderr
    as a pair of strings. Results are cached per (args, expect_error) so
    repeated invocations with identical command lines don't re-fork.
    """
    return _run_bcftools_cached(args, expect_error)


def run_vcztools(args: str, expect_error=False) -> tuple[str, str]:
    """Run run_vcztools and return stdout and stderr as a pair of strings."""
    runner = ct.CliRunner()
    result = runner.invoke(
        cli.vcztools_main,
        args,
        catch_exceptions=False,
    )
    if expect_error:
        assert result.exit_code != 0
    else:
        assert result.exit_code == 0
    return result.stdout, result.stderr


@functools.cache
def _bcftools_missing_table(vcf_path) -> tuple[tuple[str, int, float], ...]:
    """Run ``bcftools +fill-tags`` to materialise ``N_MISSING`` and
    ``F_MISSING`` as INFO fields, then read the per-row values out with
    ``bcftools query``. Returns an immutable tuple of
    ``(f"{CHROM}:{POS}", n_missing, f_missing)`` rows in file order so
    duplicate-POS fixtures (sample-split-alleles) are preserved.

    bcftools query has no ``%N_MISSING`` / ``%F_MISSING`` format
    specifier; the fill-tags plugin is the only path through bcftools
    that materialises them as queryable INFO. The plugin needs an
    explicit expression for the integer ``N_MISSING`` (the built-in
    ``N_MISSING`` it knows about is a float, same as ``F_MISSING``).
    """
    cmd = (
        f"bcftools +fill-tags {vcf_path} -- "
        f"-t F_MISSING,'N_MISSING:1=int(N_MISSING)' 2>/dev/null "
        f"| bcftools query -f "
        f"'%CHROM:%POS\\t%INFO/N_MISSING\\t%INFO/F_MISSING\\n' -"
    )
    completed = subprocess.run(cmd, capture_output=True, check=True, shell=True)
    out = completed.stdout.decode("utf-8")
    rows = []
    for line in out.splitlines():
        if len(line.strip()) == 0:
            continue
        key, n_missing_s, f_missing_s = line.split("\t")
        rows.append((key, int(n_missing_s), float(f_missing_s)))
    return tuple(rows)


# fmt: off
@pytest.mark.parametrize(
    ("args", "vcf_file"),
    [
        ("view --no-version", "sample.vcf.gz"),

        ("view --no-version", "msprime_diploid.vcf.gz"),
        ("view --no-version -i 'CHROM == \"20\"'", "sample.vcf.gz"),
        ("view --no-version -i 'CHROM != \"Z\"'", "sample.vcf.gz"),
        ("view --no-version -i 'ID == \"rs6054257\"'", "sample.vcf.gz"),
        ("view --no-version -i 'DB=0'", "sample.vcf.gz"),
        ("view --no-version -i 'DB=1'", "sample.vcf.gz"),
        ("view --no-version -i 'FILTER=\"PASS\"'", "sample.vcf.gz"),
        ("view --no-version -i 'INFO/DP > 10'", "sample.vcf.gz"),
        ("view --no-version -i 'FMT/DP >= 5'", "sample.vcf.gz"),
        ("view --no-version -i 'FMT/DP >= 5 && FMT/GQ > 10'", "sample.vcf.gz"),
        ("view --no-version -i 'FMT/DP >= 5 & FMT/GQ>10'", "sample.vcf.gz"),
        ("view --no-version -i 'FMT/DP>5 && FMT/GQ<45'", "sample.vcf.gz"),
        ("view --no-version -i 'FMT/DP>5 & FMT/GQ<45'", "sample.vcf.gz"),
        (
                "view --no-version -i '(QUAL > 10 || FMT/GQ>10) && POS > 100000'",
                "sample.vcf.gz"
        ),
        (
                "view --no-version -i '(FMT/DP >= 8 | FMT/GQ>40) && POS > 100000'",
                "sample.vcf.gz"
        ),
        (
                "view --no-version -e '(FMT/DP >= 8 | FMT/GQ>40) && POS > 100000'",
                "sample.vcf.gz"
        ),
        ("view --no-version -i 'TYPE=\"ref\"'", "sample.vcf.gz"),
        ("view --no-version -i 'TYPE!=\"ref\"'", "sample.vcf.gz"),
        ("view --no-version -i 'TYPE=\"snp\"'", "sample.vcf.gz"),
        ("view --no-version -i 'TYPE!=\"snp\"'", "sample.vcf.gz"),
        # All alleles are SNPs, 14 rows
        ("view --no-version -i 'TYPE=\"snp\"'", "1kg_2020_chrM.vcf.gz"),
        # Any allele is a SNP, 22 rows
        ("view --no-version -i 'TYPE~\"snp\"'", "1kg_2020_chrM.vcf.gz"),
        # No allele is a SNP, 1 row
        ("view --no-version -i 'TYPE!~\"snp\"'", "1kg_2020_chrM.vcf.gz"),
        # Any allele is not a SNP, 9 rows
        ("view --no-version -i 'TYPE!=\"snp\"'", "1kg_2020_chrM.vcf.gz"),
        # -v/-V/-m/-M flags. Only ``snps`` is fully wired through;
        # indels/mnps/other reach UnsupportedTypeFieldError via the
        # filter parser (see issue #166).
        ("view --no-version -v snps", "sample.vcf.gz"),
        ("view --no-version -V snps", "sample.vcf.gz"),
        ("view --no-version -m 2", "sample.vcf.gz"),
        ("view --no-version -M 2", "sample.vcf.gz"),
        ("view --no-version -m 2 -M 2", "sample.vcf.gz"),
        ("view --no-version -m 3", "sample.vcf.gz"),
        ("view --no-version -v snps -m 2", "sample.vcf.gz"),
        ("view --no-version -v snps -i 'QUAL>10'", "sample.vcf.gz"),
        # The variant-scope -m mask broadcasts against a sample-scope
        # FMT/-prefixed filter — the AND composition handles mixed scope.
        ("view --no-version -m 2 -i 'FMT/DP>3'", "sample.vcf.gz"),
        ("view --no-version -v snps", "1kg_2020_chrM.vcf.gz"),
        ("view --no-version -m 2 -M 2", "1kg_2020_chrM.vcf.gz"),
        # -v/-V/-m/-M composed with variant-scope -i/-e (INFO/site fields).
        ("view --no-version -m 2 -i 'INFO/DP>10'", "sample.vcf.gz"),
        ("view --no-version -M 2 -i 'QUAL>10'", "sample.vcf.gz"),
        ("view --no-version -v snps -i 'INFO/DP>10'", "sample.vcf.gz"),
        ("view --no-version -V snps -i 'QUAL>=10'", "sample.vcf.gz"),
        ("view --no-version -m 2 -M 2 -v snps -i 'POS>10000'", "sample.vcf.gz"),
        # -v/-V/-m/-M composed with sample-scope -i/-e (FMT/-prefixed) —
        # variant-scope synthetic mask broadcasts against the 2-D mask.
        ("view --no-version -m 2 -i 'FMT/GQ>10'", "sample.vcf.gz"),
        ("view --no-version -M 2 -i 'FMT/DP>=5'", "sample.vcf.gz"),
        (
            "view --no-version -v snps -i 'FMT/DP>5 && FMT/GQ>10'",
            "sample.vcf.gz",
        ),
        ("view --no-version -V snps -i 'FMT/GQ>=20'", "sample.vcf.gz"),
        # -v/-V/-m/-M with sample subsetting (-s/-S). Sample-scope filters
        # under bcftools view evaluate over the pre-subset axis, then -s
        # subsets the output.
        ("view --no-version -m 2 -s NA00001", "sample.vcf.gz"),
        ("view --no-version -v snps -s NA00001,NA00002", "sample.vcf.gz"),
        ("view --no-version -M 2 -s ^NA00003", "sample.vcf.gz"),
        # Three-way: variant-scope synthetic AND sample-scope filter AND
        # sample subset. Filter sees the full sample axis; -s subsets
        # after.
        (
            "view --no-version -m 2 -i 'FMT/DP>3' -s NA00002",
            "sample.vcf.gz",
        ),
        (
            "view --no-version -v snps -i 'FMT/GQ>10' -s NA00001,NA00002",
            "sample.vcf.gz",
        ),
        (
            "view --no-version -M 2 -i 'FMT/DP>=5' -s ^NA00003",
            "sample.vcf.gz",
        ),
        # Multi-allelic file exercises the per-allele AC/TYPE paths.
        ("view --no-version -v snps -i 'QUAL>50'", "1kg_2020_chrM.vcf.gz"),
        ("view --no-version -m 2 -M 2 -i 'QUAL>50'", "1kg_2020_chrM.vcf.gz"),
        # N_ALT identifier is exposed in the filter language too.
        ("view --no-version -i 'N_ALT >= 2'", "sample.vcf.gz"),
        ("view --no-version -i 'N_ALT == 1'", "sample.vcf.gz"),
        # N_MISSING / F_MISSING calculated variables computed from
        # call_genotype. sample.vcf.gz has several variants with
        # missing GTs spread across its three samples.
        ("view --no-version -i 'N_MISSING == 0'", "sample.vcf.gz"),
        ("view --no-version -i 'N_MISSING > 0'", "sample.vcf.gz"),
        ("view --no-version -i 'N_MISSING >= 1'", "sample.vcf.gz"),
        ("view --no-version -e 'N_MISSING > 0'", "sample.vcf.gz"),
        ("view --no-version -i 'F_MISSING < 0.05'", "sample.vcf.gz"),
        ("view --no-version -i 'F_MISSING > 0'", "sample.vcf.gz"),
        ("view --no-version -i 'F_MISSING >= 0.3'", "sample.vcf.gz"),
        ("view --no-version -i 'F_MISSING == 0'", "sample.vcf.gz"),
        # Combine with another variant-scope filter.
        (
            "view --no-version -i 'N_MISSING == 0 && POS > 100000'",
            "sample.vcf.gz",
        ),
        # AC / AN / AF used in filter expressions. Stored INFO/AC is
        # mostly missing in sample.vcf.gz, so the no-subset cases use
        # AC>=1 to keep at least the AC=[1,1] row alive. The -s cases
        # below also exercise force_recompute=True, where AC/AN/AF
        # come from the C-backed virtual-field path.
        ("view --no-version -i 'AC>=1'", "sample.vcf.gz"),
        ("view --no-version -i 'AN>=4'", "sample.vcf.gz"),
        ("view --no-version -i 'AF>=0.3'", "sample.vcf.gz"),
        ("view --no-version -s NA00001,NA00002 -i 'AC>=1'", "sample.vcf.gz"),
        ("view --no-version -s NA00001,NA00002 -i 'AN==4'", "sample.vcf.gz"),
        ("view --no-version -G", "sample.vcf.gz"),
        (
                "view --no-update --no-version --samples-file "
                "tests/data/txt/samples.txt",
                "sample.vcf.gz"),
        ("view -I --no-version -S tests/data/txt/samples.txt", "sample.vcf.gz"),
        ("view --no-version -s NA00001", "sample.vcf.gz"),
        ("view --no-version -s NA00001,NA00003", "sample.vcf.gz"),
        ("view --no-version -s HG00096", "1kg_2020_chrM.vcf.gz"),
        ("view --no-version -s tsk_0,tsk_1", "msprime_diploid.vcf.gz"),
        ("view --no-version -s tsk_0,tsk_1,tsk_2", "msprime_diploid.vcf.gz"),
        ("view --no-version -s 'NO_SAMPLE,NA00001' --force-samples", "sample.vcf.gz"),
        ("view --no-version -s ^NA00001", "sample.vcf.gz"),
        ("view --no-version -s ^NA00003,NA00002", "sample.vcf.gz"),
        ("view --no-version -s ^NA00003,NA00002,NA00003", "sample.vcf.gz"),
        ("view --no-version -S ^tests/data/txt/samples.txt", "sample.vcf.gz"),
        # Complement collapses duplicate excludes
        ("view --no-version -s ^NA00001,NA00001", "sample.vcf.gz"),
        # Complement with one unknown name -> the unknown is dropped
        ("view --no-version -s ^NA00001,NOPE --force-samples", "sample.vcf.gz"),
        # Complement with all-unknown excludes -> every sample retained
        ("view --no-version -s ^NOPE1,NOPE2 --force-samples", "sample.vcf.gz"),
        (
            "view --no-version -r '20:1230236-' -i 'FMT/DP>3' -s 'NA00002,NA00003'",
            "sample.vcf.gz"
        ),
        (
            "view --no-version -R tests/data/txt/regions-3col.tsv -i 'FMT/DP>3' -s 'NA00002,NA00003'",  # noqa: E501
            "sample.vcf.gz"
        ),
        (
            "view --no-version -t '20:1230236-' -i 'FMT/DP>3' -s 'NA00002,NA00003'",
            "sample.vcf.gz"
        ),
        (
            "view --no-version -t '^20:1230236-' -i 'FMT/DP>3' -s 'NA00002,NA00003'",
            "sample.vcf.gz"
        ),
        (
            "view --no-version -T tests/data/txt/regions-3col.tsv -i 'FMT/DP>3' -s 'NA00002,NA00003'",  # noqa: E501
            "sample.vcf.gz"
        ),
        (
            "view --no-version -T ^tests/data/txt/regions-3col.tsv -i 'FMT/DP>3' -s 'NA00002,NA00003'",  # noqa: E501
            "sample.vcf.gz"
        ),
        (
            "view --no-version -i 'FILTER=\"VQSRTrancheSNP99.80to100.00\"'",
            "1kg_2020_chrM.vcf.gz"
        ),
        (
            "view --no-version -i 'FILTER!=\"VQSRTrancheSNP99.80to100.00\"'",
            "1kg_2020_chrM.vcf.gz"
        ),
        (
            "view --no-version -i 'FILTER~\"VQSRTrancheINDEL99.00to100.00\"'",
            "1kg_2020_chrM.vcf.gz"
        ),
    ],
    # This is necessary when trying to run individual tests, as the arguments above
    # make for unworkable command lines
    # ids=range(36),
)
# fmt: on
def test_vcf_output(tmp_path, fx_all_vcz, args, vcf_file):
    fx = fx_all_vcz[vcf_file]

    bcftools_out, _ = run_bcftools(f"{args} {fx.vcf_path}")
    bcftools_out_file = tmp_path.joinpath("bcftools_out.vcf")
    with open(bcftools_out_file, "w") as f:
        f.write(bcftools_out)

    vcztools_out, _ = run_vcztools(f"{args} {fx.zip_path}")
    vcztools_out_file = tmp_path.joinpath("vcztools_out.vcf")
    with open(vcztools_out_file, "w") as f:
        f.write(vcztools_out)

    assert_vcfs_close(bcftools_out_file, vcztools_out_file)


@pytest.mark.parametrize(
    ("args", "vcf_file"),
    [("view --no-version", "sample.vcf.gz")],
)
def test_vcf_output_with_output_option(tmp_path, fx_all_vcz, args, vcf_file):
    fx = fx_all_vcz[vcf_file]

    bcftools_out_file = tmp_path.joinpath("bcftools_out.vcf")
    vcztools_out_file = tmp_path.joinpath("vcztools_out.vcf")

    bcftools_args = f"{args} -o {bcftools_out_file}"
    vcztools_args = f"{args} -o {vcztools_out_file}"

    run_bcftools(f"{bcftools_args} {fx.vcf_path}")
    run_vcztools(f"{vcztools_args} {fx.zip_path}")

    assert_vcfs_close(bcftools_out_file, vcztools_out_file)


@pytest.mark.parametrize(
    ("args", "vcf_name"),
    [
        ("index -n", "sample.vcf.gz"),
        ("index --nrecords", "1kg_2020_chrM.vcf.gz"),
        ("index -s", "sample.vcf.gz"),
        ("index --stats", "1kg_2020_chrM.vcf.gz"),
        ("query -l", "sample.vcf.gz"),
        ("query --list-samples", "1kg_2020_chrM.vcf.gz"),
        ("query -f 'A'", "sample.vcf.gz"),
        (r"query -f 'A\n'", "sample.vcf.gz"),
        ("query -N -f 'A'", "sample.vcf.gz"),
        (r"query -N -f 'A\n'", "sample.vcf.gz"),
        ("query -f '%CHROM:%POS'", "sample.vcf.gz"),
        ("query -f '[%CHROM %POS %GT]'", "sample.vcf.gz"),
        ("query -f '%INFO/DP'", "sample.vcf.gz"),
        ("query -f '%DP'", "sample.vcf.gz"),
        ("query -f '%AC{0}'", "sample.vcf.gz"),
        ("query -f '%AF{0}'", "sample.vcf.gz"),
        (r"query -f '%REF\t%ALT'", "sample.vcf.gz"),
        ("query -f '%ALT{1}'", "sample.vcf.gz"),
        ("query -f '%ID'", "sample.vcf.gz"),
        ("query -f '%QUAL'", "sample.vcf.gz"),
        ("query -f '%FILTER'", "sample.vcf.gz"),
        ("query --format '%FILTER'", "1kg_2020_chrM.vcf.gz"),
        ("query -f '%POS' -i 'POS=112'", "sample.vcf.gz"),
        ("query -f '%POS' -e 'POS=112'", "sample.vcf.gz"),
        (r"query -f '[%CHROM\t]'", "sample.vcf.gz"),
        (r"query -f '[%CHROM\t]' -i 'POS=112'", "sample.vcf.gz"),
        ("query -f '[%CHROM:%POS %SAMPLE %GT]'", "sample.vcf.gz"),
        ("query -f '[%SAMPLE %GT %DP]'", "sample.vcf.gz"),
        (
            "query -f '[%POS %SAMPLE %GT %DP %GQ]' -i 'INFO/DP >= 5'",
            "sample.vcf.gz",
        ),
        (
            "query -f '[%POS %QUAL]' -i'(QUAL > 10 && POS > 100000)'",
            "sample.vcf.gz",
        ),
        # Examples from bcftools query documentation
        ("query -f '%CHROM  %POS  %REF  %ALT{0}'", "sample.vcf.gz"),
        (r"query -f '%CHROM\t%POS\t%REF\t%ALT[\t%SAMPLE=%GT]'", "sample.vcf.gz"),
        (r"query -f 'GQ:[ %GQ] \t GT:[ %GT]'", "sample.vcf.gz"),
        # POS0 not supported
        # (r"query -f '%CHROM\t%POS0\t%END\t%ID'", "sample.vcf.gz"),
        # Filtering on GT not supported
        # ("query -f [%CHROM:%POS %SAMPLE %GT]' -i'GT=\"alt\"'", "sample.vcf.gz"),
        # Indexing not supported in filtering
        # ("query  -f '%AC{1}' -i 'AC[1]>10' ", "sample.vcf.gz"),
        # TODO fill-out more of these when supported for more stuff is available
        # in filtering
        ("query -f '%CHROM %POS %FILTER' -i 'FILTER=\"PASS\"'", "sample.vcf.gz"),
        # Per-sample query tests
        (
            r"query -f '[%CHROM %POS %SAMPLE %GT %DP %GQ\n]' -i 'FMT/DP>3'",
            "sample.vcf.gz",
        ),
        (
            r"query -f '[%CHROM %POS %SAMPLE %GT %DP %GQ\n]' -i 'FMT/GQ>30'",
            "sample.vcf.gz",
        ),
        (
            r"query -f '[%CHROM %POS %SAMPLE %GT %DP %GQ\n]' -i 'FMT/DP>3 & FMT/GQ>30'",
            "sample.vcf.gz",
        ),
        (
            r"query -f '[%CHROM %POS %SAMPLE %GT %DP %GQ\n]' -i 'FMT/DP>3 && FMT/GQ>30'",  # noqa: E501
            "sample.vcf.gz",
        ),
        (
            r"query -f '[%CHROM %POS %SAMPLE %GT %DP %GQ\n]' -r '20:1230236-' -i 'FMT/DP>3' -s 'NA00002,NA00003'",  # noqa: E501
            "sample.vcf.gz",
        ),
        (
            r"query -f '[%CHROM %POS %SAMPLE %GT %DP %GQ\n]' -r '20:1230236-' -i 'FMT/DP>3' -S tests/data/txt/samples.txt",  # noqa: E501
            "sample.vcf.gz",
        ),
        (
            r"query -f '[%CHROM %POS %SAMPLE %GT %DP %GQ\n]' -R tests/data/txt/regions-3col.tsv -i 'FMT/DP>3' -s 'NA00002,NA00003'",  # noqa: E501
            "sample.vcf.gz",
        ),
        (
            r"query -f '[%CHROM %POS %SAMPLE %GT %DP %GQ\n]' -T ^tests/data/txt/regions-3col.tsv -i 'FMT/DP>3' -s 'NA00002,NA00003'",  # noqa: E501
            "sample.vcf.gz",
        ),
        # Second subfield index
        ("query -f '%AC{1}'", "sample.vcf.gz"),
        # Multi-valued FORMAT field in sample loop (restrict to rows where HQ
        # is present to avoid absent-vs-all-missing ambiguity in zarr)
        (r"query -f '[%HQ ]\n' -r '19,20:1-1230237'", "sample.vcf.gz"),
        # Sample exclusion with query
        (r"query -f '[%SAMPLE %GT\n]' -s '^NA00001'", "sample.vcf.gz"),
        # Query on 1kg data (1 sample, different fields)
        ("query -f '%CHROM:%POS'", "1kg_2020_chrM.vcf.gz"),
        (r"query -f '[%GT]'", "1kg_2020_chrM.vcf.gz"),
    ],
)
def test_output(tmp_path, fx_all_vcz, args, vcf_name):
    fx = fx_all_vcz[vcf_name]
    bcftools_output, _ = run_bcftools(f"{args} {fx.vcf_path}")
    vcztools_output, _ = run_vcztools(f"{args} {fx.zip_path}")

    assert vcztools_output == bcftools_output


@pytest.mark.parametrize(
    "expr",
    [
        # Check arithmetic evaluation in filter queries. All these should
        # result to POS=112, which exists.
        "POS=(111 + 1)",
        "POS =(224 / 2)",
        "POS= (112 * 3) / 3",
        "POS=(112 * 3 / 3   )",
        "POS=25 * 4 + 24 / 2",
        "POS=112 * -1 * -1",
        "-POS=-112",
        "POS=112.25 - 1 / 4",
        "POS=112.25e3 * 1e-3 - 0.25",
    ],
)
def test_query_arithmethic(tmp_path, fx_sample_vcz, expr):
    args = f"query -f '%POS' -i '{expr}'"
    bcftools_output, _ = run_bcftools(f"{args} {fx_sample_vcz.vcf_path}")
    vcztools_output, _ = run_vcztools(f"{args} {fx_sample_vcz.zip_path}")

    assert vcztools_output == bcftools_output
    assert vcztools_output == "112\n"


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        # Check boolean logic evaluation. Will evaluate this with
        # POS=112, so POS=112 is True and POS!=112 is False
        ("POS==112 || POS!=112", True),
        ("POS==112 && POS!=112", False),
        ("POS==112 || POS!=112 && POS!= 112", True),
        ("(POS==112 || POS!=112) && POS!= 112", False),
    ],
)
def test_query_logic_precendence(tmp_path, fx_sample_vcz, expr, expected):
    args = f"query -f '%POS' -i 'POS=112 && ({expr})'"
    bcftools_output, _ = run_bcftools(f"{args} {fx_sample_vcz.vcf_path}")
    vcztools_output, _ = run_vcztools(f"{args} {fx_sample_vcz.zip_path}")

    assert vcztools_output == bcftools_output
    num_lines = len(list(vcztools_output.splitlines()))
    assert num_lines == int(expected)


# fmt: off
@pytest.mark.parametrize(
    ("args", "vcf_name", "bcftools_error_string"),
    [
        ("index -ns", "sample.vcf.gz", True),
        ("query -f '%POS' -i 'INFO/DP > 10' -e 'INFO/DP < 50'", "sample.vcf.gz", True),  # noqa: E501
        ("query -f '%GT'", "sample.vcf.gz", True),
        ("query -f '%HQ'", "sample.vcf.gz", True),
        ("query -f '%SAMPLE'", "sample.vcf.gz", True),
        ("view -i 'INFO/DP > 10' -e 'INFO/DP < 50'", "sample.vcf.gz", True),
        ("view -i 'DP > 10'", "sample.vcf.gz", True),
        # bcftools output does not start with "Error"
        ("view -i 'FILTER=\"F\"'", "sample.vcf.gz", False),
        # Duplicate samples in regular mode are rejected (bcftools error
        # has the [E:: prefix from htslib, not "Error:")
        ("view -s NA00001,NA00001", "sample.vcf.gz", True),
        # --force-samples does not relax the duplicate rule
        ("view -s NA00001,NA00001 --force-samples", "sample.vcf.gz", True),
        # Duplicates trip the check even after --force-samples removes
        # unknowns. bcftools stderr starts with "Warn: ..." for the unknown
        # before the [E:: line, so we don't pin the prefix here.
        ("view -s NA00001,NA00001,NOPE --force-samples", "sample.vcf.gz", False),
        # Empty-string sample is treated as an unknown sample
        ("view -s ''", "sample.vcf.gz", True),
    ],
)
# fmt: on
def test_error(tmp_path, fx_all_vcz, args, vcf_name, bcftools_error_string):
    fx = fx_all_vcz[vcf_name]

    _, bcftools_error = run_bcftools(f"{args} {fx.vcf_path}", expect_error=True)
    if bcftools_error_string:
        assert bcftools_error.startswith("Error:") or bcftools_error.startswith("[E::")

    _, vcztools_error = run_vcztools(f"{args} {fx.zip_path}", expect_error=True)
    assert "Error:" in vcztools_error


@pytest.mark.parametrize(
    "args",
    [
        # bcftools accepts each of these and emits variant-only output.
        # vcztools refuses because matching bcftools AC/AN semantics for
        # an empty subset would require significant internal complexity.
        "view --no-version -s 'NO_SAMPLE' --force-samples",
        "view --no-version -s FOO,BAR,BAZ --force-samples",
        "view --no-version -s ^NA00001,NA00002,NA00003",
        "view --no-version -S tests/data/txt/samples_empty.txt",
    ],
)
def test_empty_sample_set_refused(fx_all_vcz, args):
    fx = fx_all_vcz["sample.vcf.gz"]
    _, vcztools_error = run_vcztools(f"{args} {fx.zip_path}", expect_error=True)
    assert "Empty sample set is not supported" in vcztools_error
    assert "github.com/sgkit-dev/vcztools/issues" in vcztools_error


@pytest.mark.parametrize(
    "args",
    [
        "view --no-version -s ''",
        "view --no-version -s '' --force-samples",
    ],
)
def test_null_sample_name_refused(fx_all_vcz, args):
    # The empty string is the reserved null-sample ID, never a valid
    # ``-s`` target, so it is refused regardless of --force-samples.
    fx = fx_all_vcz["sample.vcf.gz"]
    _, vcztools_error = run_vcztools(f"{args} {fx.zip_path}", expect_error=True)
    assert "empty string is not a valid sample name" in vcztools_error


class TestRegionTargetSemantics:
    """Validate region/target selection against bcftools.

    Uses ``query -f '%CHROM %POS\\n'`` so we compare only which variants
    are selected, without noise from FORMAT fields or VCF headers.
    """

    FMT = r"query -f '%CHROM %POS\n'"

    # fmt: off
    @pytest.mark.parametrize(
        "args",
        [
            # --- standalone regions (no filters) ---
            "-r '20'",
            "-r '20:1230237-1235237'",
            "-r '19,X'",
            "-R tests/data/txt/regions-3col.tsv",
            # --- standalone targets (no filters) ---
            "-t '20'",
            "-t '20:1230237-1235237'",
            "-t '^20'",
            "-T tests/data/txt/regions-3col.tsv",
            "-T ^tests/data/txt/regions-3col.tsv",
            # --- regions vs targets semantic difference ---
            "-r 'X:11'",      # overlap semantics: includes POS=10
            "-t 'X:11'",      # exact position: empty result
            # --- regions + targets combined ---
            "-r '20' -t '^20:1110696-'",
            # --- regions/targets + include ---
            "-r '20' -i 'POS > 100000'",
            "-t '20' -i 'POS > 100000'",
            # --- regions/targets + exclude ---
            "-r '20' -e 'POS < 100000'",
            "-t '20' -e 'POS < 100000'",
        ],
    )
    # fmt: on
    def test_variant_selection(self, fx_sample_vcz, args):
        cmd = f"{self.FMT} {args}"
        bcftools_output, _ = run_bcftools(f"{cmd} {fx_sample_vcz.vcf_path}")
        vcztools_output, _ = run_vcztools(f"{cmd} {fx_sample_vcz.zip_path}")
        assert vcztools_output == bcftools_output


class TestFilterExpressionLanguage:
    """Validation coverage of the bcftools filter-expression subset
    implemented in :mod:`vcztools.bcftools_filter`.

    The implemented subset is: arithmetic (``+ - * /``, unary ``-``);
    comparison (``= == != < <= > >=``); variant-scope logical
    (``& |``) and sample-scope logical (``&& ||``); parenthesised
    grouping; the special identifiers ``CHROM``, ``FILTER`` (set
    semantics with ``= != ~ !~``), and ``TYPE`` (only ``"ref"`` /
    ``"snp"``, with ``= == != ~ !~``); ``INFO/X`` / ``FORMAT/X`` /
    ``FMT/X`` / bare ``X`` identifiers (with Number=A per-allele
    collapse at the root); and ``-e`` as the inverse of ``-i``.

    Each method below groups cases by feature area and compares
    vcztools to bcftools via ``query -f '%CHROM %POS\\n'``. Known
    divergences live in :meth:`test_known_divergences`, marked
    ``xfail(strict=True)`` so they surface in CI as bugs to fix.

    Features that deliberately raise ``UnsupportedFilteringFeatureError``
    at parse time (missing-value literal ``"."`` #163, GT #165, TYPE
    outside ``ref/snp`` #166, array subscripts #167, regex ``~``/``!~``
    applied to non-FILTER/TYPE #174, file refs #175, functions #190,
    higher-dim FORMAT #232) are covered in ``tests/test_filter.py``
    rather than here — they don't round-trip through bcftools.

    For ``-s`` + filter ordering coverage see
    :class:`TestSampleSubsetFilterSemantics`.
    """

    FMT = r"query -f '%CHROM %POS\n'"

    def _check(self, fx_sample_vcz, flag, expr):
        cmd = f"{self.FMT} {flag} '{expr}'"
        bcftools_output, _ = run_bcftools(f"{cmd} {fx_sample_vcz.vcf_path}")
        vcztools_output, _ = run_vcztools(f"{cmd} {fx_sample_vcz.zip_path}")
        assert vcztools_output == bcftools_output

    # fmt: off
    @pytest.mark.parametrize(
        "expr",
        [
            "POS > 100000 + 50000",
            "POS > 2 * 100000",
            "INFO/DP / 2 >= 5",
            "INFO/DP + 1 > 15",
            "POS - 10000 > 0",
        ],
    )
    # fmt: on
    def test_arithmetic(self, fx_sample_vcz, expr):
        self._check(fx_sample_vcz, "-i", expr)

    # fmt: off
    @pytest.mark.parametrize(
        "expr",
        [
            # Parenthesised OR-within-AND.
            '(POS>1000 || ID=="rs6054257") && QUAL>0',
            # Default precedence: && binds tighter than ||.
            "POS>1000 && QUAL>0 || POS<200",
            # Parenthesised AND-within-OR.
            '(QUAL>10 && POS>100000) || ID=="rs6054257"',
        ],
    )
    # fmt: on
    def test_parentheses_and_precedence(self, fx_sample_vcz, expr):
        self._check(fx_sample_vcz, "-i", expr)

    # fmt: off
    @pytest.mark.parametrize(
        ("flag", "expr"),
        [
            # Bare identifier resolves to INFO-only fields in this VCF.
            ("-i", "AN > 0"),
            ("-i", "AC != 0"),
            # Bare identifier resolves to the FMT field (no INFO/GQ).
            ("-i", "GQ > 40"),
            # Bare ID + the `-e` path.
            ("-e", 'ID == "rs6054257"'),
        ],
    )
    # fmt: on
    def test_bare_identifier(self, fx_sample_vcz, flag, expr):
        self._check(fx_sample_vcz, flag, expr)

    # fmt: off
    @pytest.mark.parametrize(
        ("flag", "expr"),
        [
            ("-i", "DB=1"),
            ("-i", "DB=0"),
            ("-i", "H2=1"),
            # Exclude of a flag: inverse of -i 'DB=1'.
            ("-e", "DB=1"),
        ],
    )
    # fmt: on
    def test_flag_fields(self, fx_sample_vcz, flag, expr):
        self._check(fx_sample_vcz, flag, expr)

    # fmt: off
    @pytest.mark.parametrize(
        "expr",
        [
            # Exact single-filter match.
            'FILTER="q10"',
            # Exact inequality.
            'FILTER!="PASS"',
            # Subset match (any filter named).
            'FILTER~"q10"',
            # Complement of subset match.
            'FILTER!~"q10"',
            # Compound filter string — order-insensitive set compare.
            # Fixture has no record with both q10 and s50 set → empty.
            'FILTER="s50;q10"',
            'FILTER~"PASS"',
        ],
    )
    # fmt: on
    def test_filter_set_ops(self, fx_sample_vcz, expr):
        self._check(fx_sample_vcz, "-i", expr)

    # fmt: off
    @pytest.mark.parametrize(
        ("flag", "expr"),
        [
            ("-i", 'TYPE="ref"'),
            ("-i", 'TYPE="snp"'),
            ("-i", 'TYPE!="snp"'),
            ("-i", 'TYPE~"snp"'),
            ("-i", 'TYPE!~"snp"'),
            ("-e", 'TYPE="snp"'),
        ],
    )
    # fmt: on
    def test_type_field(self, fx_sample_vcz, flag, expr):
        self._check(fx_sample_vcz, flag, expr)

    # fmt: off
    @pytest.mark.parametrize(
        ("flag", "expr"),
        [
            # `=` (single) and `==` should be equivalent.
            ("-i", 'CHROM="20"'),
            ("-i", 'CHROM=="20"'),
            ("-i", 'CHROM!="20"'),
            # Non-existent contig → empty result.
            ("-i", 'CHROM=="nonexistent"'),
            # Exclude path.
            ("-e", 'CHROM=="X"'),
        ],
    )
    # fmt: on
    def test_chrom_equality(self, fx_sample_vcz, flag, expr):
        self._check(fx_sample_vcz, flag, expr)

    # fmt: off
    @pytest.mark.parametrize(
        "expr",
        [
            # INFO/AC (Number=A) with equality/inequality.
            "INFO/AC==1",
            "INFO/AC!=0",
            # INFO/AF (Number=A, float) with scalar threshold.
            "INFO/AF>0.4",
            "INFO/AF>0.6",
            # Compound per-allele: both sides see the 2-D array;
            # root collapses via np.any.
            "INFO/AC>0 && INFO/AC<2",
        ],
    )
    # fmt: on
    def test_per_allele_info(self, fx_sample_vcz, expr):
        self._check(fx_sample_vcz, "-i", expr)

    # fmt: off
    @pytest.mark.parametrize(
        "expr",
        [
            # QUAL=. at 20:1235237 — NaN encoding must exclude.
            "QUAL>0",
            # INFO/AN present only at 20:1234567.
            "INFO/AN>0",
            # INFO/DP present at a subset of records.
            "INFO/DP>=14",
        ],
    )
    # fmt: on
    def test_missing_data(self, fx_sample_vcz, expr):
        self._check(fx_sample_vcz, "-i", expr)

    # fmt: off
    @pytest.mark.parametrize(
        "expr",
        [
            "POS==14370",
            "INFO/AN>0",
            'FILTER="PASS"',
            "FMT/DP>=5",
        ],
    )
    # fmt: on
    def test_exclude_inverse(self, fx_sample_vcz, expr):
        self._check(fx_sample_vcz, "-e", expr)

    # fmt: off
    @pytest.mark.parametrize(
        "expr",
        [
            # Regression: INFO/AC is Number=A and is missing on every
            # record except 20:1234567 in sample.vcf.gz. The integer
            # missing/fill sentinels (-1/-2) must not satisfy ``<2``.
            "INFO/AC<2",

            # ---- BUG 2: bare flag identifier as the whole predicate ----
            #
            #   bcftools query -i 'DB' sample.vcf.gz
            #     → empty (bcftools rejects bare flag as predicate)
            #   vcztools query -i 'DB' sample.vcz.zip
            #     → 20 14370, 20 1110696   (the two DB=1 records)
            #
            # Root cause: ``Identifier.eval`` returns the raw int
            # array; ``BcftoolsFilter.evaluate`` then uses it directly
            # as the include-mask, so truthy (=1) records pass the
            # filter.
            #
            # Fix direction: in ``BcftoolsFilter.__init__`` reject a
            # parse tree whose root is a plain ``Identifier`` (no
            # comparison / logical / filter-set operator wrapping
            # it). The correct bcftools form is ``DB=1`` — callers
            # should be told so.
            pytest.param(
                "DB",
                marks=pytest.mark.xfail(
                    strict=True,
                    reason=(
                        "Bare flag identifier is not a valid "
                        "bcftools predicate; vcztools evaluates it "
                        "as a truthy mask. See comment for details."
                    ),
                ),
            ),
        ],
    )
    # fmt: on
    def test_known_divergences(self, fx_sample_vcz, expr):
        self._check(fx_sample_vcz, "-i", expr)

    # Regression check: arithmetic on NaN-encoded missing QUAL
    # (sample.vcf.gz has ``QUAL=.`` at 20:1235237) must not emit
    # ``RuntimeWarning: invalid value encountered in ...`` from either
    # the unary-minus or binary-arithmetic path.
    @pytest.mark.parametrize(
        "expr",
        [
            # UnaryMinus path.
            "-QUAL>-20",
            # BinaryOperator path.
            "QUAL + 1 > 11",
        ],
    )
    def test_nan_arithmetic_warning(self, fx_sample_vcz, expr):
        # Run vcztools in a subprocess rather than through the in-
        # process CliRunner: Python's per-module warning registry
        # and numpy's FPU state both persist across tests in the
        # same process, making ``warnings.catch_warnings`` unreliable
        # for detecting a warning that earlier tests may have
        # "consumed". A fresh subprocess guarantees the RuntimeWarning
        # surfaces on stderr whenever the buggy code path runs.
        cmd = f"{self.FMT} -i '{expr}' {fx_sample_vcz.vcf_path}"
        bcftools_output, _ = run_bcftools(cmd)
        result = subprocess.run(
            f"uv run vcztools {self.FMT} -i '{expr}' {fx_sample_vcz.zip_path}",
            capture_output=True,
            check=True,
            shell=True,
        )
        vcztools_output = result.stdout.decode("utf-8")
        vcztools_stderr = result.stderr.decode("utf-8")
        assert vcztools_output == bcftools_output
        assert "RuntimeWarning" not in vcztools_stderr, (
            f"unexpected RuntimeWarning in stderr:\n{vcztools_stderr}"
        )


class TestSampleSubsetFilterSemantics:
    """Sample subset vs filter-evaluation ordering.

    bcftools applies ``-i``/``-e`` filter expressions at different
    stages depending on the subcommand:

    - ``bcftools view``: filters evaluate on the ORIGINAL record,
      before ``-s``/``-S`` subsetting (and before INFO/AC,AN
      recompute). Both INFO-scope and FMT-scope filters see the full
      sample set.
    - ``bcftools query``: INFO-scope filters evaluate on the original
      record too (matching ``view``). FMT-scope filters, however,
      evaluate on the SUBSET record — sample subsetting happens first.

    vcztools matches both via ``set_bcftools_semantics``: the ``view``
    CLI passes ``full_sample_filter=True`` (pre-subset evaluation over
    the full non-null sample axis); the ``query`` CLI uses the default
    ``full_sample_filter=False`` (the sample subset, post-subset
    evaluation). INFO-scope filters touch no sample dimension, so the
    axis choice is a no-op for them — both read the stored INFO.

    The three methods below cover:

    - ``test_query_variant_selection``: ``query`` cases where pre/post
      agree (INFO-scope, and a mixed case chosen so the FMT leg does
      not alter the outcome).
    - ``test_query_fmt_post_subset``: ``query`` cases with FMT-scope
      filters under ``-s`` — pins vcztools to bcftools query's
      post-subset semantics.
    - ``test_view_fmt_pre_subset``: ``view`` cases exercising
      pre-subset FMT semantics, with selections where pre- and
      post-subset would give different answers. vcztools must match
      bcftools view's pre-subset answer.

    The ``query`` methods use ``query -f '%CHROM %POS\\n'`` to compare
    selected variants without FORMAT/header noise; the ``view`` method
    uses :func:`assert_vcfs_close` to compare full VCF output without
    tripping on header/INFO/FORMAT field ordering differences.
    """

    FMT = r"query -f '%CHROM %POS\n'"

    # fmt: off
    @pytest.mark.parametrize(
        "args",
        [
            # Scalar INFO filter x sample subset: only 20:1234567 has
            # source INFO/AN; if the filter ran post-subset (with AN
            # recomputed to 0) no row would pass.
            "-s NA00001 -i 'INFO/AN>0'",
            "-s NA00003 -i 'INFO/AN>0'",
            "-s ^NA00001 -i 'INFO/AN>0'",
            "-S tests/data/txt/samples.txt -i 'INFO/AN>0'",
            "-S ^tests/data/txt/samples.txt -i 'INFO/AN>0'",

            # Per-allele INFO filter (2-D variant-scoped mask — used
            # to crash with ``IndexError`` when combined with ``-s``).
            "-s NA00001 -i 'INFO/AC>0'",
            "-s NA00003 -i 'INFO/AC>0'",
            "-s ^NA00002 -i 'INFO/AC>0'",
            "-s NA00001 -e 'INFO/AC=0'",

            # Complement + scalar INFO filter.
            "-s ^NA00003 -i 'INFO/AN>0'",

            # Region + sample + INFO filter.
            "-r '20' -s NA00001 -i 'INFO/AN>0'",

            # Mixed variant-scope and sample-scope filters under -s.
            # Selection chosen so the FMT leg gives the same answer
            # pre- and post-subset (both empty), side-stepping the
            # query-vs-view divergence — see class docstring.
            "-s NA00001 -i 'INFO/AN>0 && FMT/DP>5'",
        ],
    )
    # fmt: on
    def test_query_variant_selection(self, fx_sample_vcz, args):
        cmd = f"{self.FMT} {args}"
        bcftools_output, _ = run_bcftools(f"{cmd} {fx_sample_vcz.vcf_path}")
        vcztools_output, _ = run_vcztools(f"{cmd} {fx_sample_vcz.zip_path}")
        assert vcztools_output == bcftools_output

    # fmt: off
    @pytest.mark.parametrize(
        "args",
        [
            # -s NA00001 & FMT/DP>5: DP per sample at 20:14370 is
            # (1, 8, 5). bcftools view picks 20:14370 + 20:1110696;
            # bcftools query (NA00001=1 only) picks only 20:1110696.
            "-s NA00001 -i 'FMT/DP>5'",

            # -s NA00002 & FMT/DP>5: at 20:1110696 NA00002=0 so
            # bcftools query drops it; view keeps it.
            "-s NA00002 -i 'FMT/DP>5'",

            # File-based subset (NA00001, NA00003).
            "-S tests/data/txt/samples.txt -i 'FMT/DP>5'",

            # Complement + FMT filter: ^NA00002 = {NA00001, NA00003};
            # at 20:14370 max DP in subset is 5, fails DP>5 in query;
            # view sees NA00002=8, passes.
            "-s ^NA00002 -i 'FMT/DP>5'",

            # Per-sample conjunction: at 20:14370 no single sample
            # satisfies GQ<45 && DP>=5 in subset {NA00001 alone}; but
            # view sees NA00003 with GQ=43, DP=5 and keeps it.
            "-s NA00001 -i 'FMT/GQ<45 && FMT/DP>=5'",
        ],
    )
    # fmt: on
    def test_query_fmt_post_subset(self, fx_sample_vcz, args):
        cmd = f"{self.FMT} {args}"
        bcftools_output, _ = run_bcftools(f"{cmd} {fx_sample_vcz.vcf_path}")
        vcztools_output, _ = run_vcztools(f"{cmd} {fx_sample_vcz.zip_path}")
        assert vcztools_output == bcftools_output

    # fmt: off
    @pytest.mark.parametrize(
        "args",
        [
            # Same selections as the divergence cases above, but via
            # `view` — locks in vcztools's pre-subset FMT semantics.
            "-s NA00001 -i 'FMT/DP>5'",
            "-s NA00002 -i 'FMT/DP>5'",
            "-s NA00003 -i 'FMT/DP>=5'",
            "-s ^NA00002 -i 'FMT/DP>5'",
            "-S tests/data/txt/samples.txt -i 'FMT/DP>5'",
            "-s NA00001 -i 'FMT/GQ<45 && FMT/DP>=5'",
        ],
    )
    # fmt: on
    def test_view_fmt_pre_subset(self, tmp_path, fx_sample_vcz, args):
        cmd = f"view --no-version {args}"
        bcftools_output, _ = run_bcftools(f"{cmd} {fx_sample_vcz.vcf_path}")
        vcztools_output, _ = run_vcztools(f"{cmd} {fx_sample_vcz.zip_path}")
        bcftools_out_file = tmp_path / "bcftools_out.vcf"
        bcftools_out_file.write_text(bcftools_output)
        vcztools_out_file = tmp_path / "vcztools_out.vcf"
        vcztools_out_file.write_text(vcztools_output)
        assert_vcfs_close(bcftools_out_file, vcztools_out_file)


# fmt: off
class TestChr22:
    """
    Validation tests using the chr22 fixture (100 variants, 100 samples).
    Each test case includes the expected number of variants so the reader
    can see the mixture of result sizes at a glance.
    """

    @pytest.mark.parametrize(
        ("args", "expected_variants"),
        [
            # All variants
            ("view --no-version", 100),
            ("view --no-version -G", 100),
            ("view --no-version -s ^HG00096", 100),
            # Large subsets
            ("view --no-version -i 'QUAL>100'", 86),
            ("view --no-version -i 'INFO/DP>5000'", 86),
            # Medium subsets
            ("view --no-version -i 'FMT/DP>10'", 72),
            ("view --no-version -i 'FMT/GQ>50'", 35),
            ("view --no-version -r 'chr22:10513000-10514000'", 34),
            ("view --no-version -r 'chr22:10510000-10511000' -i 'QUAL>100'", 24),
            # Small subsets
            ("view --no-version -i 'INFO/AC>2'", 9),
            ("view --no-version -e 'TYPE=\"snp\"'", 9),
            ("view --no-version -i 'INFO/AF>0.1'", 6),
            ("view --no-version -r 'chr22:10510000-10510200'", 6),
            ("view --no-version -i 'INFO/AC>10'", 4),
            ("view --no-version -r 'chr22:10512000-10513000'", 3),
            ("view --no-version -i 'FILTER=\"PASS\"'", 2),
            ("view --no-version -i 'INFO/AF>0.5'", 1),
            # Empty sets
            ("view --no-version -r 'chr22:10514100-'", 0),
            ("view --no-version -r 'chr22:1-10510000'", 0),
            ("view --no-version -i 'INFO/AC>200'", 0),
            # -s + -i/-e: filter evaluates on original record; sample
            # subsetting and INFO/AC,AN recompute happen after.
            ("view --no-version -s HG00096,HG00101 -i 'INFO/AC>10'", 4),
            ("view --no-version -s ^HG00096 -i 'INFO/AC>2'", 9),
            ("view --no-version -s HG00101 -i 'FMT/DP>20'", 19),
            ("view --no-version -s HG00101 --no-update -i 'INFO/AC>2'", 9),
        ],
    )
    # fmt: on
    def test_vcf_output(self, tmp_path, fx_chr22_vcz, args, expected_variants):
        bcftools_out, _ = run_bcftools(f"{args} {fx_chr22_vcz.vcf_path}")
        bcftools_out_file = tmp_path / "bcftools_out.vcf"
        bcftools_out_file.write_text(bcftools_out)

        vcztools_out, _ = run_vcztools(f"{args} {fx_chr22_vcz.zip_path}")
        vcztools_out_file = tmp_path / "vcztools_out.vcf"
        vcztools_out_file.write_text(vcztools_out)

        allow_zero = expected_variants == 0
        assert_vcfs_close(
            bcftools_out_file, vcztools_out_file, allow_zero_variants=allow_zero
        )
        variant_lines = [
            line for line in vcztools_out.splitlines() if not line.startswith("#")
        ]
        assert len(variant_lines) == expected_variants

    # fmt: off
    @pytest.mark.parametrize(
        ("args", "expected_lines"),
        [
            ("index -n", 1),
            ("index -s", 1),
            ("query -l", 100),
            (r"query -f '%CHROM %POS %REF %ALT{0}\n' -r 'chr22:10510000-10510200'", 6),
            (r"query -f '%CHROM %POS\n' -i 'INFO/AC>10'", 4),
            (r"query -f '%CHROM %POS\n' -i 'INFO/AC>200'", 0),
        ],
    )
    # fmt: on
    def test_output(self, tmp_path, fx_chr22_vcz, args, expected_lines):
        bcftools_output, _ = run_bcftools(f"{args} {fx_chr22_vcz.vcf_path}")
        vcztools_output, _ = run_vcztools(f"{args} {fx_chr22_vcz.zip_path}")
        assert vcztools_output == bcftools_output
        non_empty_lines = [line for line in vcztools_output.splitlines() if line]
        assert len(non_empty_lines) == expected_lines


class TestCalculatedMissingTags:
    """Validate the exact per-row value of ``N_MISSING`` and
    ``F_MISSING`` against bcftools.

    ``bcftools query -f`` has no ``%N_MISSING`` / ``%F_MISSING`` format
    specifier — they are computed identifiers exposed only inside
    filter expressions. The canonical per-row value is obtained by
    piping the fixture through ``bcftools +fill-tags
    -t F_MISSING,N_MISSING`` (which materialises them as INFO fields)
    and reading the integers/floats out with
    ``bcftools query -f '...%INFO/N_MISSING...%INFO/F_MISSING...'``.

    vcztools cannot output the tags via ``query -f`` either, so its
    per-row value is recovered by bucketing: ``vcztools query
    -i 'N_MISSING == k' -f '%CHROM:%POS\\n'`` for each integer ``k`` the
    canonical table reports. The vcztools bucket for each ``k`` must
    equal the multiset of rows bcftools assigned to ``N_MISSING == k``
    — duplicate-POS rows in ``sample-split-alleles.vcf.gz`` are
    preserved in row order. ``F_MISSING`` is checked the same way using
    a ``±0.5/n_samples`` band around bcftools' six-digit canonical
    fraction (the bucket spacing is exactly ``1/n_samples``).

    The chosen fixtures cover:

    - ``sample.vcf.gz``: 3 diploid samples plus the mixed-ploidy row
      at ``X:10`` (one haploid sample, two diploid). bcftools reports
      ``N_MISSING=0`` there — the haploid ``0`` is not a missing call.
    - ``msprime_diploid.vcf.gz``: 3 diploid samples, every GT called.
    - ``1kg_2020_chrM.vcf.gz``: 3 samples, every GT called.
    - ``chr22.vcf.gz``: 100 samples with a wide spread of N_MISSING
      values (>30 distinct buckets), exercising the multi-bucket path.
    - ``sample-split-alleles.vcf.gz``: duplicate-POS rows must end up
      in the same bucket as in bcftools' output.
    """

    FIXTURES = [
        "sample.vcf.gz",
        "msprime_diploid.vcf.gz",
        "1kg_2020_chrM.vcf.gz",
        "chr22.vcf.gz",
        "sample-split-alleles.vcf.gz",
    ]

    @pytest.mark.parametrize("vcf_file", FIXTURES)
    def test_n_missing(self, fx_all_vcz, vcf_file):
        fx = fx_all_vcz[vcf_file]
        rows = _bcftools_missing_table(fx.vcf_path)
        bcftools_buckets: dict[int, list[str]] = {}
        for key, n_missing, _ in rows:
            bcftools_buckets.setdefault(n_missing, []).append(key)
        for k, expected_keys in bcftools_buckets.items():
            out, _ = run_vcztools(
                f"query -f '%CHROM:%POS\\n' -i 'N_MISSING == {k}' "
                f"{fx.zip_path}"
            )
            actual_keys = [line for line in out.splitlines() if len(line) > 0]
            assert actual_keys == expected_keys, (
                f"N_MISSING == {k}: vcztools selected {actual_keys}, "
                f"bcftools selected {expected_keys}"
            )

    @pytest.mark.parametrize("vcf_file", FIXTURES)
    def test_f_missing(self, fx_all_vcz, vcf_file):
        fx = fx_all_vcz[vcf_file]
        rows = _bcftools_missing_table(fx.vcf_path)
        n_samples = fx.group["sample_id"].shape[0]
        # Adjacent F_MISSING values differ by 1/n_samples, so a band
        # of half-step around bcftools' canonical six-digit fraction
        # is wide enough to absorb the printing rounding but narrow
        # enough to exclude neighbouring fractions.
        eps = 0.5 / n_samples
        bcftools_buckets: dict[float, list[str]] = {}
        for key, _, f_missing in rows:
            bcftools_buckets.setdefault(f_missing, []).append(key)
        for f_canonical, expected_keys in bcftools_buckets.items():
            lower = f_canonical - eps
            upper = f_canonical + eps
            expr = f"F_MISSING > {lower} && F_MISSING < {upper}"
            out, _ = run_vcztools(
                f"query -f '%CHROM:%POS\\n' -i '{expr}' {fx.zip_path}"
            )
            actual_keys = [line for line in out.splitlines() if len(line) > 0]
            assert actual_keys == expected_keys, (
                f"F_MISSING in ({lower}, {upper}) (canonical "
                f"{f_canonical}): vcztools={actual_keys}, "
                f"bcftools={expected_keys}"
            )

    def test_fill_tags_does_not_apply_to_missing(self, fx_sample_vcz, tmp_path):
        # --fill-tags rejects N_MISSING / F_MISSING; bcftools +fill-tags
        # doesn't surface them either. Pin the parse-time rejection so a
        # regression doesn't silently widen the accept-list.
        _, stderr = run_vcztools(
            f"view --fill-tags=N_MISSING {fx_sample_vcz.zip_path}",
            expect_error=True,
        )
        assert "unsupported tag" in stderr.lower()

    def test_mixed_ploidy_x10_not_missing(self, fx_sample_vcz):
        # X:10 in sample.vcf.gz mixes a haploid call ('0') with two
        # diploid calls ('0/1', '0|2'). bcftools reports N_MISSING=0
        # at that row — the haploid encoding (with the unused ploidy
        # slot held as INT_FILL=-2) must not be counted as missing.
        # Pin the behaviour explicitly so a regression that miscounts
        # the fill sentinel fails here with a localised error.
        rows = _bcftools_missing_table(fx_sample_vcz.vcf_path)
        x10 = [row for row in rows if row[0] == "X:10"]
        assert x10 == [("X:10", 0, 0.0)]
        out, _ = run_vcztools(
            f"query -f '%CHROM:%POS\\n' -i 'N_MISSING == 0' "
            f"{fx_sample_vcz.zip_path}"
        )
        assert "X:10" in out.splitlines()


def _bcftools_view_with_fill_tags(vcf_path, fill_tags: str, view_args: str = "") -> str:
    """Run ``bcftools view ... | bcftools +fill-tags -t TAGS`` for the
    fixture and return the resulting VCF text. Used as the oracle for
    :class:`TestFillTagsParity`. ``view_args`` lets a test compose with
    flags like ``-s`` so the +fill-tags rule observes the same
    pre-/post-subset semantics."""
    cmd = (
        f"bcftools view --no-version {view_args} {vcf_path} "
        f"| bcftools +fill-tags -- -t {fill_tags}"
    )
    completed = subprocess.run(
        cmd, capture_output=True, check=True, shell=True
    )
    return completed.stdout.decode("utf-8")


class TestFillTagsParity:
    """Validate ``vcztools view --fill-tags=...`` output matches
    ``bcftools view | bcftools +fill-tags -t ...`` for the supported
    tag-set. One parametrised test method per ``(tag, fixture)`` pair.
    """

    # Diploid fixtures whose stored INFO already has AC/AN/AF/NS, so
    # the recompute path has something to overwrite. ``chr22`` is the
    # widest (100 samples, real-world variability). ``msprime_diploid``
    # is the simplest. ``sample`` exercises mixed-ploidy.
    FIXTURES = [
        "sample.vcf.gz",
        "msprime_diploid.vcf.gz",
        "1kg_2020_chrM.vcf.gz",
        "chr22.vcf.gz",
    ]

    @pytest.mark.parametrize("vcf_file", FIXTURES)
    def test_single_tag(self, tmp_path, fx_all_vcz, vcf_file):
        tag = "NS"
        fx = fx_all_vcz[vcf_file]

        bcftools_text = _bcftools_view_with_fill_tags(fx.vcf_path, tag)
        bcftools_file = tmp_path / "bcftools.vcf"
        bcftools_file.write_text(bcftools_text)

        vcztools_text, _ = run_vcztools(
            f"view --no-version --fill-tags={tag} {fx.zip_path}"
        )
        vcztools_file = tmp_path / "vcztools.vcf"
        vcztools_file.write_text(vcztools_text)

        assert_vcfs_close(bcftools_file, vcztools_file)

    @pytest.mark.parametrize("vcf_file", FIXTURES)
    def test_full_tag_list(self, tmp_path, fx_all_vcz, vcf_file):
        fx = fx_all_vcz[vcf_file]
        tags = "AC,AN,AF,NS"

        bcftools_text = _bcftools_view_with_fill_tags(fx.vcf_path, tags)
        bcftools_file = tmp_path / "bcftools.vcf"
        bcftools_file.write_text(bcftools_text)

        vcztools_text, _ = run_vcztools(
            f"view --no-version --fill-tags={tags} {fx.zip_path}"
        )
        vcztools_file = tmp_path / "vcztools.vcf"
        vcztools_file.write_text(vcztools_text)

        assert_vcfs_close(bcftools_file, vcztools_file)


@functools.cache
def _bcftools_fill_tags_table(vcf_path, view_args: str = "") -> pd.DataFrame:
    """Run ``bcftools view {view_args} {vcf_path} | bcftools +fill-tags
    | bcftools query`` and return a tidy DataFrame with one row per
    (variant, ALT allele): columns ``key, allele, AC, AN, AF, NS``. The
    Number=A AC/AF vectors are exploded over their alleles; an entirely
    missing AF vector (rendered as a scalar ``.``) becomes NaN for every
    allele so it aligns with the per-element MISSING the reader emits."""
    fmt = "%CHROM:%POS\\t%INFO/AC\\t%INFO/AN\\t%INFO/AF\\t%INFO/NS\\n"
    cmd = (
        f"bcftools view --no-version {view_args} {vcf_path} "
        f"| bcftools +fill-tags -- -t AC,AN,AF,NS "
        f"| bcftools query -f '{fmt}' -"
    )
    completed = subprocess.run(cmd, capture_output=True, check=True, shell=True)
    out = completed.stdout.decode("utf-8")
    records = []
    for line in out.splitlines():
        if len(line.strip()) == 0:
            continue
        key, ac_s, an_s, af_s, ns_s = line.split("\t")
        ac = [int(p) for p in ac_s.split(",")]
        if af_s == ".":
            af = [np.nan] * len(ac)
        else:
            af = [np.nan if p == "." else float(p) for p in af_s.split(",")]
        an = int(an_s)
        ns = int(ns_s)
        for allele, (ac_i, af_i) in enumerate(zip(ac, af, strict=True)):
            records.append(_parity_record(key, allele, ac_i, an, af_i, ns))
    return _parity_frame(records)


def _vcztools_fill_tags_table(zip_path, view_args_kwargs: dict) -> pd.DataFrame:
    """Open a VczReader on ``zip_path`` configured to match the bcftools
    view filters in ``view_args_kwargs``, recompute AC/AN/AF/NS, and emit
    the same tidy DataFrame as :func:`_bcftools_fill_tags_table`."""
    fields = [
        "variant_contig",
        "variant_position",
        "variant_AC",
        "variant_AN",
        "variant_AF",
        "variant_NS",
    ]
    missing_bits = constants_module.FLOAT32_MISSING.view(np.int32)
    records = []
    with cli.make_reader(
        str(zip_path), view_semantics=True, **view_args_kwargs
    ) as reader:
        contig_ids = reader.contig_ids
        for variant in reader.variants(fields=fields, force_recompute=True):
            contig = contig_ids[variant["variant_contig"]]
            key = f"{contig}:{variant['variant_position']}"
            ac_row = variant["variant_AC"]
            af_row = np.asarray(variant["variant_AF"], dtype=np.float32)
            af_bits = af_row.view(np.int32)
            an = int(variant["variant_AN"])
            ns = int(variant["variant_NS"])
            allele = 0
            for ac_i, af_i, bits in zip(ac_row, af_row, af_bits):
                if ac_i == constants_module.INT_FILL:
                    continue
                af_val = np.nan if bits == missing_bits else float(af_i)
                records.append(_parity_record(key, allele, int(ac_i), an, af_val, ns))
                allele += 1
    return _parity_frame(records)


def _parity_record(key, allele, ac, an, af, ns):
    return {"key": key, "allele": allele, "AC": ac, "AN": an, "AF": af, "NS": ns}


def _parity_frame(records) -> pd.DataFrame:
    columns = ["key", "allele", "AC", "AN", "AF", "NS"]
    return pd.DataFrame.from_records(records, columns=columns)


class TestFillTagsColumnParity:
    """Compare AC/AN/AF/NS column-wise between bcftools (+fill-tags |
    query) and VczReader with recomputed virtual fields.

    Both paths are reduced to a tidy DataFrame (one row per variant and
    ALT allele) and compared with :func:`pandas.testing.assert_frame_equal`,
    so AC/AN/NS match exactly while AF is compared with a float tolerance.
    Used to broadly cover filter compositions."""

    FILTER_CASES = [
        ("no-filter", "", {}),
        ("single-sample", "-s HG00096", {"samples": "HG00096"}),
        (
            "multi-sample",
            "-s HG00096,HG00097,HG00099",
            {"samples": "HG00096,HG00097,HG00099"},
        ),
        (
            "region",
            "-r chr22:10510000-10520000",
            {"regions": "chr22:10510000-10520000"},
        ),
        ("include-expr", "-i 'AN > 150'", {"include": "AN > 150"}),
        (
            "subset-and-region",
            "-s HG00096 -r chr22:10510000-10520000",
            {"samples": "HG00096", "regions": "chr22:10510000-10520000"},
        ),
    ]

    @pytest.mark.parametrize(
        ("view_args", "reader_kwargs"),
        [(view, kw) for _, view, kw in FILTER_CASES],
        ids=[case_id for case_id, _, _ in FILTER_CASES],
    )
    def test_column_parity(self, fx_chr22_vcz, view_args, reader_kwargs):
        oracle = _bcftools_fill_tags_table(fx_chr22_vcz.vcf_path, view_args=view_args)
        actual = _vcztools_fill_tags_table(
            fx_chr22_vcz.zip_path, view_args_kwargs=reader_kwargs
        )
        pd.testing.assert_frame_equal(oracle, actual, check_exact=False, atol=1e-3)
