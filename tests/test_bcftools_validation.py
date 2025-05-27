import pathlib
import subprocess

import click.testing as ct
import pytest

import vcztools.cli as cli

from .utils import assert_vcfs_close, vcz_path_cache


def run_bcftools(args: str, expect_error=False) -> tuple[str, str]:
    """
    Run bcftools (which must be on the PATH) and return stdout and stderr
    as a pair of strings.
    """
    completed = subprocess.run(
        f"bcftools {args}", capture_output=True, check=False, shell=True
    )
    if expect_error:
        assert completed.returncode != 0
    else:
        assert completed.returncode == 0
    return completed.stdout.decode("utf-8"), completed.stderr.decode("utf-8")


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


# fmt: off
@pytest.mark.parametrize(
    ("args", "vcf_file"),
    [
        ("view --no-version", "sample.vcf.gz"),
        ("view --no-version", "chr22.vcf.gz"),
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
        ("view --no-version -s ^tsk_0,tsk_1,tsk_2", "msprime_diploid.vcf.gz"),
        ("view --no-version -s '' --force-samples", "sample.vcf.gz"),
        ("view --no-version -s 'NO_SAMPLE' --force-samples", "sample.vcf.gz"),
        ("view --no-version -s 'NO_SAMPLE,NA00001' --force-samples", "sample.vcf.gz"),
        ("view --no-version -s ^NA00001", "sample.vcf.gz"),
        ("view --no-version -s ^NA00003,NA00002", "sample.vcf.gz"),
        ("view --no-version -s ^NA00003,NA00002,NA00003", "sample.vcf.gz"),
        ("view --no-version -S ^tests/data/txt/samples.txt", "sample.vcf.gz"),
        (
            "view --no-version -r '20:1230236-' -i 'FMT/DP>3' -s 'NA00002,NA00003'",
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
def test_vcf_output(tmp_path, args, vcf_file):
    # print("args:", args)
    original = pathlib.Path("tests/data/vcf") / vcf_file
    vcz = vcz_path_cache(original)

    bcftools_out, _ = run_bcftools(f"{args} {original}")
    bcftools_out_file = tmp_path.joinpath("bcftools_out.vcf")
    with open(bcftools_out_file, "w") as f:
        f.write(bcftools_out)

    vcztools_out, _ = run_vcztools(f"{args} {vcz}")
    vcztools_out_file = tmp_path.joinpath("vcztools_out.vcf")
    with open(vcztools_out_file, "w") as f:
        f.write(vcztools_out)

    assert_vcfs_close(bcftools_out_file, vcztools_out_file)


@pytest.mark.parametrize(
    ("args", "vcf_file"),
    [("view --no-version", "sample.vcf.gz")],
)
def test_vcf_output_with_output_option(tmp_path, args, vcf_file):
    vcf_path = pathlib.Path("tests/data/vcf") / vcf_file
    vcz_path = vcz_path_cache(vcf_path)

    bcftools_out_file = tmp_path.joinpath("bcftools_out.vcf")
    vcztools_out_file = tmp_path.joinpath("vcztools_out.vcf")

    bcftools_args = f"{args} -o {bcftools_out_file}"
    vcztools_args = f"{args} -o {vcztools_out_file}"

    run_bcftools(f"{bcftools_args} {vcf_path}")
    run_vcztools(f"{vcztools_args} {vcz_path}")

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
        (r"query -f 'A\n'", "sample.vcf.gz"),
        (r"query -f '%CHROM:%POS\n'", "sample.vcf.gz"),
        (r"query -f '[%CHROM %POS %GT\n]'", "sample.vcf.gz"),
        (r"query -f '%INFO/DP\n'", "sample.vcf.gz"),
        (r"query -f '%AC{0}\n'", "sample.vcf.gz"),
        (r"query -f '%REF\t%ALT\n'", "sample.vcf.gz"),
        (r"query -f '%ALT{1}\n'", "sample.vcf.gz"),
        (r"query -f '%ID\n'", "sample.vcf.gz"),
        (r"query -f '%QUAL\n'", "sample.vcf.gz"),
        (r"query -f '%FILTER\n'", "sample.vcf.gz"),
        (r"query --format '%FILTER\n'", "1kg_2020_chrM.vcf.gz"),
        (r"query -f '%POS\n' -i 'POS=112'", "sample.vcf.gz"),
        (r"query -f '%POS\n' -e 'POS=112'", "sample.vcf.gz"),
        (r"query -f '[%CHROM\t]\n'", "sample.vcf.gz"),
        (r"query -f '[%CHROM\t]\n' -i 'POS=112'", "sample.vcf.gz"),
        (r"query -f '[%CHROM:%POS %SAMPLE %GT\n]'", "sample.vcf.gz"),
        (r"query -f '[%SAMPLE %GT %DP\n]'", "sample.vcf.gz"),
        (
            r"query -f '[%POS %SAMPLE %GT %DP %GQ\n]' -i 'INFO/DP >= 5'",
            "sample.vcf.gz",
        ),
        (
            r"query -f '[%POS %QUAL\n]' -i'(QUAL > 10 && POS > 100000)'",
            "sample.vcf.gz",
        ),
        # Examples from bcftools query documentation
        (r"query -f '%CHROM  %POS  %REF  %ALT{0}\n'", "sample.vcf.gz"),
        (r"query -f '%CHROM\t%POS\t%REF\t%ALT[\t%SAMPLE=%GT]\n'", "sample.vcf.gz"),
        (r"query -f 'GQ:[ %GQ] \t GT:[ %GT]\n'", "sample.vcf.gz"),
        # POS0 not supported
        # (r"query -f '%CHROM\t%POS0\t%END\t%ID\n'", "sample.vcf.gz"),
        # Filtering on GT not supported
        # (r"query -f [%CHROM:%POS %SAMPLE %GT\n]' -i'GT=\"alt\"'", "sample.vcf.gz"),
        # Indexing not supported in filtering
        # (r"query  -f '%AC{1}\n' -i 'AC[1]>10' ", "sample.vcf.gz"),
        # TODO fill-out more of these when supported for more stuff is available
        # in filtering
        ("query -f '%CHROM %POS %FILTER\n' -i 'FILTER=\"PASS\"'", "sample.vcf.gz"),
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
    ],
)
def test_output(tmp_path, args, vcf_name):
    vcf_path = pathlib.Path("tests/data/vcf") / vcf_name
    vcz_path = vcz_path_cache(vcf_path)

    bcftools_output, _ = run_bcftools(f"{args} {vcf_path}")
    vcztools_output, _ = run_vcztools(f"{args} {vcz_path}")

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
def test_query_arithmethic(tmp_path, expr):

    args = r"query -f '%POS\n'" + f" -i '{expr}'"
    vcf_name = "sample.vcf.gz"
    vcf_path = pathlib.Path("tests/data/vcf") / vcf_name
    vcz_path = vcz_path_cache(vcf_path)

    bcftools_output, _ = run_bcftools(f"{args} {vcf_path}")
    vcztools_output, _ = run_vcztools(f"{args} {vcz_path}")

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
def test_query_logic_precendence(tmp_path, expr, expected):

    args = r"query -f '%POS\n'" + f" -i 'POS=112 && ({expr})'"
    vcf_name = "sample.vcf.gz"
    vcf_path = pathlib.Path("tests/data/vcf") / vcf_name
    vcz_path = vcz_path_cache(vcf_path)

    bcftools_output, _ = run_bcftools(f"{args} {vcf_path}")
    vcztools_output, _ = run_vcztools(f"{args} {vcz_path}")

    assert vcztools_output == bcftools_output
    num_lines = len(list(vcztools_output.splitlines()))
    assert num_lines == int(expected)


# fmt: off
@pytest.mark.parametrize(
    ("args", "vcf_name", "bcftools_error_string"),
    [
        ("index -ns", "sample.vcf.gz", True),
        ("query -f '%POS\n' -i 'INFO/DP > 10' -e 'INFO/DP < 50'", "sample.vcf.gz", True),  # noqa: E501
        ("view -i 'INFO/DP > 10' -e 'INFO/DP < 50'", "sample.vcf.gz", True),
        # bcftools output does not start with "Error"
        ("view -i 'FILTER=\"F\"'", "sample.vcf.gz", False),
    ],
)
# fmt: on
def test_error(tmp_path, args, vcf_name, bcftools_error_string):
    vcf_path = pathlib.Path("tests/data/vcf") / vcf_name
    vcz_path = vcz_path_cache(vcf_path)

    _, bcftools_error = run_bcftools(f"{args} {vcf_path}", expect_error=True)
    if bcftools_error_string:
        assert bcftools_error.startswith("Error:") or bcftools_error.startswith("[E::")

    _, vcztools_error = run_vcztools(f"{args} {vcz_path}", expect_error=True)
    assert "Error:" in vcztools_error
