import functools
import subprocess
import sys

import click.testing as ct
import pytest

import vcztools.cli as cli

from .utils import assert_vcfs_close

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="Not supported on Windows"
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
        # Complement collapses duplicate excludes
        ("view --no-version -s ^NA00001,NA00001", "sample.vcf.gz"),
        # Complement that excludes every sample (no sample columns)
        ("view --no-version -s ^NA00001,NA00002,NA00003", "sample.vcf.gz"),
        # All requested samples are unknown -> no sample columns
        ("view --no-version -s FOO,BAR,BAZ --force-samples", "sample.vcf.gz"),
        # Complement with one unknown name -> the unknown is dropped
        ("view --no-version -s ^NA00001,NOPE --force-samples", "sample.vcf.gz"),
        # Complement with all-unknown excludes -> every sample retained
        ("view --no-version -s ^NOPE1,NOPE2 --force-samples", "sample.vcf.gz"),
        # Empty samples file -> no sample columns
        ("view --no-version -S tests/data/txt/samples_empty.txt", "sample.vcf.gz"),
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


class TestSampleSubsetFilterSemantics:
    """Filter expressions are evaluated on the original record, before
    ``-s``/``-S`` sample subsetting is applied. Each case below is
    chosen so pre-subset and post-subset evaluation give different
    results — a regression in ordering would fail the assertion.

    Uses ``query -f '%CHROM %POS\\n'`` to compare only the selected
    variants without noise from FORMAT fields or VCF headers. Note
    that for INFO-scoped filters ``bcftools query`` matches
    ``bcftools view``, but for FMT-scoped filters ``query`` applies
    subsetting first — so ``-i 'FMT/DP>7'`` is exercised with a
    selection where that distinction does not alter the outcome.
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
            "-s NA00001 -i 'INFO/AN>0 && FMT/DP>5'",
        ],
    )
    # fmt: on
    def test_variant_selection(self, fx_sample_vcz, args):
        cmd = f"{self.FMT} {args}"
        bcftools_output, _ = run_bcftools(f"{cmd} {fx_sample_vcz.vcf_path}")
        vcztools_output, _ = run_vcztools(f"{cmd} {fx_sample_vcz.zip_path}")
        assert vcztools_output == bcftools_output


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
