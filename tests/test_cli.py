import os.path
import pathlib

from tests.test_bcftools_validation import run_vcztools
from tests.utils import vcz_path_cache


def test_view_output(tmp_path):
    vcf_path = pathlib.Path("tests/data/vcf/sample.vcf.gz")
    vcz_path = vcz_path_cache(vcf_path)
    output = tmp_path / "output.vcf"

    run_vcztools(f"view {vcz_path} --output {output}")
    assert os.path.exists(output)
