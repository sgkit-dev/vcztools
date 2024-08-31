import pathlib
import re

import pytest

from tests.test_bcftools_validation import run_vcztools
from tests.utils import vcz_path_cache


def test_view_bad_output(tmp_path):
    vcf_path = pathlib.Path("tests/data/vcf/sample.vcf.gz")
    vcz_path = vcz_path_cache(vcf_path)
    bad_output = tmp_path / "output.vcf.gz"

    with pytest.raises(
        ValueError, match=re.escape("Output file extension must be .vcf, got: .gz")
    ):
        run_vcztools(f"view --no-version {vcz_path} -o {bad_output}")
