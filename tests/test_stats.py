import pathlib
from io import StringIO

from vcztools.stats import nrecords

from .utils import vcz_path_cache


def test_nrecords():
    original = pathlib.Path("tests/data/vcf") / "sample.vcf.gz"
    vcz = vcz_path_cache(original)

    output_str = StringIO()
    nrecords(vcz, output_str)
    assert output_str.getvalue() == "9\n"
