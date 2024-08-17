import pathlib
from io import StringIO

from tests.utils import vcz_path_cache
from vcztools.query import list_samples


def test_list_samples(tmp_path):
    vcf_path = pathlib.Path("tests/data/vcf") / "sample.vcf.gz"
    vcz_path = vcz_path_cache(vcf_path)
    expected_output = "NA00001\nNA00002\nNA00003\n"

    with StringIO() as output:
        list_samples(vcz_path, output)
        assert output.getvalue() == expected_output
