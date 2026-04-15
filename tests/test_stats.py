from io import StringIO

import pytest

from vcztools.stats import nrecords, stats

from . import vcz_builder


def test_nrecords(fx_sample_vcz):
    output_str = StringIO()
    nrecords(fx_sample_vcz.group, output_str)
    assert output_str.getvalue() == "9\n"


def test_stats(fx_sample_vcz):
    output_str = StringIO()
    stats(fx_sample_vcz.group, output_str)

    assert (
        output_str.getvalue()
        == """19	.	2
20	.	6
X	.	1
"""
    )


def test_stats__no_index(fx_sample_vcz):
    mutated = vcz_builder.copy_vcz(fx_sample_vcz.group)
    del mutated["region_index"]

    with pytest.raises(ValueError, match="Could not load 'region_index' variable."):
        stats(mutated, StringIO())
