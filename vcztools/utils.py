from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np

from vcztools.constants import RESERVED_VCF_FIELDS


def search(a, v):
    """Like `np.searchsorted`, but array `a` does not have to be sorted."""
    sorter = np.argsort(a)
    rank = np.searchsorted(a, v, sorter=sorter)
    return sorter[rank]


@contextmanager
def open_file_like(file):
    """A context manager for opening a file path or string (and closing on exit),
    or passing a file-like object through."""
    with ExitStack() as stack:
        if isinstance(file, (str, Path)):
            file = stack.enter_context(open(file, mode="w"))
        yield file


def vcf_name_to_vcz_name(vcz_names: set[str], vcf_name: str) -> str:
    """
    Convert the name of a VCF field to the name of the corresponding VCF Zarr array.

    :param set[str] vcz_names: A set of potential VCF Zarr field names
    :param str vcf_name: The name of the VCF field
    :return: The name of the corresponding VCF Zarr array
    :rtype: str
    """
    split = vcf_name.split("/")
    assert 1 <= len(split) <= 2
    is_genotype_field = split[-1] == "GT"
    is_format_field = (
        split[0] in {"FORMAT", "FMT"}
        if len(split) > 1
        else is_genotype_field or f"call_{split[-1]}" in vcz_names
    )
    is_info_field = split[0] == "INFO" or split[-1] not in RESERVED_VCF_FIELDS

    if is_format_field:
        if is_genotype_field:
            return "call_genotype"
        else:
            return "call_" + split[-1]
    elif is_info_field:
        return f"variant_{split[-1]}"
    else:
        return RESERVED_VCF_FIELDS[vcf_name]
