import pathlib
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import zip_longest

import cyvcf2
import numpy as np
from bio2zarr import vcf


@contextmanager
def open_vcf(path) -> Iterator[cyvcf2.VCF]:
    """A context manager for opening a VCF file."""
    vcf = cyvcf2.VCF(path)
    try:
        yield vcf
    finally:
        vcf.close()


def normalise_info_missingness(info_dict, key):
    value = info_dict.get(key, None)
    if isinstance(value, tuple):
        if all(x is None for x in value):
            value = None
    elif isinstance(value, str):
        if all(x == "." for x in value.split(",")):
            value = None
    return value


def _get_header_field_dicts(vcf, header_type):
    def to_dict(header_field):
        d = header_field.info(extra=True)
        del d[b"IDX"]  # remove IDX since we don't care about ordering

        # cyvcf2 duplicates some keys as strings and bytes, so remove the bytes one
        for k in list(d.keys()):
            if isinstance(k, bytes) and k.decode("utf-8") in d:
                del d[k]
        return d

    return {
        field["ID"]: to_dict(field)
        for field in vcf.header_iter()
        if field["HeaderType"] == header_type
    }


def _assert_header_field_dicts_equivalent(field_dicts1, field_dicts2):
    assert len(field_dicts1) == len(field_dicts2)

    for id in field_dicts1.keys():
        assert id in field_dicts2
        field_dict1 = field_dicts1[id]
        field_dict2 = field_dicts2[id]

        assert len(field_dict1) == len(field_dict2)
        # all fields should be the same, except Number="." which can match any value
        for k in field_dict1.keys():
            assert k in field_dict2
            v1 = field_dict1[k]
            v2 = field_dict2[k]
            if k == "Number" and (v1 == "." or v2 == "."):
                continue
            assert v1 == v2, f"Failed in field {id} with key {k}"


def _assert_vcf_headers_equivalent(vcf1, vcf2):
    # Only compare INFO, FORMAT, FILTER, CONTIG fields, ignoring order
    # Other fields are ignored

    info1 = _get_header_field_dicts(vcf1, "INFO")
    info2 = _get_header_field_dicts(vcf2, "INFO")
    _assert_header_field_dicts_equivalent(info1, info2)

    format1 = _get_header_field_dicts(vcf1, "FORMAT")
    format2 = _get_header_field_dicts(vcf2, "FORMAT")
    _assert_header_field_dicts_equivalent(format1, format2)

    filter1 = _get_header_field_dicts(vcf1, "FILTER")
    filter2 = _get_header_field_dicts(vcf2, "FILTER")
    _assert_header_field_dicts_equivalent(filter1, filter2)

    contig1 = _get_header_field_dicts(vcf1, "CONTIG")
    contig2 = _get_header_field_dicts(vcf2, "CONTIG")
    _assert_header_field_dicts_equivalent(contig1, contig2)


def assert_vcfs_close(f1, f2, *, rtol=1e-05, atol=1e-03, allow_zero_variants=False):
    """Like :py:func:`numpy.testing.assert_allclose()`, but for VCF files.

    Raises an `AssertionError` if two VCF files are not equal to one another.
    Float values in QUAL, INFO, or FORMAT fields are compared up to the
    desired tolerance. All other values must match exactly.

    Parameters
    ----------
    f1
        Path to first VCF to compare.
    f2
        Path to second VCF to compare.
    rtol
        Relative tolerance.
    atol
        Absolute tolerance.
    """
    with open_vcf(f1) as vcf1, open_vcf(f2) as vcf2:
        _assert_vcf_headers_equivalent(vcf1, vcf2)
        assert vcf1.samples == vcf2.samples

        count = 0
        for v1, v2 in zip_longest(vcf1, vcf2):
            if v1 is None and v2 is not None:
                raise AssertionError(f"Right contains extra variant: {v2}")
            if v1 is not None and v2 is None:
                raise AssertionError(f"Left contains extra variant: {v1}")

            count += 1

            assert v1.CHROM == v2.CHROM, f"CHROM not equal for variants\n{v1}{v2}"
            assert v1.POS == v2.POS, f"POS not equal for variants\n{v1}{v2}"
            assert v1.ID == v2.ID, f"ID not equal for variants\n{v1}{v2}"
            assert v1.REF == v2.REF, f"REF not equal for variants\n{v1}{v2}"
            assert v1.ALT == v2.ALT, f"ALT not equal for variants\n{v1}{v2}"
            np.testing.assert_allclose(
                np.array(v1.QUAL, dtype=np.float32),
                np.array(v2.QUAL, dtype=np.float32),
                rtol=rtol,
                atol=atol,
                err_msg=f"QUAL not equal for variants\n{v1}{v2}",
            )
            assert set(v1.FILTERS) == set(
                v2.FILTERS
            ), f"FILTER not equal for variants\n{v1}{v2}"

            v1_info = dict(v1.INFO)
            v2_info = dict(v2.INFO)
            all_keys = set(v1_info.keys()) | set(v2_info.keys())
            for k in all_keys:
                val1 = normalise_info_missingness(v1_info, k)
                val2 = normalise_info_missingness(v2_info, k)
                # values are python objects (not np arrays)
                if isinstance(val1, float) or (
                    isinstance(val1, tuple) and any(isinstance(v, float) for v in val1)
                ):
                    np.testing.assert_allclose(
                        np.array(val1, dtype=np.float32),
                        np.array(val2, dtype=np.float32),
                        rtol=rtol,
                        atol=atol,
                        err_msg=f"INFO {k} not equal for variants\n{v1}{v2}",
                    )
                else:
                    assert val1 == val2, f"INFO {k} not equal for variants\n{v1}{v2}"

            # NOTE skipping this because it requires items to be in the same order.
            # assert v1.FORMAT == v2.FORMAT, f"FORMAT not equal for variants\n{v1}{v2}"
            for field in v1.FORMAT:
                if field == "GT":
                    assert (
                        v1.genotypes == v2.genotypes
                    ), f"GT not equal for variants\n{v1}{v2}"
                else:
                    val1 = v1.format(field)
                    val2 = v2.format(field)
                    if val2 is None:
                        # FIXME this is a quick hack to workaround missing support for
                        # dealing with the field missing vs all-elements-in-field
                        # missing issue.
                        # https://github.com/jeromekelleher/vcztools/issues/14
                        assert [str(x) == "." for x in val1]
                    else:
                        if val1.dtype.kind == "f":
                            np.testing.assert_allclose(
                                val1,
                                val2,
                                rtol=rtol,
                                atol=atol,
                                err_msg=f"FORMAT {field} not equal for "
                                f"variants\n{v1}{v2}",
                            )
                        else:
                            np.testing.assert_array_equal(
                                val1,
                                val2,
                                err_msg=f"FORMAT {field} not equal for "
                                f"variants\n{v1}{v2}",
                            )

        if not allow_zero_variants:
            assert count > 0, "No variants in file"


def vcz_path_cache(vcf_path):
    """
    Store converted files in a cache to speed up tests. We're not testing
    vcf2zarr here, so no point in running over and over again.
    """
    cache_path = pathlib.Path("vcz_test_cache")
    if not cache_path.exists():
        cache_path.mkdir()
    cached_vcz_path = (cache_path / vcf_path.name).with_suffix(".vcz")
    if not cached_vcz_path.exists():
        if vcf_path.name.startswith("chr22"):
            vcf.convert(
                [vcf_path],
                cached_vcz_path,
                worker_processes=0,
                variants_chunk_size=10,
                samples_chunk_size=10,
            )
        else:
            vcf.convert(
                [vcf_path], cached_vcz_path, worker_processes=0, local_alleles=False
            )
    return cached_vcz_path
