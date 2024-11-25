import pathlib
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import zip_longest

import cyvcf2
import numpy as np
from bio2zarr import vcf2zarr


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


def assert_vcfs_close(f1, f2, *, rtol=1e-05, atol=1e-03):
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
        assert vcf1.raw_header == vcf2.raw_header
        assert vcf1.samples == vcf2.samples

        for v1, v2 in zip_longest(vcf1, vcf2):
            if v1 is None and v2 is not None:
                raise AssertionError(f"Right contains extra variant: {v2}")
            if v1 is not None and v2 is None:
                raise AssertionError(f"Left contains extra variant: {v1}")

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
            vcf2zarr.convert(
                [vcf_path],
                cached_vcz_path,
                worker_processes=0,
                variants_chunk_size=10,
                samples_chunk_size=10,
            )
        else:
            vcf2zarr.convert(
                [vcf_path], cached_vcz_path, worker_processes=0, local_alleles=False
            )
    return cached_vcz_path
