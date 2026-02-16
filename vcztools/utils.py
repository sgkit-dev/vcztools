from contextlib import ExitStack, contextmanager
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import zarr

from vcztools import constants
from vcztools.constants import RESERVED_VCF_FIELDS


def search(a, v):
    """
    Finds the indices into an array a corresponding to the elements in v.
    The behaviour is undefined if any elements in v are not in a.
    """
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


def open_zarr(
    file_or_url: str | Path,
    /,
    *,
    mode: str = "r",
    zarr_backend_storage: str | None = None,
):
    if zarr_backend_storage is None or zarr_backend_storage == "fsspec":
        return zarr.open(file_or_url, mode=mode)
    elif zarr_backend_storage == "obstore":
        import obstore as obs
        from zarr.storage import ObjectStore

        if isinstance(file_or_url, str):
            if "://" not in file_or_url:  # local path
                path = Path(file_or_url).resolve()  # make absolute
                store = ObjectStore(obs.store.from_url(path.as_uri(), mkdir=True))
            else:  # remote path
                store = ObjectStore(obs.store.from_url(file_or_url))
        elif isinstance(file_or_url, Path):
            path = file_or_url.resolve()  # make absolute
            store = ObjectStore(obs.store.from_url(path.as_uri(), mkdir=True))
        return zarr.open(store, mode=mode)
    elif zarr_backend_storage == "icechunk":
        import icechunk as ic

        if isinstance(file_or_url, str):
            if "://" not in file_or_url:  # local path
                storage = ic.Storage.new_local_filesystem(file_or_url)
            elif file_or_url.startswith("s3://"):
                url_parsed = urlparse(file_or_url)
                storage = ic.s3_storage(
                    bucket=url_parsed.netloc,
                    prefix=url_parsed.path.lstrip("/"),
                    from_env=True,
                )
            else:
                raise ValueError(f"Unsupported URL for icechunk: {file_or_url}")
        elif isinstance(file_or_url, Path):
            path = file_or_url.resolve()  # make absolute
            storage = ic.Storage.new_local_filesystem(str(path))

        repo = ic.Repository.open(storage)
        session = repo.readonly_session("main")
        store = session.store
        return zarr.open(store, mode=mode)
    else:
        raise ValueError(
            f"Unsupported Zarr backend storage: {zarr_backend_storage}. "
            "Must be one of 'fsspec', 'obstore', or 'icechunk'"
        )


def vcf_name_to_vcz_names(vcz_names: set[str], vcf_name: str) -> list[str]:
    """
    Convert the name of a VCF field to the names of corresponding VCF Zarr arrays.

    :param set[str] vcz_names: A set of allowed VCF Zarr field names
    :param str vcf_name: The name of the VCF field
    :return: The names of corresponding VCF Zarr arrays, with call (FORMAT) fields
    before variant (INFO) fields, if both are possible matches, or an empty list
    if there are no matches.
    :rtype: list[str]
    """

    candidates = []
    split = vcf_name.split("/")
    assert 1 <= len(split) <= 2

    if split[-1] == "GT":
        candidates.append("call_genotype")
    elif len(split) > 1:
        if split[0] in {"FORMAT", "FMT"}:
            candidates.append(f"call_{split[-1]}")
        elif split[0] in {"INFO"}:
            candidates.append(f"variant_{split[-1]}")
    else:
        candidates.append(f"call_{split[-1]}")
        candidates.append(f"variant_{split[-1]}")

    matches = [candidate for candidate in candidates if candidate in vcz_names]

    if vcf_name in RESERVED_VCF_FIELDS:
        matches.append(RESERVED_VCF_FIELDS[vcf_name])

    return matches


# See https://numpy.org/devdocs/user/basics.strings.html#casting-to-and-from-fixed-width-strings


def _max_len(arr: np.ndarray) -> int:
    lengths = np.strings.str_len(arr)  # numpy 2
    max_len = int(np.max(lengths)) if lengths.size > 0 else 1
    return max(max_len, 1)


def _as_fixed_length_string(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind == "O":
        return arr.astype("S")
    else:
        # convert from StringDType to a fixed-length null-terminated byte sequence
        # (character code S)
        return arr.astype(f"S{_max_len(arr)}")


def _as_fixed_length_unicode(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind == "O":
        return arr.astype("U")
    else:
        # convert from StringDType to a fixed-length unicode string (character code U)
        return arr.astype(f"U{_max_len(arr)}")


def missing(arr: np.ndarray) -> np.ndarray:
    """Return a boolean array indicating which values are missing sentinels."""
    if arr.dtype.kind == "i":
        return arr == constants.INT_MISSING
    elif arr.dtype.kind == "f":
        return arr.view(np.int32) == constants.FLOAT32_MISSING_AS_INT32
    elif arr.dtype.kind in ("O", "U", "T"):
        return arr == constants.STR_MISSING
    elif arr.dtype.kind == "b":
        return ~arr
    else:
        raise ValueError(f"unrecognised dtype: {arr.dtype}")
