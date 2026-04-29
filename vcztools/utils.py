import dataclasses
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import zarr

from vcztools import constants
from vcztools.constants import RESERVED_VCF_FIELDS


@dataclasses.dataclass
class ChunkRead:
    """A single chunk read: which chunk, and which local entries.

    Axis-agnostic: used for both the variants axis (by
    :mod:`vcztools.regions`) and the samples axis (by
    :mod:`vcztools.samples`). ``selection=None`` means "emit the full
    chunk, no slicing". ``selection`` may be a ``slice`` for a
    contiguous range (basic indexing returns a view), or an ndarray of
    local indices for arbitrary fancy-index gather/permutation.
    """

    index: int
    selection: np.ndarray | slice | None = None


def normalise_local_selection(
    local_sel: np.ndarray, chunk_size: int
) -> np.ndarray | slice | None:
    """Collapse a contiguous, sorted, no-duplicate per-chunk selection
    into a basic-indexing form.

    - Full chunk in order → ``None`` (caller should emit the raw block).
    - Contiguous range in order → ``slice(start, stop)`` (basic
      indexing returns a view, avoiding a fancy-index gather).
    - Anything else → the original ndarray (gather as before).

    The check is O(N) per chunk; chunk-plan builders run once per query,
    so the cost is paid once and the saving is per-chunk.
    """
    if local_sel.size == 0:
        return local_sel
    start = int(local_sel[0])
    stop = start + int(local_sel.size)
    # Quick reject: if last element doesn't match a contiguous range,
    # skip the array_equal scan.
    if int(local_sel[-1]) != stop - 1:
        return local_sel
    if not np.array_equal(local_sel, np.arange(start, stop)):
        return local_sel
    if start == 0 and stop == chunk_size:
        return None
    return slice(start, stop)


def array_dims(arr):
    """Return the dimension names for a Zarr array.

    Zarr v2 stores dimension names in ``_ARRAY_DIMENSIONS`` attrs;
    Zarr v3 exposes them via ``metadata.dimension_names``.
    """
    return arr.attrs.get("_ARRAY_DIMENSIONS", None) or arr.metadata.dimension_names


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


def _is_zip_path(file_or_url) -> bool:
    """Whether ``file_or_url`` should be opened as a ``ZipStore``.

    The check is "ends in .zip" applied as a string suffix; remote
    URLs whose final path segment ends in ``.zip`` (e.g.
    ``s3://bucket/foo.vcz.zip``) are not currently supported.
    """
    if not isinstance(file_or_url, (str, Path)):
        return False
    return str(file_or_url).endswith(".zip")


def _is_remote_url(file_or_url) -> bool:
    """Whether ``file_or_url`` is a remote URL (string with ``://``).

    ``file://`` URLs count as remote — Zarr 3 routes them through
    fsspec, not LocalStore.
    """
    return isinstance(file_or_url, str) and "://" in file_or_url


def open_zarr(
    file_or_url: str | Path,
    /,
    *,
    mode: str = "r",
    zarr_backend_storage: str | None = None,
    storage_options: dict | None = None,
):
    """Open a VCZ store under a controllable Zarr backend.

    An already-built :class:`zarr.Group` or :class:`zarr.abc.store.Store`
    is passed through (wrapped in :func:`zarr.open` for the latter), so
    callers that have constructed their own store don't need to know
    which backend produced it.

    Otherwise resolution depends on ``zarr_backend_storage``:

    - ``None`` (default — local-only): ``.zip`` path →
      :class:`zarr.storage.ZipStore`; local path →
      :class:`zarr.storage.LocalStore`. URLs (str with ``://``) and
      non-empty ``storage_options`` raise.
    - ``"fsspec"``: explicit :class:`zarr.storage.FsspecStore` via
      :meth:`~zarr.storage.FsspecStore.from_url`. Local paths are
      auto-promoted to ``file://`` URIs. ``storage_options`` is
      forwarded to fsspec.
    - ``"obstore"``: :class:`zarr.storage.ObjectStore` over
      ``obstore.store.from_url``. ``storage_options`` is unpacked as
      kwargs to ``from_url`` (e.g. ``client_options``, ``retry_config``).
    - ``"icechunk"``: an :mod:`icechunk` storage built by
      :func:`make_icechunk_storage`. ``storage_options`` is forwarded
      to the chosen storage constructor (S3 / Azure); for local
      Icechunk paths non-empty options raise.
    """
    if zarr_backend_storage not in (None, "fsspec", "obstore", "icechunk"):
        raise ValueError(
            f"Unsupported Zarr backend storage: {zarr_backend_storage}. "
            "Must be None (local-only), 'fsspec', 'obstore', or 'icechunk'"
        )

    if isinstance(file_or_url, zarr.Group):
        return file_or_url
    if isinstance(file_or_url, zarr.abc.store.Store):
        return zarr.open(file_or_url, mode=mode)

    storage_options = storage_options or {}

    if zarr_backend_storage is None:
        return _open_zarr_local(file_or_url, mode=mode, storage_options=storage_options)
    if zarr_backend_storage == "fsspec":
        return _open_zarr_fsspec(
            file_or_url, mode=mode, storage_options=storage_options
        )
    if zarr_backend_storage == "obstore":
        return _open_zarr_obstore(
            file_or_url, mode=mode, storage_options=storage_options
        )
    return _open_zarr_icechunk(file_or_url, mode=mode, storage_options=storage_options)


def _open_zarr_local(file_or_url, *, mode, storage_options):
    if storage_options:
        raise ValueError("storage_options not supported for local stores")
    if _is_remote_url(file_or_url):
        raise ValueError(
            f"URL {file_or_url!r} requires zarr_backend_storage='fsspec', "
            "'obstore', or 'icechunk'"
        )
    if _is_zip_path(file_or_url):
        return zarr.open(zarr.storage.ZipStore(file_or_url, mode="r"), mode=mode)
    return zarr.open(zarr.storage.LocalStore(file_or_url), mode=mode)


def _open_zarr_fsspec(file_or_url, *, mode, storage_options):
    if isinstance(file_or_url, Path):
        url = file_or_url.resolve().as_uri()
    elif isinstance(file_or_url, str):
        if _is_remote_url(file_or_url):
            url = file_or_url
        else:
            url = Path(file_or_url).resolve().as_uri()
    else:
        raise TypeError(f"Unsupported file_or_url type: {type(file_or_url)}")
    store = zarr.storage.FsspecStore.from_url(
        url, storage_options=storage_options or None
    )
    return zarr.open(store, mode=mode)


def _open_zarr_obstore(file_or_url, *, mode, storage_options):
    import obstore as obs  # noqa PLC0415
    from zarr.storage import ObjectStore  # noqa PLC0415

    if isinstance(file_or_url, Path):
        url = file_or_url.resolve().as_uri()
        backend = obs.store.from_url(url, mkdir=True, **storage_options)
    elif isinstance(file_or_url, str):
        if _is_remote_url(file_or_url):
            backend = obs.store.from_url(file_or_url, **storage_options)
        else:
            url = Path(file_or_url).resolve().as_uri()
            backend = obs.store.from_url(url, mkdir=True, **storage_options)
    else:
        raise TypeError(f"Unsupported file_or_url type: {type(file_or_url)}")
    return zarr.open(ObjectStore(backend), mode=mode)


def _open_zarr_icechunk(file_or_url, *, mode, storage_options):
    import icechunk as ic  # noqa PLC0415

    storage = make_icechunk_storage(file_or_url, storage_options=storage_options)
    repo = ic.Repository.open(storage)
    session = repo.readonly_session("main")
    return zarr.open(session.store, mode=mode)


def make_icechunk_storage(file_or_url, *, storage_options=None):
    """Convert a file or URL to an Icechunk Storage object.

    ``storage_options`` is forwarded as keyword arguments to the chosen
    :mod:`icechunk` storage constructor (``s3_storage``, ``azure_storage``).
    Local-filesystem storage rejects non-empty ``storage_options``.
    """
    import icechunk as ic  # noqa: PLC0415

    storage_options = storage_options or {}
    if isinstance(file_or_url, str):
        if "://" not in file_or_url:  # local path
            if storage_options:
                raise ValueError(
                    "storage_options not supported for local icechunk storage"
                )
            return ic.Storage.new_local_filesystem(file_or_url)
        elif file_or_url.startswith("s3://"):
            url_parsed = urlparse(file_or_url)
            return ic.s3_storage(
                bucket=url_parsed.netloc,
                prefix=url_parsed.path.lstrip("/"),
                from_env=True,
                **storage_options,
            )
        elif file_or_url.startswith(
            ("az://", "azure://", "abfs://", "abfss://", "https://")
        ):
            return _make_azure_storage(ic, file_or_url, storage_options=storage_options)
        else:
            raise ValueError(f"Unsupported URL for icechunk: {file_or_url}")
    elif isinstance(file_or_url, Path):
        if storage_options:
            raise ValueError("storage_options not supported for local icechunk storage")
        path = file_or_url.resolve()  # make absolute
        return ic.Storage.new_local_filesystem(str(path))
    else:
        raise TypeError(f"Unsupported URL type for icechunk: {type(file_or_url)}")


def _split_azure_container_path(path, *, file_or_url):
    path_parts = [part for part in path.split("/") if part]
    if len(path_parts) == 0:
        raise ValueError(
            f"Azure Icechunk URLs must include a container name: {file_or_url}"
        )
    return path_parts[0], "/".join(path_parts[1:])


def _make_azure_storage(ic, file_or_url, *, storage_options=None):
    storage_options = storage_options or {}
    parsed = urlparse(file_or_url)

    if parsed.scheme in ("az", "azure"):
        account = parsed.netloc
        if account == "":
            raise ValueError(
                "Azure Icechunk URLs must use the form "
                "'az://<account>/<container>/<prefix>': "
                f"{file_or_url}"
            )
        container, prefix = _split_azure_container_path(
            parsed.path, file_or_url=file_or_url
        )
    elif parsed.scheme in ("abfs", "abfss"):
        if "@" not in parsed.netloc:
            raise ValueError(
                "ABFS Icechunk URLs must use the form "
                "'abfs://<container>@<account>.dfs.core.windows.net/<prefix>': "
                f"{file_or_url}"
            )
        container, account_host = parsed.netloc.split("@", 1)
        account = account_host.removesuffix(".dfs.core.windows.net")
        prefix = parsed.path.lstrip("/")
    elif parsed.scheme == "https" and parsed.netloc.endswith(
        (".blob.core.windows.net", ".dfs.core.windows.net")
    ):
        account = parsed.netloc.removesuffix(".blob.core.windows.net").removesuffix(
            ".dfs.core.windows.net"
        )
        container, prefix = _split_azure_container_path(
            parsed.path, file_or_url=file_or_url
        )
    else:
        raise ValueError(f"Unsupported Azure URL for icechunk: {file_or_url}")

    return ic.azure_storage(
        account=account,
        container=container,
        prefix=prefix,
        from_env=True,
        **storage_options,
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


def array_memory_bytes(arr: np.ndarray) -> int:
    """Best-effort in-memory footprint of ``arr``, in bytes.

    For fixed-size dtypes ``arr.nbytes`` is exact. For variable-length
    string dtypes it only covers the per-element metadata cells; the
    string content itself lives in Python heap (``object`` dtype) or
    numpy's ``StringDType`` arena, both of which we have to walk.

    - kind ``"O"`` (numpy ``object``): per-element ``sys.getsizeof``
      captures each Python str's header + content.
    - kind ``"T"`` (numpy ``StringDType``): ``arr.nbytes`` for the
      metadata cells, plus the UTF-8 byte length of every element.
    - everything else: ``arr.nbytes`` is exact.
    """
    if arr.dtype.kind == "O":
        return sum(sys.getsizeof(x) for x in arr.flat)
    if arr.dtype.kind == "T":
        content = sum(len(s.encode("utf-8")) for s in arr.flat)
        return int(arr.nbytes) + content
    return int(arr.nbytes)
