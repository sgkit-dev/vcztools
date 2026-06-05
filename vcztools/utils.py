import concurrent.futures as cf
import dataclasses
import functools
import importlib
import math
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import zarr

from vcztools import constants
from vcztools.constants import RESERVED_VCF_FIELDS


class PrefetchIterator:
    """One-deep prefetch wrapper around an iterator.

    On every ``__next__`` returns the previously prefetched item and
    submits the next ``__next__`` call on the underlying iterator to
    a dedicated single-worker pool. While the consumer's per-item
    work runs, the producer's next item is being computed in the
    background. Exceptions raised by the underlying iterator surface
    on the consumer's ``__next__`` call.

    Lifetime: the worker pool is created in ``__init__`` and shut
    down by ``close()`` (also called from ``__del__`` defensively to
    prevent thread leaks if a caller forgets to close).
    """

    _SENTINEL = object()

    def __init__(self, source):
        self._source = source
        self._executor = cf.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="vcztools-prefetch"
        )
        self._next_future = self._executor.submit(self._fetch)
        self._closed = False

    def _fetch(self):
        try:
            return next(self._source)
        except StopIteration:
            return self._SENTINEL

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        result = self._next_future.result()
        if result is self._SENTINEL:
            self._closed = True
            self._executor.shutdown(wait=False)
            raise StopIteration
        self._next_future = self._executor.submit(self._fetch)
        return result

    def close(self):
        if self._closed:
            return
        self._closed = True
        # Drain the in-flight fetch so the worker isn't left producing
        # into the void; the result (next item, sentinel, or
        # exception) is no longer needed.
        try:
            self._next_future.result()
        except BaseException:
            pass
        # Plain iterators (e.g. list_iterator) have no close(); only
        # generators and similar resource-holding iterators do.
        source_close = getattr(self._source, "close", None)
        if source_close is not None:
            source_close()
        self._executor.shutdown(wait=True)

    def __del__(self):
        # Defensive: prevent thread leaks if a caller forgets close().
        # Mirrors generator finalisation semantics.
        try:
            self.close()
        except Exception:
            pass


@dataclasses.dataclass
class ChunkRead:
    """A single chunk read: which chunk, and which local entries.

    Axis-agnostic: used for both the variants axis (by
    :mod:`vcztools.regions`) and the samples axis (by
    :mod:`vcztools.retrieval`). ``selection=None`` means "emit the full
    chunk, no slicing". ``selection`` may be a ``slice`` for a
    contiguous range (basic indexing returns a view), or an ndarray of
    local indices for arbitrary fancy-index gather/permutation.

    On the variants axis, ``index`` is in units of *minimum* chunk
    size (see :func:`compute_min_variants_chunk_size`) for the
    canonical plan held on :class:`vcztools.retrieval.VczReader`.
    When every variant-axis field shares one chunk size — the
    historical case — that minimum equals every field's chunk size,
    so the index is the Zarr block index for every variant-axis
    field. When variant-only fields have larger chunks, the per-query
    stream plan (see :func:`rebucket_to_stream_plan`) re-expresses
    selections in stream-chunk units and the retrieval pipeline
    translates them to per-field block indexes.

    ``num_selected`` is the number of entries this chunk contributes
    to its plan — equal to ``len(selection)`` for the ndarray form,
    ``stop - start`` for ``slice``, and the chunk's actual size for
    ``selection=None`` (which is ``chunk_size`` for non-last chunks
    and the remainder for the partial last chunk). Stored explicitly
    so consumers don't have to re-derive it from selection +
    chunk_size + axis_size.
    """

    index: int
    num_selected: int
    selection: np.ndarray | slice | None = None

    @classmethod
    def simple_plan(cls, length: int, chunk_size: int) -> list["ChunkRead"]:
        """Build the trivial all-chunks plan covering ``length`` entries
        with ``chunk_size`` per chunk.

        Returns ``ceil(length / chunk_size)`` ChunkReads with
        ``selection=None`` (full chunk, no slicing). ``num_selected``
        is ``chunk_size`` for every full chunk and the remainder for
        the partial last chunk when ``length`` is not a multiple of
        ``chunk_size``.
        """
        num_chunks = (length + chunk_size - 1) // chunk_size
        return [
            cls(
                index=i,
                num_selected=min(chunk_size, length - i * chunk_size),
            )
            for i in range(num_chunks)
        ]


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


def has_variants_axis(arr) -> bool:
    """Whether ``arr``'s first dimension is the variants axis."""
    dims = array_dims(arr)
    return dims is not None and len(dims) > 0 and dims[0] == "variants"


def compute_min_variants_chunk_size(root) -> int:
    """Compute the minimum variants-axis chunk size in a VCZ root.

    By spec ``call_*`` fields define the floor; every variant-only
    field must use a chunk size that is a positive integer multiple of
    it. Two ``call_*`` fields with different chunk sizes are a writer
    bug and raise ``ValueError`` here. When no ``call_*`` field is
    present, falls back to the GCD of variant-only chunk sizes — the
    largest value that divides every chunk, so the writer-side
    chunking invariant (every chunks[0] is a positive multiple of the
    result) still holds.
    """
    call_sizes: dict[str, int] = {}
    other_sizes: list[int] = []
    for name in root.array_keys():
        arr = root[name]
        if not has_variants_axis(arr):
            continue
        chunk_size = int(arr.chunks[0])
        if name.startswith("call_"):
            call_sizes[name] = chunk_size
        else:
            other_sizes.append(chunk_size)
    if len(call_sizes) > 0:
        sizes_set = set(call_sizes.values())
        if len(sizes_set) > 1:
            raise ValueError(
                f"call_* fields must share a single variants chunk size; "
                f"found {call_sizes}"
            )
        return next(iter(sizes_set))
    if len(other_sizes) > 0:
        return functools.reduce(math.gcd, other_sizes)
    raise ValueError("no variant-axis fields in store")


def validate_variants_axis_chunking(root, min_chunk: int) -> None:
    """Assert every variant-axis field's chunks[0] is a positive
    integer multiple of ``min_chunk``. Raises ``ValueError`` otherwise.
    """
    if min_chunk <= 0:
        raise ValueError(f"min_chunk must be positive (got {min_chunk})")
    for name in root.array_keys():
        arr = root[name]
        if not has_variants_axis(arr):
            continue
        chunk_size = int(arr.chunks[0])
        if chunk_size <= 0 or chunk_size % min_chunk != 0:
            raise ValueError(
                f"{name}.chunks[0]={chunk_size} is not a positive multiple "
                f"of min variants chunk size {min_chunk}"
            )


def compute_stream_chunk_size(root, read_fields, min_chunk: int) -> int:
    """Stream-chunk size for a per-query iteration: GCD of
    ``chunks[0]`` across ``read_fields``, falling back to ``min_chunk``
    when ``read_fields`` is empty.

    Using the GCD (rather than the minimum) ensures every field in
    ``read_fields`` has a chunk size that is a positive integer
    multiple of this stream chunk size — so each (stream chunk, field)
    pair maps to exactly one Zarr block on the variants axis, possibly
    with an intra-block slice. Taking the minimum is only correct when
    the read fields' chunk sizes are pairwise multiples; with
    proportional / heterogeneous variant-only chunking they need not be.

    By the variants-axis chunking invariant
    (:func:`validate_variants_axis_chunking`) every variant-axis chunk
    size is a positive multiple of ``min_chunk``, so ``min_chunk``
    divides the GCD and the result is always a positive integer
    multiple of ``min_chunk`` — :func:`rebucket_to_stream_plan`'s
    precondition.
    """
    if len(read_fields) == 0:
        return min_chunk
    sizes = [int(root[f].chunks[0]) for f in read_fields]
    return functools.reduce(math.gcd, sizes)


def rebucket_to_stream_plan(
    canonical_plan: list["ChunkRead"],
    min_chunk: int,
    stream_chunk_size: int,
) -> list["ChunkRead"]:
    """Rebucket a min-chunk-indexed plan into stream-chunk units.

    Merges consecutive :class:`ChunkRead` entries that fall in the
    same stream chunk; per-chunk selections are rebased into
    stream-chunk-local coordinates and run through
    :func:`normalise_local_selection` to collapse contiguous ranges
    back to ``slice`` or ``None``. Returns the input list unchanged
    when ``stream_chunk_size == min_chunk``.

    ``stream_chunk_size`` must be a positive integer multiple of
    ``min_chunk`` (the variants-axis chunking invariant).
    """
    if stream_chunk_size <= 0 or stream_chunk_size % min_chunk != 0:
        raise ValueError(
            f"stream_chunk_size={stream_chunk_size} must be a positive "
            f"multiple of min_chunk={min_chunk}"
        )
    if stream_chunk_size == min_chunk:
        return canonical_plan

    multiplier = stream_chunk_size // min_chunk
    out: list[ChunkRead] = []
    i = 0
    n = len(canonical_plan)
    while i < n:
        stream_idx = int(canonical_plan[i].index) // multiplier
        j = i
        while j < n and int(canonical_plan[j].index) // multiplier == stream_idx:
            j += 1
        group = canonical_plan[i:j]
        i = j
        out.append(
            _merge_canonical_group(
                group, stream_idx, multiplier, min_chunk, stream_chunk_size
            )
        )
    return out


def _merge_canonical_group(
    group: list["ChunkRead"],
    stream_idx: int,
    multiplier: int,
    min_chunk: int,
    stream_chunk_size: int,
) -> "ChunkRead":
    """Merge canonical entries that fall in one stream chunk."""
    base = stream_idx * multiplier
    # Fast path: every entry is a full min-chunk and the indices are
    # contiguous from ``base``. The merged selection is exactly
    # ``[0, num_selected)`` of the stream chunk, so we can skip the
    # ndarray construction and emit ``None`` / ``slice(0, k)`` directly.
    if all(e.selection is None for e in group) and int(group[0].index) == base:
        contiguous = True
        for k, entry in enumerate(group):
            if int(entry.index) != base + k:
                contiguous = False
                break
        if contiguous:
            num_selected = sum(e.num_selected for e in group)
            selection = (
                None if num_selected == stream_chunk_size else slice(0, num_selected)
            )
            return ChunkRead(
                index=stream_idx, num_selected=num_selected, selection=selection
            )

    parts = []
    num_selected = 0
    for entry in group:
        intra_offset = (int(entry.index) % multiplier) * min_chunk
        sel = entry.selection
        if sel is None:
            parts.append(
                np.arange(
                    intra_offset,
                    intra_offset + entry.num_selected,
                    dtype=np.int64,
                )
            )
        elif isinstance(sel, slice):
            start = sel.start if sel.start is not None else 0
            stop = sel.stop if sel.stop is not None else min_chunk
            parts.append(
                np.arange(intra_offset + start, intra_offset + stop, dtype=np.int64)
            )
        else:
            parts.append(np.asarray(sel, dtype=np.int64) + intra_offset)
        num_selected += entry.num_selected

    merged = parts[0] if len(parts) == 1 else np.concatenate(parts)
    return ChunkRead(
        index=stream_idx,
        num_selected=num_selected,
        selection=normalise_local_selection(merged, stream_chunk_size),
    )


def search(a, v):
    """
    Finds the indices into an array a corresponding to the elements in v.
    The behaviour is undefined if any elements in v are not in a.
    """
    sorter = np.argsort(a)
    rank = np.searchsorted(a, v, sorter=sorter)
    return sorter[rank]


@contextmanager
def open_file_like(file, mode="w"):
    """A context manager for opening a file path or string (and closing on exit),
    or passing a file-like object through. ``mode`` is forwarded to
    :func:`open` for path inputs and ignored for already-open streams.
    """
    with ExitStack() as stack:
        if isinstance(file, (str, Path)):
            file = stack.enter_context(open(file, mode=mode))
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
    zarr_format: int | None = None,
    backend_storage: str | None = None,
    storage_options: dict | None = None,
):
    """Open a Zarr store with configurable backends.

    An already-built :class:`zarr.Group` or :class:`zarr.abc.store.Store`
    is passed through (wrapped in :func:`zarr.open` for the latter).

    Otherwise resolution depends on ``backend_storage``:

    - ``None`` (default — local-only): ``.zip`` path →
      :class:`zarr.storage.ZipStore`; local path →
      :class:`zarr.storage.LocalStore`. URLs (str with ``://``) and
      non-empty ``storage_options`` raise.
    - ``"fsspec"``: explicit :class:`zarr.storage.FsspecStore` via its
      ``from_url`` classmethod. Local paths become ``file://`` URIs.
      ``storage_options`` is forwarded to fsspec.
    - ``"obstore"``: :class:`zarr.storage.ObjectStore` over
      ``obstore.store.from_url``. ``storage_options`` is unpacked as
      kwargs to ``from_url`` (e.g. ``client_options``, ``retry_config``).
    - ``"icechunk"``: an Icechunk storage built by
      ``make_icechunk_storage``. ``storage_options`` is forwarded
      to the chosen storage constructor (S3 / Azure); for local
      Icechunk paths non-empty options raise.
    """
    if backend_storage not in (None, "fsspec", "obstore", "icechunk"):
        raise ValueError(
            f"Unsupported Zarr backend storage: {backend_storage}. "
            "Must be None (local-only), 'fsspec', 'obstore', or 'icechunk'"
        )

    if isinstance(file_or_url, zarr.Group):
        return file_or_url
    if isinstance(file_or_url, zarr.abc.store.Store):
        return zarr.open(file_or_url, mode=mode, zarr_format=zarr_format)

    if backend_storage is None:
        return _open_zarr_local(
            file_or_url,
            mode=mode,
            zarr_format=zarr_format,
            storage_options=storage_options,
        )
    if backend_storage == "fsspec":
        return _open_zarr_fsspec(
            file_or_url,
            mode=mode,
            zarr_format=zarr_format,
            storage_options=storage_options,
        )
    if backend_storage == "obstore":
        return _open_zarr_obstore(
            file_or_url,
            mode=mode,
            zarr_format=zarr_format,
            storage_options=storage_options,
        )
    return _open_zarr_icechunk(
        file_or_url, mode=mode, zarr_format=zarr_format, storage_options=storage_options
    )


def _import_optional(module_name, *, extra):
    """Import a module gated behind a vcztools install extra.

    On ImportError, re-raise with a message pointing at the extra so the
    user can install the missing backend with one ``pip`` invocation.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"{module_name} is required for backend_storage={extra!r}; "
            f"install with `pip install vcztools[{extra}]`"
        ) from e


def _open_zarr_local(file_or_url, *, mode, zarr_format, storage_options):
    if storage_options is not None and len(storage_options) > 0:
        raise ValueError("storage_options not supported for local stores")
    if _is_remote_url(file_or_url):
        raise ValueError(
            f"URL {file_or_url!r} requires backend_storage='fsspec', "
            "'obstore', or 'icechunk'"
        )
    if _is_zip_path(file_or_url):
        return zarr.open(
            zarr.storage.ZipStore(file_or_url, mode="r"),
            mode=mode,
            zarr_format=zarr_format,
        )
    return zarr.open(
        zarr.storage.LocalStore(file_or_url), mode=mode, zarr_format=zarr_format
    )


def _open_zarr_fsspec(file_or_url, *, mode, zarr_format, storage_options):
    if isinstance(file_or_url, Path):
        url = file_or_url.resolve().as_uri()
    elif isinstance(file_or_url, str):
        if _is_remote_url(file_or_url):
            url = file_or_url
        else:
            url = Path(file_or_url).resolve().as_uri()
    else:
        raise TypeError(f"Unsupported file_or_url type: {type(file_or_url)}")
    store = zarr.storage.FsspecStore.from_url(url, storage_options=storage_options)
    return zarr.open(store, mode=mode, zarr_format=zarr_format)


def _open_zarr_obstore(file_or_url, *, mode, zarr_format, storage_options):
    obs = _import_optional("obstore", extra="obstore")
    kwargs = storage_options if storage_options is not None else {}
    if isinstance(file_or_url, Path):
        url = file_or_url.resolve().as_uri()
        backend = obs.store.from_url(url, mkdir=True, **kwargs)
    elif isinstance(file_or_url, str):
        if _is_remote_url(file_or_url):
            backend = obs.store.from_url(file_or_url, **kwargs)
        else:
            url = Path(file_or_url).resolve().as_uri()
            backend = obs.store.from_url(url, mkdir=True, **kwargs)
    else:
        raise TypeError(f"Unsupported file_or_url type: {type(file_or_url)}")
    return zarr.open(
        zarr.storage.ObjectStore(backend), mode=mode, zarr_format=zarr_format
    )


def _open_zarr_icechunk(file_or_url, *, mode, zarr_format, storage_options):
    ic = _import_optional("icechunk", extra="icechunk")
    storage = make_icechunk_storage(file_or_url, storage_options=storage_options)
    repo = ic.Repository.open(storage)
    session = repo.readonly_session("main")
    return zarr.open(session.store, mode=mode, zarr_format=zarr_format)


def make_icechunk_storage(file_or_url, *, storage_options=None):
    """Convert a file or URL to an Icechunk Storage object.

    ``storage_options`` is forwarded as keyword arguments to the chosen
    :mod:`icechunk` storage constructor (``s3_storage``, ``azure_storage``).
    Local-filesystem storage rejects non-empty ``storage_options``.
    """
    ic = _import_optional("icechunk", extra="icechunk")
    kwargs = storage_options if storage_options is not None else {}
    if isinstance(file_or_url, str):
        if "://" not in file_or_url:  # local path
            if storage_options is not None and len(storage_options) > 0:
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
                **kwargs,
            )
        elif file_or_url.startswith(
            ("az://", "azure://", "abfs://", "abfss://", "https://")
        ):
            return _make_azure_storage(ic, file_or_url, storage_options=storage_options)
        else:
            raise ValueError(f"Unsupported URL for icechunk: {file_or_url}")
    elif isinstance(file_or_url, Path):
        if storage_options is not None and len(storage_options) > 0:
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
    kwargs = storage_options if storage_options is not None else {}
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
        **kwargs,
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


# (signed int view dtype, missing-as-int) per supported float width.
_FLOAT_MISSING = {
    np.dtype(np.float16): (np.int16, constants.FLOAT16_MISSING.view(np.int16)),
    np.dtype(np.float32): (np.int32, constants.FLOAT32_MISSING_AS_INT32),
    np.dtype(np.float64): (np.int64, constants.FLOAT64_MISSING.view(np.int64)),
}


# (signed int view dtype, fill-as-int) per supported float width.
_FLOAT_FILL = {
    np.dtype(np.float16): (np.int16, constants.FLOAT16_FILL.view(np.int16)),
    np.dtype(np.float32): (np.int32, constants.FLOAT32_FILL_AS_INT32),
    np.dtype(np.float64): (np.int64, constants.FLOAT64_FILL.view(np.int64)),
}


# (signed int view dtype, missing-as-int, fill-as-int) per supported float width.
_FLOAT_SENTINELS = {
    np.dtype(np.float16): (
        np.int16,
        constants.FLOAT16_MISSING.view(np.int16),
        constants.FLOAT16_FILL.view(np.int16),
    ),
    np.dtype(np.float64): (
        np.int64,
        constants.FLOAT64_MISSING.view(np.int64),
        constants.FLOAT64_FILL.view(np.int64),
    ),
}


def to_vcf_float32(arr: np.ndarray) -> np.ndarray:
    """Return ``arr`` as canonical VCF float32.

    float32 input is returned unchanged. float16/float64 input is cast to
    float32 with the width-generalised missing / end-of-vector sentinels
    relabelled to the float32 sentinels, so downstream code sees a single
    canonical representation.
    """
    if arr.dtype == np.float32:
        return arr
    int_dtype, missing_int, fill_int = _FLOAT_SENTINELS[arr.dtype]
    src = np.ascontiguousarray(arr)
    int_view = src.view(int_dtype)
    missing_mask = int_view == missing_int
    fill_mask = int_view == fill_int
    # The sentinels are signalling NaNs; casting them raises the FP "invalid"
    # flag. The cast values at those positions are discarded below, so ignore it.
    with np.errstate(invalid="ignore"):
        out = src.astype(np.float32)
    out[missing_mask] = constants.FLOAT32_MISSING
    out[fill_mask] = constants.FLOAT32_FILL
    return out


def to_vcf_int32(arr: np.ndarray, name: str) -> np.ndarray:
    """Return ``arr`` as int32 for the VCF encoder, which only supports 1-, 2-
    and 4-byte integers.

    Narrower or equal widths are cast straight to int32. Wider widths (int64)
    are range-checked first: if any value falls outside the int32 range a
    ``ValueError`` naming ``name`` is raised. The missing / end-of-vector
    sentinels (-1 / -2) are within range and cast unchanged.
    """
    if arr.dtype.itemsize > 4 and arr.size > 0:
        info = np.iinfo(np.int32)
        low = int(arr.min())
        high = int(arr.max())
        if low < info.min or high > info.max:
            raise ValueError(
                f"{name}: integer values [{low}, {high}] fall outside the "
                f"32-bit range supported by VCF output"
            )
    return arr.astype(np.int32, copy=False)


def is_missing(arr: np.ndarray) -> np.ndarray:
    """Return a boolean array indicating which values are missing sentinels."""
    if arr.dtype.kind == "i":
        return arr == constants.INT_MISSING
    elif arr.dtype.kind == "f":
        int_dtype, missing_int = _FLOAT_MISSING[arr.dtype]
        return arr.view(int_dtype) == missing_int
    elif arr.dtype.kind in ("O", "U", "T"):
        return arr == constants.STR_MISSING
    elif arr.dtype.kind == "b":
        return ~arr
    else:
        raise ValueError(f"unrecognised dtype: {arr.dtype}")


def is_fill(arr: np.ndarray) -> np.ndarray:
    """Return a boolean array indicating which values are end-of-vector
    (fill) sentinels. Flag (boolean) fields have no fill, so the mask is
    all-False for them."""
    if arr.dtype.kind == "i":
        return arr == constants.INT_FILL
    elif arr.dtype.kind == "f":
        int_dtype, fill_int = _FLOAT_FILL[arr.dtype]
        return arr.view(int_dtype) == fill_int
    elif arr.dtype.kind in ("O", "U", "T"):
        return arr == constants.STR_FILL
    elif arr.dtype.kind == "b":
        return np.zeros(arr.shape, dtype=bool)
    else:
        raise ValueError(f"unrecognised dtype: {arr.dtype}")


def trim_fill(arr: np.ndarray) -> np.ndarray:
    """Drop trailing fill from a 1-D array.

    Returns a view up to and including the last non-fill element, or an
    empty slice if every element is fill. Only trailing fill is trimmed;
    interior fill (a malformed vector) is left in place.
    """
    fill_mask = is_fill(arr)
    non_fill = np.nonzero(~fill_mask)[0]
    if non_fill.size == 0:
        return arr[:0]
    last = int(non_fill[-1]) + 1
    return arr[:last]


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
