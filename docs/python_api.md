(sec-python-api)=
# Python API

## Opening a VCZ dataset

`vcztools.open_zarr` is the recommended entry point for opening a VCZ
dataset. It returns a {class}`zarr.Group` and dispatches to one of four
storage backends; see {ref}`sec-storage-backends` for the full
description of each.

A local directory or `.zip` archive needs no explicit backend:

```python
import vcztools

root = vcztools.open_zarr("sample.vcz")
root_zip = vcztools.open_zarr("sample.vcz.zip")
```

Remote URLs require selecting a backend:

```python
root = vcztools.open_zarr(
    "s3://bucket/sample.vcz", backend_storage="fsspec"
)
```

`storage_options` is forwarded to the underlying backend (fsspec
filesystem, `obstore.store.from_url`, or the chosen Icechunk storage
constructor):

```python
root = vcztools.open_zarr(
    "s3://bucket/sample.vcz",
    backend_storage="fsspec",
    storage_options={"anon": True},
)
```

An already-built {class}`zarr.Group` or {class}`zarr.abc.store.Store`
is accepted and passed through unchanged, which is useful when you've
configured a store yourself:

```python
import zarr

store = zarr.storage.MemoryStore()
# ... populate ``store`` ...
root = vcztools.open_zarr(store)
```

## API reference

```{eval-rst}

.. autofunction:: vcztools.open_zarr

```
