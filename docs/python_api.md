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

## Streaming PLINK 1 binary output

`vcztools.PlinkStreamingSource` exposes a VCZ store as the byte content
of a PLINK 1 binary fileset (`.bed` / `.bim` / `.fam`) without
materialising it on disk. It is intended for FUSE / range-HTTP
serving / preview pipelines where consumers want random byte access
to the virtual fileset.

`.bim` and `.fam` are computed eagerly at construction (small enough
to keep in memory). `.bed` bytes are produced on demand:

```python
import vcztools

root = vcztools.open_zarr("sample.vcz")
with vcztools.PlinkStreamingSource(root) as src:
    # Eager metadata.
    print(src.bed_size, src.bim_size, src.fam_size)
    print(src.bim_bytes.decode("utf-8"))
    print(src.fam_bytes.decode("utf-8"))

    # Random `.bed` reads.
    head = src.read_bed(offset=0, size=4096)
    tail = src.read_tail(nbytes=4096)

    # Variant-axis selection: slice or sorted ndarray.
    rows = src.read_variants(slice(100, 200))

    # Streaming `.bed` for forward-only consumers.
    with open("sample.bed", "wb") as fh:
        for fragment in src.stream_bed(chunk_size=1 << 20):
            fh.write(fragment)
```

Every `read_variants` / `read_bed` / `stream_bed` call constructs a
fresh underlying reader, so multiple in-flight calls (e.g. concurrent
HTTP range reads) are independent. Output bytes are byte-identical to
what `vcztools.plink.write_plink` would write for the same store.

## API reference

```{eval-rst}

.. autofunction:: vcztools.open_zarr

.. autoclass:: vcztools.PlinkStreamingSource
   :members: read_variants, read_bed, read_tail, stream_bed, close,
             num_variants, num_samples, bytes_per_variant, bed_size,
             bim_size, fam_size, bim_bytes, fam_bytes

```
