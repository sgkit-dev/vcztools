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

`vcztools.VczReader` is the reader entry point over an opened VCZ dataset;
its `variant_chunks` and `variants` methods iterate the selected variant data
at chunk and row granularity. Selection and filtering are configured before
iterating with `set_samples`, `set_variants`, and `set_variant_filter` — the
latter takes any `VariantFilter`, such as a `BcftoolsFilter` built from a
bcftools `-i`/`-e` expression. `write_plink` and `write_bgen` are the one-shot
writers that turn a reader into a complete PLINK 1 fileset or BGEN file. For
finer control, `BedEncoder` and `BgenEncoder` expose the `.bed` / `.bgen` byte
streams directly; both share the `FormatEncoder` random-access read/write API.
The `write_bim` / `write_fam` / `write_bgi` / `write_sample` functions emit the
individual sidecar files.

```{eval-rst}

.. autofunction:: vcztools.open_zarr

.. autoclass:: vcztools.VczReader
   :members: sample_ids, contig_ids, num_variants, num_samples,
             field_names, virtual_field_names, set_samples, set_variants,
             set_variant_filter, materialise_variant_filter,
             get_field_info, variant_chunks, variants
   :member-order: bysource

.. autoclass:: vcztools.FieldInfo
   :members:

.. autoclass:: vcztools.VariantFilter
   :members:
   :undoc-members:

.. autoclass:: vcztools.BcftoolsFilter

.. autofunction:: vcztools.write_plink
.. autofunction:: vcztools.write_bgen

.. autoclass:: vcztools.FormatEncoder
   :members:

.. autoclass:: vcztools.BedEncoder
   :members:
   :show-inheritance:

.. autoclass:: vcztools.BgenEncoder
   :members:
   :show-inheritance:

.. autofunction:: vcztools.write_bim
.. autofunction:: vcztools.write_fam
.. autofunction:: vcztools.write_bgi
.. autofunction:: vcztools.write_sample

```
