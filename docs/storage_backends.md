(sec-storage-backends)=
# Storage backends

Vcztools opens VCZ datasets through one of four Zarr storage backends,
selected via the `--backend-storage` CLI option or the `backend_storage`
keyword argument to {func}`vcztools.open_zarr`. The matching
`--storage-option KEY=VALUE` (CLI, repeatable) or `storage_options`
dict (Python) forwards configuration to the underlying library; the
keys it accepts depend on the backend (and, for fsspec, on the URL
scheme).

| Backend | Selector | Best for |
| --- | --- | --- |
| local (default) | omit `--backend-storage` | Local files and `.zip` archives |
| fsspec | `--backend-storage fsspec` | Broadest cloud coverage via `s3fs`, `gcsfs`, `adlfs`, … |
| obstore | `--backend-storage obstore` | High-throughput object-store reads (Rust [object_store](https://docs.rs/object_store/) under the hood) |
| icechunk | `--backend-storage icechunk` | Versioned Icechunk repositories on local disk, S3, or Azure |

`--zarr-backend-storage` is accepted as a deprecated alias for
`--backend-storage` and emits a `DeprecationWarning`.

## Local (default)

With no `--backend-storage` option, vcztools opens local data only:

- `.zip` paths are opened as a {class}`zarr.storage.ZipStore`.
- Directory paths are opened as a {class}`zarr.storage.LocalStore`.
- Remote URLs (any string containing `://`) and non-empty
  `storage_options` raise — pick an explicit backend for those.

```bash
vcztools view sample.vcz
vcztools view sample.vcz.zip
```

Definitive docs: [Zarr storage](https://zarr.readthedocs.io/en/stable/api/zarr/storage/index.html).

## fsspec

```bash
vcztools view --backend-storage fsspec s3://<bucket>/path/to.vcz
```

Routes through {class}`zarr.storage.FsspecStore` via
`FsspecStore.from_url`. Local paths and `pathlib.Path` inputs are
auto-promoted to `file://` URIs.

`storage_options` is forwarded to the fsspec filesystem constructor
that fsspec selects from the URL scheme. The accepted keys therefore
depend on the protocol — see the per-protocol documentation for the
exhaustive list:

- S3: [s3fs](https://s3fs.readthedocs.io/) (e.g. `key`, `secret`, `endpoint_url`, `anon`).
- GCS: [gcsfs](https://gcsfs.readthedocs.io/) (e.g. `token`, `project`).
- Azure: [adlfs](https://github.com/fsspec/adlfs) (e.g. `account_name`, `account_key`, `sas_token`).
- HTTP(S): [fsspec http](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.implementations.http.HTTPFileSystem) (e.g. `client_kwargs`).

Install fsspec plus the protocol package you need:

```bash
python3 -m pip install fsspec s3fs   # for S3
```

Definitive docs: [fsspec](https://filesystem-spec.readthedocs.io/).

## obstore

```bash
vcztools view --backend-storage obstore s3://<bucket>/path/to.vcz
```

Routes through {class}`zarr.storage.ObjectStore` built from
`obstore.store.from_url`.

`storage_options` is unpacked as keyword arguments to
`obstore.store.from_url`. Common keys:

- `client_options` — request-level options (timeouts, headers, TLS).
- `retry_config` — backoff and jitter for retries.
- Scheme-specific credentials, e.g. `aws_access_key_id`,
  `aws_secret_access_key`, `aws_region` for S3; `azure_storage_account_name`,
  `azure_storage_account_key` for Azure.

Install:

```bash
python3 -m pip install obstore
```

Definitive docs: [obstore](https://developmentseed.org/obstore/) and
the [`store.from_url` API reference](https://developmentseed.org/obstore/latest/api/store/).

## icechunk

```bash
vcztools view --backend-storage icechunk s3://<bucket>/repo
```

Opens an [Icechunk](https://icechunk.io/) repository's `main` branch as
a read-only Zarr session. The URL scheme picks the storage
constructor:

- Local paths → `icechunk.Storage.new_local_filesystem`.
- `s3://…` → `icechunk.s3_storage(..., from_env=True)`.
- `az://…`, `azure://…`, `abfs://…`, `abfss://…`, and Azure
  `https://…blob.core.windows.net` URLs →
  `icechunk.azure_storage(..., from_env=True)`.

`storage_options` is forwarded as keyword arguments to the chosen
constructor (e.g. `region`, `endpoint_url` for `s3_storage`). Local
paths reject non-empty `storage_options`.

Install:

```bash
python3 -m pip install icechunk
```

Definitive docs: [Icechunk](https://icechunk.io/).

## Example: read from S3 with fsspec

Set up credentials (e.g. via environment variables described in the
[s3fs documentation](https://s3fs.readthedocs.io/en/latest/#credentials)):

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

Then run vcztools against an `s3://` URL:

```bash
vcztools view --backend-storage fsspec s3://<bucket>/path/to.vcz
```

Equivalent in Python:

```python
import vcztools

root = vcztools.open_zarr(
    "s3://<bucket>/path/to.vcz", backend_storage="fsspec"
)
```
