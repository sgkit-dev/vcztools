(sec-cloud-stores)=
# Cloud stores

Vcztools can read VCZ files from cloud stores using
[fsspec](https://filesystem-spec.readthedocs.io/en/latest/).

For example, to read from Amazon S3, first install the `s3fs` fsspec library:

```bash
python3 -m pip install s3fs
```

Then provide your AWS credentials as described in the
[s3fs documentation](https://s3fs.readthedocs.io/en/latest/#credentials),
for example by setting environment variables:

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

You can then run vcztools using an `s3://` URL:

```bash
vcztools view s3://<bucket-name>/path/to.vcz
```

## Backend storage

By default, vcztools uses `fsspec` for remote storage access. You can also
use [obstore](https://developmentseed.org/obstore/) as an alternative
backend by passing the `--zarr-backend-storage` option:

```bash
vcztools view --zarr-backend-storage obstore s3://<bucket-name>/path/to.vcz
```
