(sec-installation)=
# Installation

`vcztools` is available on PyPI:

```bash
python3 -m pip install vcztools
```

This will install the `vcztools` program into your local Python path.

Alternatively, calling:
```bash
python3 -m vcztools <args>
```
is equivalent to:
```bash
vcztools <args>
```
and will always work.

## Optional dependencies

Reading from some remote storage backends requires extra packages. The
`obstore` and `icechunk` backends are install extras:

```bash
python3 -m pip install vcztools[obstore]
python3 -m pip install vcztools[icechunk]
```

The `fsspec` backend ships with the default install, but the driver for each
cloud protocol is a separate package which must be installed separately:

```bash
python3 -m pip install s3fs    # S3
python3 -m pip install gcsfs   # Google Cloud Storage
python3 -m pip install adlfs   # Azure
```

See {ref}`sec-storage-backends` for which backend and package to use for a
given store.

## Shell completion

To enable shell completion for a particular session in Bash do:

```bash
eval "$(_VCZTOOLS_COMPLETE=bash_source vcztools)"
```

If you add this to your `.bashrc` shell completion should be available
in all new shell sessions.

See the [Click documentation](https://click.palletsprojects.com/en/8.1.x/shell-completion/#enabling-completion)
for instructions on how to enable completion in other shells.
