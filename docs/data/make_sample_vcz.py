"""Generate the example dataset used by the Reading VCZ documentation.

Derives ``docs/data/sample.vcz.zip`` from the canonical test fixture
``tests/data/vcf/sample.vcz.zip`` by inserting a single null sample
(``sample_id == ""``) at index 1, padding every sample-dimensioned array
accordingly. The result demonstrates how the reader treats null samples.

Run with::

    uv run python docs/data/make_sample_vcz.py
"""

import pathlib
import tempfile
import zipfile

import numpy as np
import zarr

import vcztools
from vcztools import constants

NULL_INDEX = 1

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "tests" / "data" / "vcf" / "sample.vcz.zip"
DEST = REPO_ROOT / "docs" / "data" / "sample.vcz.zip"


def _pad_value(name, dtype):
    """The fill used for the inserted null column of array ``name``.

    Null-sample columns are never read, so the value is only chosen to keep
    each column internally consistent: a missing genotype is masked, and
    integer fields use the missing sentinel.
    """
    if name == "sample_id":
        return ""
    if dtype.kind == "b":
        return name == "call_genotype_mask"
    return constants.INT_MISSING


def _deflate_zip_directory(src_dir, out_path):
    out_path.unlink(missing_ok=True)
    with zipfile.ZipFile(
        out_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir).as_posix())


def _copy_with_null_sample(source, dest):
    dest.attrs.update(dict(source.attrs))
    for name in sorted(source.array_keys()):
        src_array = source[name]
        dims = src_array.attrs["_ARRAY_DIMENSIONS"]
        data = src_array[...]
        chunks = list(src_array.chunks)
        if "samples" in dims:
            samples_axis = dims.index("samples")
            pad = _pad_value(name, src_array.dtype)
            data = np.insert(data, NULL_INDEX, pad, axis=samples_axis)
            chunks[samples_axis] = data.shape[samples_axis]
        new_array = dest.create_array(
            name=name,
            shape=data.shape,
            dtype=src_array.dtype,
            chunks=tuple(chunks),
            fill_value=src_array.fill_value,
        )
        new_array[...] = data
        new_array.attrs.update(dict(src_array.attrs))


def main():
    source = vcztools.open_zarr(SOURCE)
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = pathlib.Path(tmp_dir) / "sample.vcz"
        dest = zarr.open_group(store=str(store), mode="w", zarr_format=2)
        _copy_with_null_sample(source, dest)
        _deflate_zip_directory(store, DEST)
    print(f"Wrote {DEST}")


if __name__ == "__main__":
    main()
