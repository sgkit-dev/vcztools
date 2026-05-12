"""Regenerate committed VCZ fixtures from source VCFs.

Run manually when the source VCFs or the bio2zarr schema change:

    uv run python tests/data/vcf/regenerate_vcz_fixtures.py

The script converts each source VCF into a temporary directory VCZ
via bio2zarr, rechunks variant-only fields to mirror the proportional
chunking shape that production stores use (see
``tests.vcz_builder.DEFAULT_VARIANT_CHUNK_MULTIPLIERS``), then repacks
the result as a DEFLATE-compressed .vcz.zip. We do not let bio2zarr
write the .zip directly because bio2zarr's ZipStore output is
uncompressed (ZIP_STORED), and the per-array zarr metadata files
(.zarray / .zattrs / zarr.json) compress extremely well under
DEFLATE — typically 20-45 % shrinkage on fixture-sized inputs.

Per-fixture chunk sizes are tuned so most fixtures land at 2-3
variant chunks on ``call_*``, giving the test suite multi-chunk
coverage on the committed fixtures themselves rather than only via
on-the-fly copy_vcz rechunks. Variant-only fields then expand to
multiples of that floor per the recipe, so reading multiple fields
together exercises the GCD-vs-min stream-chunk-size regime.
"""

import dataclasses
import os
import pathlib
import subprocess
import sys
import tempfile
import zipfile

import zarr

# Make sure the in-tree ``tests`` package is importable when this script
# is invoked directly (``uv run python tests/data/vcf/regenerate_vcz_fixtures.py``).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.vcz_builder import DEFAULT_VARIANT_CHUNK_MULTIPLIERS  # noqa: E402

FIXTURE_DIR = pathlib.Path(__file__).parent


def _bio2zarr_convert(
    src: pathlib.Path,
    dst: pathlib.Path,
    *,
    variants_chunk_size: int | None,
    samples_chunk_size: int | None,
    zarr_format: int | None,
):
    """Run ``bio2zarr.vcf.convert`` in a subprocess so the
    ``BIO2ZARR_ZARR_FORMAT`` env var (which bio2zarr reads at import
    time) can vary per fixture in a single regenerate run.
    """
    env = dict(os.environ)
    if zarr_format is not None:
        env["BIO2ZARR_ZARR_FORMAT"] = str(zarr_format)
    args = [
        sys.executable,
        "-c",
        "import sys, pathlib; "
        "from bio2zarr import vcf; "
        "vcf.convert("
        "[pathlib.Path(sys.argv[1])], pathlib.Path(sys.argv[2]), "
        "worker_processes=0, "
        f"variants_chunk_size={variants_chunk_size!r}, "
        f"samples_chunk_size={samples_chunk_size!r})",
        str(src),
        str(dst),
    ]
    subprocess.run(args, env=env, check=True)


@dataclasses.dataclass(frozen=True)
class Fixture:
    vcf_name: str
    variants_chunk_size: int | None = None
    samples_chunk_size: int | None = None
    zarr_format: int | None = None


FIXTURES = [
    Fixture("sample.vcf.gz", variants_chunk_size=4),
    Fixture("sample.vcf.gz", variants_chunk_size=4, zarr_format=3),
    Fixture("sample-split-alleles.vcf.gz", variants_chunk_size=7),
    Fixture("msprime_diploid.vcf.gz", variants_chunk_size=3),
    Fixture("1kg_2020_chrM.vcf.gz", variants_chunk_size=10),
    Fixture("chr22.vcf.gz", variants_chunk_size=50, samples_chunk_size=50),
    Fixture("field_type_combos.vcf.gz", variants_chunk_size=100),
    Fixture("1kg_2020_chr20_annotations.bcf", variants_chunk_size=11),
]


def _rechunk_variant_axis_in_place(group, new_chunks: dict[str, int]):
    """Rebuild every variant-axis array whose name appears in
    ``new_chunks`` with the requested chunks[0]. Codecs, filters, and
    attrs are preserved; the source array is deleted and recreated in
    place. ``region_index`` is intentionally left alone — it indexes at
    ``call_*`` (min_chunk) granularity, which the recipe never touches.
    """
    zarr_format = group.metadata.zarr_format
    for name in sorted(group.array_keys()):
        if name not in new_chunks:
            continue
        arr = group[name]
        if name.startswith("call_"):
            continue
        chunk_size = new_chunks[name]
        existing = tuple(arr.chunks)
        if existing[0] == chunk_size:
            continue
        target_chunks = (chunk_size,) + existing[1:]
        data = arr[...]
        attrs = dict(arr.attrs)
        # Preserve dimension_names — the array→dim mapping vcztools relies
        # on (e.g. utils.has_variants_axis). zarr v2 stores it in the
        # ``_ARRAY_DIMENSIONS`` attr; zarr v3 stores it in the array
        # metadata. Read whichever is populated and propagate to both
        # surfaces on the rebuilt array so the result loads the same way
        # the original did.
        v2_dim_attr = attrs.get("_ARRAY_DIMENSIONS")
        v3_dim_meta = getattr(arr.metadata, "dimension_names", None)
        dimension_names = v3_dim_meta
        if dimension_names is None and v2_dim_attr is not None:
            dimension_names = tuple(v2_dim_attr)
        kwargs = dict(
            name=name,
            shape=data.shape,
            chunks=target_chunks,
            dtype=data.dtype,
        )
        if zarr_format == 2:
            # v2 stores dimension names in the ``_ARRAY_DIMENSIONS`` attr,
            # which is copied via the attrs loop below — no create_array
            # kwarg.
            kwargs["compressors"] = arr.metadata.compressor
            if arr.metadata.filters is not None:
                kwargs["filters"] = arr.metadata.filters
        else:
            # Zarr v3 splits the codec pipeline into one array-to-bytes
            # serializer (e.g. BytesCodec / VLenUTF8Codec) followed by
            # zero or more bytes-to-bytes compressors (e.g. Blosc).
            codecs = arr.metadata.codecs
            kwargs["serializer"] = codecs[0]
            kwargs["compressors"] = tuple(codecs[1:])
            if dimension_names is not None:
                kwargs["dimension_names"] = tuple(dimension_names)
        del group[name]
        new_arr = group.create_array(**kwargs)
        new_arr[...] = data
        for k, v in attrs.items():
            new_arr.attrs[k] = v


def _deflate_zip_directory(src_dir: pathlib.Path, out_path: pathlib.Path):
    out_path.unlink(missing_ok=True)
    with zipfile.ZipFile(
        out_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir).as_posix())


def main():
    for fx in FIXTURES:
        src = FIXTURE_DIR / fx.vcf_name
        stem = src
        while stem.suffix:
            stem = stem.with_suffix("")
        vcz_suffix = ".vcz3" if fx.zarr_format == 3 else ".vcz"
        dst = FIXTURE_DIR / f"{stem.name}{vcz_suffix}.zip"
        before = dst.stat().st_size if dst.exists() else 0

        with tempfile.TemporaryDirectory() as tmp:
            tmp_vcz = pathlib.Path(tmp) / f"{stem.name}{vcz_suffix}"
            print(f"Converting {src.name} -> {dst.name}")
            _bio2zarr_convert(
                src,
                tmp_vcz,
                variants_chunk_size=fx.variants_chunk_size,
                samples_chunk_size=fx.samples_chunk_size,
                zarr_format=fx.zarr_format,
            )
            v_chunk = fx.variants_chunk_size
            if v_chunk is not None:
                new_chunks = {
                    name: factor * v_chunk
                    for name, factor in DEFAULT_VARIANT_CHUNK_MULTIPLIERS.items()
                }
                group = zarr.open(str(tmp_vcz), mode="a")
                _rechunk_variant_axis_in_place(group, new_chunks)
            _deflate_zip_directory(tmp_vcz, dst)

        after = dst.stat().st_size
        delta = (after - before) / before * 100 if before else 0.0
        print(f"  wrote {dst.name} {before:7d} -> {after:7d} bytes ({delta:+.0f}%)")


if __name__ == "__main__":
    main()
