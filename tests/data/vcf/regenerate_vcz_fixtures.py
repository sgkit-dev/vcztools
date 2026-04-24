"""Regenerate committed VCZ fixtures from source VCFs.

Run manually when the source VCFs or the bio2zarr schema change:

    uv run python tests/data/vcf/regenerate_vcz_fixtures.py

The script converts each source VCF into a temporary directory VCZ
via bio2zarr, then repacks it as a DEFLATE-compressed .vcz.zip. We
do not let bio2zarr write the .zip directly because bio2zarr's
ZipStore output is uncompressed (ZIP_STORED), and the per-array
zarr metadata files (.zarray / .zattrs / zarr.json) compress
extremely well under DEFLATE — typically 20-45 % shrinkage on
fixture-sized inputs.

Per-fixture chunk sizes are tuned so most fixtures land at 2-3
variant chunks, giving the test suite multi-chunk coverage on the
committed fixtures themselves rather than only via on-the-fly
copy_vcz rechunks.
"""

import dataclasses
import pathlib
import tempfile
import zipfile

from bio2zarr import vcf

FIXTURE_DIR = pathlib.Path(__file__).parent


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
            vcf.convert(
                [src],
                tmp_vcz,
                worker_processes=0,
                variants_chunk_size=fx.variants_chunk_size,
                samples_chunk_size=fx.samples_chunk_size,
                zarr_format=fx.zarr_format,
            )
            _deflate_zip_directory(tmp_vcz, dst)

        after = dst.stat().st_size
        delta = (after - before) / before * 100 if before else 0.0
        print(f"  wrote {dst.name} {before:7d} -> {after:7d} bytes ({delta:+.0f}%)")


if __name__ == "__main__":
    main()
