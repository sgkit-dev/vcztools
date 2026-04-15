"""Regenerate committed VCZ fixtures from source VCFs.

Run manually when the source VCFs or the bio2zarr schema change:

    uv run python tests/data/vcf/regenerate_vcz_fixtures.py

bio2zarr.vcf.convert detects a .zip suffix on the output path and
writes a ZipStore natively.
"""

import pathlib

from bio2zarr import vcf

FIXTURE_DIR = pathlib.Path(__file__).parent

# (source VCF filename, variants_chunk_size, samples_chunk_size)
FIXTURES = [
    ("sample.vcf.gz", None, None),
]


def main():
    for vcf_name, variants_chunk_size, samples_chunk_size in FIXTURES:
        src = FIXTURE_DIR / vcf_name
        stem = src
        while stem.suffix:
            stem = stem.with_suffix("")
        dst = FIXTURE_DIR / f"{stem.name}.vcz.zip"
        if dst.exists():
            dst.unlink()
        print(f"Converting {src.name} -> {dst.name}")
        vcf.convert(
            [src],
            dst,
            worker_processes=0,
            variants_chunk_size=variants_chunk_size,
            samples_chunk_size=samples_chunk_size,
        )
        print(f"  wrote {dst.name} ({dst.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
