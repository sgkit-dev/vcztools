import pathlib
import sys

from . import utils

_VCF_DIR = pathlib.Path(__file__).parent / "data" / "vcf"

# Chunk size variants used by parametrized tests. Prewarming these avoids
# races between xdist workers on first bio2zarr conversion.
_SAMPLE_VARIANTS_CHUNK_SIZES = (1, 3, 4, 7)


def _prewarm_vcz_cache():
    if sys.platform == "win32":
        # We don't do any conversion on Windows
        return
    for vcf_path in sorted(_VCF_DIR.glob("*.vcf.gz")):
        utils.vcz_path_cache(vcf_path)
    for vcf_path in sorted(_VCF_DIR.glob("*.bcf")):
        utils.vcz_path_cache(vcf_path)

    sample_vcf = _VCF_DIR / "sample.vcf.gz"
    # test_write_vcf[obstore] needs a directory VCZ.
    utils.vcz_path_cache(sample_vcf, as_directory=True)
    for vcs in _SAMPLE_VARIANTS_CHUNK_SIZES:
        utils.vcz_path_cache(sample_vcf, variants_chunk_size=vcs)

    split_alleles_vcf = _VCF_DIR / "sample-split-alleles.vcf.gz"
    utils.vcz_path_cache(split_alleles_vcf, variants_chunk_size=4)


def pytest_configure(config):
    # Prewarm only on the xdist controller (or non-xdist runs). Workers
    # inherit the populated cache and never race on first conversion.
    if hasattr(config, "workerinput"):
        return
    _prewarm_vcz_cache()
