import numpy
from setuptools import Extension
from setuptools import setup

_vcztools_module = Extension(
    "_vcztools",
    sources=["_vcztoolsmodule.c", "lib/vcf_encoder.c"],
    extra_compile_args=["-std=c99"],
    include_dirs=["lib", numpy.get_include()],
)

setup(
    ext_modules=[_vcztools_module],
)
