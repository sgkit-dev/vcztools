import numpy
from setuptools import Extension, setup

_vcztools_module = Extension(
    "vcztools._vcztools",
    sources=["vcztools/_vcztoolsmodule.c", "lib/vcf_encoder.c"],
    extra_compile_args=["-std=c99"],
    include_dirs=["lib", numpy.get_include()],
)

setup(
    name="vcztools",
    ext_modules=[_vcztools_module],
)
