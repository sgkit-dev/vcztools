project('vcf_encoder', ['c'],
    default_options: ['c_std=c99']
)

cc = meson.get_compiler('c')
m_dep = cc.find_library('m', required: false)

extra_c_args = [
    '-Wall', '-Wextra', '-Werror', '-Wpedantic', '-W',
    '-Wmissing-prototypes',  '-Wstrict-prototypes',
    '-Wconversion', '-Wshadow', '-Wpointer-arith', '-Wcast-align',
    '-Wcast-qual', '-Wwrite-strings', '-Wnested-externs',
    '-fshort-enums', '-fno-common']

lib_sources = ['formats.c', 'compute.c']
lib_headers = ['vcztools.h']

cunit_dep = dependency('cunit')

test_formats = executable('test_formats',
    sources: ['test_formats.c', 'formats.c'],
    dependencies: [cunit_dep, m_dep],
    c_args: extra_c_args,
)
test('test_formats', test_formats)

test_compute = executable('test_compute',
    sources: ['test_compute.c', 'compute.c'],
    dependencies: [cunit_dep, m_dep],
    c_args: extra_c_args,
)
test('test_compute', test_compute)

# Cross-check the in-tree zlib replacements against libz. Production
# builds (setup.py) don't link libz; the test runner is the one place
# we're allowed to depend on it.
z_dep = cc.find_library('z', required: true)
test_zlib = executable('test_zlib',
    sources: ['test_zlib.c', 'formats.c'],
    dependencies: [cunit_dep, m_dep, z_dep],
    c_args: extra_c_args,
)
test('test_zlib', test_zlib)
