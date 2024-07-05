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

lib_sources = ['vcf_encoder.c']
lib_headers = ['vcf_encoder.h']

cunit_dep = dependency('cunit')

tests = executable('tests',
    sources: ['tests.c', 'vcf_encoder.c'],
    dependencies: [cunit_dep, m_dep],
    c_args: extra_c_args, 
)
test('tests', tests)
