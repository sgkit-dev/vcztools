/* CUnit cross-checks for vcz_adler32 / vcz_compress2 / vcz_compress_bound.
 *
 * Compares our in-tree replacements byte-for-byte against the corresponding
 * zlib functions. Linking libz is allowed in the test suite only; production
 * builds keep vcztools dependency-free. */
#define _GNU_SOURCE
#include <CUnit/Basic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <zlib.h>
#include <vcf_encoder.h>

/* Deterministic LCG so the random tests stay reproducible across runs and
 * platforms. */
static void
fill_lcg(uint8_t *buf, size_t len, uint32_t seed)
{
    uint32_t state = seed | 1u;
    size_t i;
    for (i = 0; i < len; i++) {
        state = state * 1664525u + 1013904223u;
        buf[i] = (uint8_t) (state >> 16);
    }
}

/* Drive both adlers, including a multi-call accumulation pass that exercises
 * the running-checksum entry point identically. */
static void
check_adler_pair(const uint8_t *buf, size_t len)
{
    uint32_t got, ref, got2;
    uLong z, z2;
    size_t mid;

    got = vcz_adler32(1, buf, len);
    z = adler32(1uL, buf, (uInt) len);
    ref = (uint32_t) z;
    CU_ASSERT_EQUAL_FATAL(got, ref);

    if (len >= 2) {
        mid = len / 2;
        got2 = vcz_adler32(1, buf, mid);
        got2 = vcz_adler32(got2, buf + mid, len - mid);
        z2 = adler32(1uL, buf, (uInt) mid);
        z2 = adler32(z2, buf + mid, (uInt) (len - mid));
        CU_ASSERT_EQUAL_FATAL(got2, (uint32_t) z2);
        CU_ASSERT_EQUAL_FATAL(got2, ref);
    }
}

static void
test_adler32_empty(void)
{
    /* zlib's contract: adler32(1, NULL, 0) == 1. */
    CU_ASSERT_EQUAL_FATAL(vcz_adler32(1, NULL, 0), 1u);
    CU_ASSERT_EQUAL_FATAL(vcz_adler32(1, NULL, 0), (uint32_t) adler32(1uL, NULL, 0));
}

static void
test_adler32_single_byte(void)
{
    uint8_t b;
    for (b = 0; b < 32; b++) {
        check_adler_pair(&b, 1);
    }
}

static void
test_adler32_small(void)
{
    /* "Wikipedia" example: adler32("Wikipedia") == 0x11E60398. */
    const uint8_t msg[] = { 'W', 'i', 'k', 'i', 'p', 'e', 'd', 'i', 'a' };
    uint32_t got = vcz_adler32(1, msg, sizeof(msg));
    CU_ASSERT_EQUAL_FATAL(got, 0x11E60398u);
    CU_ASSERT_EQUAL_FATAL(got, (uint32_t) adler32(1uL, msg, (uInt) sizeof(msg)));
}

static void
test_adler32_lengths(void)
{
    /* Cover lengths that bracket the NMAX=5552 fold and the vectorised
     * 16-byte unroll boundary. */
    static const size_t lengths[]
        = { 0, 1, 2, 15, 16, 17, 31, 32, 33, 100, 1000, 5551, 5552, 5553, 11104, 65536 };
    size_t i, n;
    uint8_t *buf;
    for (i = 0; i < sizeof(lengths) / sizeof(lengths[0]); i++) {
        n = lengths[i];
        if (n == 0) {
            check_adler_pair(NULL, 0);
            continue;
        }
        buf = malloc(n);
        CU_ASSERT_FATAL(buf != NULL);
        fill_lcg(buf, n, (uint32_t) (n + 17));
        check_adler_pair(buf, n);
        free(buf);
    }
}

static void
test_adler32_constant_input(void)
{
    /* All-zero and all-0xFF: catches sign-extension bugs in the inner sum. */
    size_t n = 1u << 16;
    uint8_t *buf = malloc(n);
    CU_ASSERT_FATAL(buf != NULL);

    memset(buf, 0x00, n);
    check_adler_pair(buf, n);

    memset(buf, 0xFF, n);
    check_adler_pair(buf, n);

    free(buf);
}

static void
test_adler32_large(void)
{
    /* 1 MiB random buffer crosses the NMAX fold many times. */
    size_t n = 1u << 20;
    uint8_t *buf = malloc(n);
    CU_ASSERT_FATAL(buf != NULL);
    fill_lcg(buf, n, 0xABCDEFu);
    check_adler_pair(buf, n);
    free(buf);
}

static void
test_adler32_seed_propagation(void)
{
    /* Non-default seed: result must still match. */
    uint8_t buf[64];
    uint32_t got;
    uLong z;

    fill_lcg(buf, sizeof(buf), 42);
    got = vcz_adler32(0xDEAD0001u, buf, sizeof(buf));
    z = adler32(0xDEAD0001uL, buf, (uInt) sizeof(buf));
    CU_ASSERT_EQUAL_FATAL(got, (uint32_t) z);
}

/* For vcz_compress_bound + vcz_compress2 we want the exact same bytes
 * zlib's level-0 path produces. */
static void
check_compress_pair(const uint8_t *src, size_t src_len)
{
    size_t bound = vcz_compress_bound(src_len);
    uint8_t *got = malloc(bound + 8);
    uint8_t *ref = malloc(bound + 8);
    size_t got_len = bound;
    uLongf ref_len = (uLongf) (bound + 8);
    int rc;

    CU_ASSERT_FATAL(got != NULL);
    CU_ASSERT_FATAL(ref != NULL);

    rc = vcz_compress2(got, &got_len, src, src_len, 0);
    CU_ASSERT_EQUAL_FATAL(rc, VCZ_Z_OK);

    rc = compress2(ref, &ref_len, src, (uLong) src_len, 0);
    CU_ASSERT_EQUAL_FATAL(rc, Z_OK);

    CU_ASSERT_EQUAL_FATAL(got_len, (size_t) ref_len);
    CU_ASSERT_EQUAL_FATAL(got_len, bound); /* bound is exact for stored */
    CU_ASSERT_EQUAL_FATAL(memcmp(got, ref, got_len), 0);

    free(got);
    free(ref);
}

static void
test_compress_bound_matches_compress2(void)
{
    /* zlib's compressBound is an upper bound; for stored level-0 the
     * output is deterministic, so our bound must equal the actual
     * compress2 output length. Sample sizes bracket the 65535-byte
     * stored-block boundary. */
    static const size_t lengths[]
        = { 0, 1, 7, 65, 1024, 65534, 65535, 65536, 100000, 200000 };
    size_t i, n;
    uint8_t *buf;
    uint8_t *out;
    uLongf actual;
    int rc;

    for (i = 0; i < sizeof(lengths) / sizeof(lengths[0]); i++) {
        n = lengths[i];
        buf = malloc(n == 0 ? 1 : n);
        CU_ASSERT_FATAL(buf != NULL);
        if (n > 0) {
            fill_lcg(buf, n, (uint32_t) (n + 1));
        }
        actual = (uLongf) (vcz_compress_bound(n) + 8);
        out = malloc((size_t) actual);
        CU_ASSERT_FATAL(out != NULL);
        rc = compress2(out, &actual, buf, (uLong) n, 0);
        CU_ASSERT_EQUAL_FATAL(rc, Z_OK);
        CU_ASSERT_EQUAL_FATAL((size_t) actual, vcz_compress_bound(n));
        free(out);
        free(buf);
    }
}

static void
test_compress2_byte_for_byte(void)
{
    static const size_t lengths[]
        = { 0, 1, 17, 1024, 65535, 65536, 70000, 131070, 131071, 200000 };
    size_t i, n;
    uint8_t *buf;
    for (i = 0; i < sizeof(lengths) / sizeof(lengths[0]); i++) {
        n = lengths[i];
        buf = malloc(n == 0 ? 1 : n);
        CU_ASSERT_FATAL(buf != NULL);
        if (n > 0) {
            fill_lcg(buf, n, (uint32_t) (i * 11 + 1));
        }
        check_compress_pair(buf, n);
        free(buf);
    }
}

static void
test_compress2_roundtrip(void)
{
    /* Inflate our output and check we get the original payload back. */
    size_t n = 100000;
    uint8_t *src = malloc(n);
    uint8_t *enc;
    uint8_t *dec = malloc(n);
    size_t enc_len;
    uLongf dec_len = (uLongf) n;
    int rc;

    CU_ASSERT_FATAL(src != NULL);
    CU_ASSERT_FATAL(dec != NULL);
    fill_lcg(src, n, 7);
    enc_len = vcz_compress_bound(n);
    enc = malloc(enc_len);
    CU_ASSERT_FATAL(enc != NULL);

    rc = vcz_compress2(enc, &enc_len, src, n, 0);
    CU_ASSERT_EQUAL_FATAL(rc, VCZ_Z_OK);

    rc = uncompress(dec, &dec_len, enc, (uLong) enc_len);
    CU_ASSERT_EQUAL_FATAL(rc, Z_OK);
    CU_ASSERT_EQUAL_FATAL((size_t) dec_len, n);
    CU_ASSERT_EQUAL_FATAL(memcmp(src, dec, n), 0);

    free(src);
    free(enc);
    free(dec);
}

static void
test_compress2_rejects_nonzero_level(void)
{
    uint8_t src = 0;
    uint8_t out[64];
    size_t out_len;
    int rc;
    int level;

    for (level = 1; level <= 9; level++) {
        out_len = sizeof(out);
        rc = vcz_compress2(out, &out_len, &src, 1, level);
        CU_ASSERT_EQUAL_FATAL(rc, VCZ_Z_STREAM_ERROR);
    }
    /* Z_DEFAULT_COMPRESSION (-1) too. */
    out_len = sizeof(out);
    rc = vcz_compress2(out, &out_len, &src, 1, -1);
    CU_ASSERT_EQUAL_FATAL(rc, VCZ_Z_STREAM_ERROR);
}

static void
test_compress2_buffer_too_small(void)
{
    uint8_t src[64];
    uint8_t out[8];
    size_t out_len;
    int rc;

    fill_lcg(src, sizeof(src), 3);
    out_len = sizeof(out);
    rc = vcz_compress2(out, &out_len, src, sizeof(src), 0);
    CU_ASSERT_EQUAL_FATAL(rc, VCZ_Z_BUF_ERROR);
    /* On error, dest_len must remain at the caller-supplied capacity so
     * the caller can distinguish "wrote N bytes" from "didn't write". */
    CU_ASSERT_EQUAL_FATAL(out_len, sizeof(out));
}

/*=================================================
  Test suite management. Mirrors lib/tests.c so the
  test binary behaves the same way for runners.
  =================================================
*/

static int
vcz_suite_init(void)
{
    return CUE_SUCCESS;
}

static int
vcz_suite_cleanup(void)
{
    return CUE_SUCCESS;
}

static void
handle_cunit_error(void)
{
    fprintf(stderr, "CUnit error occured: %d: %s\n", CU_get_error(), CU_get_error_msg());
    exit(EXIT_FAILURE);
}

static int
test_main(CU_TestInfo *tests, int argc, char **argv)
{
    int ret;
    CU_pTest test;
    CU_pSuite suite;
    CU_SuiteInfo suites[] = {
        {
            .pName = "vcz_zlib",
            .pTests = tests,
            .pInitFunc = vcz_suite_init,
            .pCleanupFunc = vcz_suite_cleanup,
        },
        CU_SUITE_INFO_NULL,
    };
    if (CUE_SUCCESS != CU_initialize_registry()) {
        handle_cunit_error();
    }
    if (CUE_SUCCESS != CU_register_suites(suites)) {
        handle_cunit_error();
    }
    CU_basic_set_mode(CU_BRM_VERBOSE);

    if (argc == 1) {
        CU_basic_run_tests();
    } else if (argc == 2) {
        suite = CU_get_suite_by_name("vcz_zlib", CU_get_registry());
        if (suite == NULL) {
            printf("Suite not found\n");
            return EXIT_FAILURE;
        }
        test = CU_get_test_by_name(argv[1], suite);
        if (test == NULL) {
            printf("Test '%s' not found\n", argv[1]);
            return EXIT_FAILURE;
        }
        CU_basic_run_test(suite, test);
    } else {
        printf("usage: %s <test_name>\n", argv[0]);
        return EXIT_FAILURE;
    }

    ret = EXIT_SUCCESS;
    if (CU_get_number_of_tests_failed() != 0) {
        printf("Test failed!\n");
        ret = EXIT_FAILURE;
    }
    CU_cleanup_registry();
    return ret;
}

int
main(int argc, char **argv)
{
    CU_TestInfo tests[] = {
        { "test_adler32_empty", test_adler32_empty },
        { "test_adler32_single_byte", test_adler32_single_byte },
        { "test_adler32_small", test_adler32_small },
        { "test_adler32_lengths", test_adler32_lengths },
        { "test_adler32_constant_input", test_adler32_constant_input },
        { "test_adler32_large", test_adler32_large },
        { "test_adler32_seed_propagation", test_adler32_seed_propagation },
        { "test_compress_bound_matches_compress2",
            test_compress_bound_matches_compress2 },
        { "test_compress2_byte_for_byte", test_compress2_byte_for_byte },
        { "test_compress2_roundtrip", test_compress2_roundtrip },
        { "test_compress2_rejects_nonzero_level", test_compress2_rejects_nonzero_level },
        { "test_compress2_buffer_too_small", test_compress2_buffer_too_small },
        { NULL, NULL },
    };
    return test_main(tests, argc, argv);
}
