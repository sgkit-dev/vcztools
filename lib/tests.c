#define _GNU_SOURCE
#include <CUnit/Basic.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <float.h>
#include <vcf_encoder.h>

FILE *_devnull;

static void
test_int_field_1d(void)
{
    const size_t num_rows = 5;
    const int32_t data[] = { 1, 2, 12345789, -1, -100 };
    vcz_field_t field = { .name = "test",
        .type = VCZ_TYPE_INT,
        .item_size = 4,
        .num_columns = 1,
        .data = (const char *) data };
    char buf[1000];
    const char *expected[] = { "1", "2", "12345789", ".", "-100" };
    int64_t ret;
    size_t j;

    for (j = 0; j < num_rows; j++) {
        ret = vcz_field_write(&field, j, buf, 1000, 0);
        /* printf("ret = %d\n", (int)ret); */
        /* printf("'%.*s': %s\n", (int) ret, buf, expected[j]); */
        CU_ASSERT_EQUAL_FATAL(ret, strlen(expected[j]));
        CU_ASSERT_NSTRING_EQUAL(buf, expected[j], ret);
    }
}

static void
test_int_field_1d_overflow(void)
{
    const int32_t data[] = { 1, 2, 12345789, -100, INT32_MIN, INT32_MAX, -1 };
    const size_t num_rows = sizeof(data) / sizeof(*data);
    vcz_field_t field = { .name = "test",
        .type = VCZ_TYPE_INT,
        .item_size = 4,
        .num_columns = 1,
        .data = (const char *) data };
    int64_t ret, buflen;
    size_t j;
    char *buf;

    for (j = 0; j < num_rows - 1; j++) {
        /* printf("%d\n", (int) data[j]); */
        for (buflen = 0; buflen <= VCZ_INT32_BUF_SIZE; buflen++) {
            /* printf("buflen = %d\n", (int) buflen); */
            buf = malloc((size_t) buflen);
            CU_ASSERT_FATAL(buf != NULL);
            ret = vcz_field_write(&field, j, buf, buflen, 0);
            free(buf);
            CU_ASSERT_FATAL(ret == VCZ_ERR_BUFFER_OVERFLOW);
        }
    }
    j = num_rows - 1;
    CU_ASSERT_FATAL(data[j] == -1);
    /* Missing data is treated differently. Just need 1 byte for "." */
    for (buflen = 0; buflen < 1; buflen++) {
        buf = malloc((size_t) buflen);
        CU_ASSERT_FATAL(buf != NULL);
        ret = vcz_field_write(&field, j, buf, buflen, 0);
        free(buf);
        CU_ASSERT_FATAL(ret == VCZ_ERR_BUFFER_OVERFLOW);
    }
}

static void
test_int_field_2d(void)
{
    const size_t num_rows = 4;
    const int32_t data[] = { 1, 2, 3, 1234, 5678, -2, -1, -2, -2, -2, -2, -2 };
    vcz_field_t field = { .name = "test",
        .type = VCZ_TYPE_INT,
        .item_size = 4,
        .num_columns = 3,
        .data = (const char *) data };
    char buf[1000];
    const char *expected[] = { "1,2,3", "1234,5678", ".", "" };
    int64_t ret;
    size_t j;

    for (j = 0; j < num_rows; j++) {
        ret = vcz_field_write(&field, j, buf, 1000, 0);
        CU_ASSERT_EQUAL_FATAL(ret, strlen(expected[j]));
        /* printf("%s: %s\n", buf, expected[j]); */
        CU_ASSERT_NSTRING_EQUAL_FATAL(buf, expected[j], ret);
    }
}

static void
test_float_field_1d(void)
{
    float data[] = { 1.0f, 2.1f, INT32_MIN, 12345789.0f, -1, -100.123f, 0 };

    const size_t num_rows = sizeof(data) / sizeof(*data);
    vcz_field_t field = { .name = "test",
        .type = VCZ_TYPE_FLOAT,
        .item_size = 4,
        .num_columns = 1,
        .data = (const char *) data };
    char buf[1000];
    const char *expected[]
        = { "1", "2.1", "-2147483648", "12345789", "-1", "-100.123", "." };
    int64_t ret;
    size_t j;
    int32_t *int_data = (int32_t *) data;

    int_data[num_rows - 1] = VCZ_FLOAT32_MISSING_AS_INT32;

    for (j = 0; j < num_rows; j++) {
        ret = vcz_field_write(&field, j, buf, 1000, 0);
        /* printf("ret = %d\n", (int)ret); */
        /* printf("'%.*s':'%s'\n", (int) ret, buf, expected[j]); */
        CU_ASSERT_EQUAL_FATAL(ret, strlen(expected[j]));
        CU_ASSERT_NSTRING_EQUAL(buf, expected[j], ret);
    }
}

static void
test_float_field_1d_overflow(void)
{
    float data[] = { 1.0f, 2.1f, 12345789.0f, (float) M_PI, -1, -100.123f, 0 };
    const size_t num_rows = sizeof(data) / sizeof(*data);
    vcz_field_t field = { .name = "test",
        .type = VCZ_TYPE_FLOAT,
        .item_size = 4,
        .num_columns = 1,
        .data = (const char *) data };
    int64_t ret, buflen;
    size_t j;
    char *buf;
    int32_t *int_data = (int32_t *) data;

    int_data[num_rows - 1] = VCZ_FLOAT32_MISSING_AS_INT32;

    for (j = 0; j < num_rows - 1; j++) {
        /* printf("%d\n", (int) data[j]); */
        for (buflen = 0; buflen <= VCZ_FLOAT32_BUF_SIZE; buflen++) {
            /* printf("buflen = %d\n", (int) buflen); */
            buf = malloc((size_t) buflen);
            CU_ASSERT_FATAL(buf != NULL);
            ret = vcz_field_write(&field, j, buf, buflen, 0);
            free(buf);
            CU_ASSERT_FATAL(ret == VCZ_ERR_BUFFER_OVERFLOW);
        }
    }
    j = num_rows - 1;
    /* Missing data is treated differently. Just need 1 byte for "." */
    for (buflen = 0; buflen < 1; buflen++) {
        buf = malloc((size_t) buflen);
        CU_ASSERT_FATAL(buf != NULL);
        ret = vcz_field_write(&field, j, buf, buflen, 0);
        free(buf);
        CU_ASSERT_FATAL(ret == VCZ_ERR_BUFFER_OVERFLOW);
    }
}

static void
test_string_field_1d(void)
{
    /* item_size=3, rows=3, cols=1 */
    const size_t num_rows = 3;
    const char data[] = "X\0\0" /* X */
                        "XX\0"  /* XX*/
                        "XXX";  /* XXX, */

    vcz_field_t field = { .name = "test",
        .type = VCZ_TYPE_STRING,
        .item_size = 3,
        .num_columns = 1,
        .data = data };
    char buf[1000];
    const char *expected[] = { "X", "XX", "XXX" };
    int64_t ret;
    size_t j;

    CU_ASSERT_EQUAL_FATAL(sizeof(data), 10);
    for (j = 0; j < num_rows; j++) {
        ret = vcz_field_write(&field, j, buf, 1000, 0);
        CU_ASSERT_EQUAL_FATAL(ret, strlen(expected[j]));
        CU_ASSERT_NSTRING_EQUAL(buf, expected[j], ret);
    }
}

static void
test_string_field_1d_overflow(void)
{
    /* item_size=4, rows=3, cols=1 */
    const size_t num_rows = 4;
    const size_t item_size = 3;
    const char data[] = "X\0\0"  /* X */
                        "XX\0"   /* XX*/
                        "XXX"    /* XXX, */
                        ".\0\0"; /* .  */
    vcz_field_t field = { .name = "test",
        .type = VCZ_TYPE_STRING,
        .item_size = 3,
        .num_columns = 1,
        .data = data };
    int64_t ret, buflen;
    size_t j;
    char *buf;

    for (j = 0; j < num_rows - 1; j++) {
        for (buflen = 0; buflen < (int) item_size; buflen++) {
            /* printf("buflen = %d\n", (int) buflen); */
            buf = malloc((size_t) buflen);
            CU_ASSERT_FATAL(buf != NULL);
            ret = vcz_field_write(&field, j, buf, buflen, 0);
            free(buf);
            if (ret < 0) {
                CU_ASSERT_FATAL(ret == VCZ_ERR_BUFFER_OVERFLOW);
            } else {
                CU_ASSERT_EQUAL_FATAL(ret, j + 1);
            }
        }
    }
    j = num_rows - 1;
    /* Missing data is treated differently. Just need 1 byte for "." */
    for (buflen = 0; buflen < 1; buflen++) {
        buf = malloc((size_t) buflen);
        CU_ASSERT_FATAL(buf != NULL);
        ret = vcz_field_write(&field, j, buf, buflen, 0);
        free(buf);
        CU_ASSERT_FATAL(ret == VCZ_ERR_BUFFER_OVERFLOW);
    }
}

static void
test_string_field_2d(void)
{
    /* item_size=3, rows=3, cols=3 */
    const size_t num_rows = 3;
    const char data[] = "X\0\0Y\0\0\0\0\0" /* [X, Y] */
                        "XX\0YY\0Z\0\0"    /* [XX, YY, Z], */
                        "XXX\0\0\0\0\0";   /* [XXX], */
    vcz_field_t field = { .name = "test",
        .type = VCZ_TYPE_STRING,
        .item_size = 3,
        .num_columns = 3,
        .data = data };
    char buf[1000];
    const char *expected[] = { "X,Y", "XX,YY,Z", "XXX" };
    int64_t ret;
    size_t j;

    CU_ASSERT_EQUAL_FATAL(sizeof(data), 27);

    for (j = 0; j < num_rows; j++) {
        ret = vcz_field_write(&field, j, buf, 1000, 0);
        CU_ASSERT_EQUAL_FATAL(ret, strlen(expected[j]));
        CU_ASSERT_NSTRING_EQUAL(buf, expected[j], ret);
    }
}

static void
test_variant_encoder_minimal(void)
{
    // Two rows, one column in each field, two samples
    const size_t num_rows = 2;
    const char contig_data[] = "X\0YY";
    const int32_t pos_data[] = { 123, 45678 };
    const char id_data[] = "RS1RS2";
    const char ref_data[] = "AG";
    const char alt_data[] = "T";
    const float qual_data[] = { 9, 12.1f };
    const char filter_id_data[] = "PASS\0FILT1";
    const int8_t filter_data[] = { 1, 0, 0, 1 };
    const int32_t an_data[] = { -1, 9 };
    const char *aa_data = "G.";
    const int8_t flag_data[] = { 0, 1 };
    const int32_t gt_data[] = { 0, 0, 0, 1, 1, 1, 1, 0 };
    const int8_t gt_phased_data[] = { 0, 1, 1, 0 };
    const int32_t hq_data[] = { 10, 15, 7, 12, -1, -1, -1, -1 };
    const float gl_data[] = { 1, 2, 3, 4, 1.1f, 1.2f, 1.3f, 1.4f };
    int64_t ret;
    size_t j;
    vcz_variant_encoder_t writer;
    const char *expected[] = {
        "X\t123\tRS1\tA\tT\t9\tPASS\tAA=G\tGT:HQ:GL\t0/0:10,15:1,2\t0|1:7,12:3,4",
        "YY\t45678\tRS2\tG\t.\t12.1\tFILT1\tAN=9;FLAG\tGT:GL\t1|1:1.1,1.2\t1/0:1.3,1.4",
    };
    char buf[1000];

    ret = vcz_variant_encoder_init(&writer, 2, 2, contig_data, 2, pos_data, id_data, 3,
        1, ref_data, 1, alt_data, 1, 1, qual_data, filter_id_data, 5, 2, filter_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_gt_field(&writer, gt_data, 4, 2, gt_phased_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_info_field(&writer, "AN", VCZ_TYPE_INT, 4, 1, an_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_info_field(
        &writer, "AA", VCZ_TYPE_STRING, 1, 1, aa_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_info_field(
        &writer, "FLAG", VCZ_TYPE_BOOL, 1, 1, flag_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_format_field(
        &writer, "HQ", VCZ_TYPE_INT, 4, 2, hq_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_format_field(
        &writer, "GL", VCZ_TYPE_FLOAT, 4, 2, gl_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    vcz_variant_encoder_print_state(&writer, _devnull);
    /* printf("\n"); */
    /* vcz_variant_encoder_print_state(&writer, stdout); */

    for (j = 0; j < num_rows; j++) {
        ret = vcz_variant_encoder_write_row(&writer, j, buf, 1000);
        /* printf("ret = %d\n", (int) ret); */
        /* printf("GOT:%s\n", buf); */
        /* printf("EXP:%s\n", expected[j]); */
        /* printf("GOT:%d\n", (int) strlen(buf)); */
        /* printf("EXP:%d\n", (int) strlen(expected[j])); */
        CU_ASSERT_EQUAL(ret, strlen(expected[j]));
        CU_ASSERT_STRING_EQUAL(buf, expected[j]);
    }
    vcz_variant_encoder_free(&writer);
}

static void
test_variant_encoder_fields_all_missing(void)
{
    const size_t num_rows = 1;
    const char contig_data[] = "X";
    const int32_t pos_data[] = { 123 };
    const char id_data[] = ".";
    const char ref_data[] = "A";
    const char alt_data[] = "T";
    const float qual_data[] = { 9 };
    const char filter_id_data[] = "PASS";
    const int8_t filter_data[] = { 1 };
    const int32_t an_data[] = { -1 };
    const char *aa_data = ".";
    const int8_t flag_data[] = { 0 };
    const int32_t gt_data[] = { -1, -1, -1, -1 };
    const int8_t gt_phased_data[] = { 0, 0 };
    const int32_t hq_data[] = { -1, -1, -1, -1 };
    const int32_t gl_data[] = {
        VCZ_FLOAT32_MISSING_AS_INT32,
        VCZ_FLOAT32_MISSING_AS_INT32,
        VCZ_FLOAT32_MISSING_AS_INT32,
        VCZ_FLOAT32_MISSING_AS_INT32,
    };
    int64_t ret;
    size_t j;
    vcz_variant_encoder_t writer;
    const char *expected[] = {
        "X\t123\t.\tA\tT\t9\tPASS\t.\t.\t.\t.",
    };
    char buf[1000];

    ret = vcz_variant_encoder_init(&writer, 2, num_rows, contig_data, 1, pos_data,
        id_data, 1, 1, ref_data, 1, alt_data, 1, 1, qual_data, filter_id_data, 4, 1,
        filter_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_gt_field(&writer, gt_data, 4, 2, gt_phased_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_info_field(&writer, "AN", VCZ_TYPE_INT, 4, 1, an_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_info_field(
        &writer, "AA", VCZ_TYPE_STRING, 1, 1, aa_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_info_field(
        &writer, "FLAG", VCZ_TYPE_BOOL, 1, 1, flag_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_format_field(
        &writer, "HQ", VCZ_TYPE_INT, 4, 2, hq_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_format_field(
        &writer, "GL", VCZ_TYPE_FLOAT, 4, 2, gl_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    vcz_variant_encoder_print_state(&writer, _devnull);
    /* printf("\n"); */
    /* vcz_variant_encoder_print_state(&writer, stdout); */

    for (j = 0; j < num_rows; j++) {
        ret = vcz_variant_encoder_write_row(&writer, j, buf, 1000);
        /* printf("ret = %d\n", (int) ret); */
        /* printf("GOT:%s\n", buf); */
        /* printf("EXP:%s\n", expected[j]); */
        /* printf("GOT:%d\n", (int) strlen(buf)); */
        /* printf("EXP:%d\n", (int) strlen(expected[j])); */
        CU_ASSERT_EQUAL(ret, strlen(expected[j]));
        CU_ASSERT_STRING_EQUAL(buf, expected[j]);
    }
    vcz_variant_encoder_free(&writer);
}

static void
test_itoa_small(void)
{
    char dest1[64], dest2[64];
    int len1, len2;
    int32_t j;

    for (j = -255; j <= 256; j++) {
        len1 = sprintf(dest1, "%d", j);
        len2 = vcz_itoa(dest2, j);
        /* printf("%s %s\n", dest1, dest2); */
        CU_ASSERT_STRING_EQUAL(dest1, dest2);
        CU_ASSERT_EQUAL(len1, len2);
    }
}

static void
test_itoa_pow10(void)
{
    char dest1[64], dest2[64];
    int len1, len2;
    int32_t j, k, power, value;

    for (power = 0; power < 10; power++) {
        j = (int32_t) pow(10, power);
        for (k = -1; k < 2; k++) {
            value = j + k;
            len1 = sprintf(dest1, "%d", value);
            len2 = vcz_itoa(dest2, value);
            /* printf("%s %s\n", dest1, dest2); */
            CU_ASSERT_STRING_EQUAL_FATAL(dest1, dest2);
            CU_ASSERT_EQUAL(len1, len2);
        }
    }
}

static void
test_itoa_boundary(void)
{
    char dest1[64], dest2[64];
    int len1, len2;
    size_t j;
    int32_t value;
    int32_t cases[] = { INT32_MIN, INT32_MAX };

    for (j = 0; j < sizeof(cases) / sizeof(*cases); j++) {
        value = cases[j];
        len1 = sprintf(dest1, "%d", value);
        len2 = vcz_itoa(dest2, value);
        /* printf("%s %s\n", dest1, dest2); */
        CU_ASSERT_STRING_EQUAL(dest1, dest2);
        CU_ASSERT_EQUAL(len1, len2);
    }
}

static void
test_ftoa(void)
{
    struct test_case {
        float val;
        const char *expected;
    };
    // clang-format off
    struct test_case cases[] = {
        {0.0f, "0"},
        {0.0001f, "0"},
        {0.0005f, "0.001"},
        {0.3f, "0.3"},
        {0.32f, "0.32"},
        {0.329f, "0.329"},
        {0.3217f, "0.322"},
        {8.0f, "8"},
        {8.0001f, "8"},
        {8.3f, "8.3"},
        {8.32f, "8.32"},
        {8.329f, "8.329"},
        {8.3217f, "8.322"},
        {443.998f, "443.998"},
        {1028.0f, "1028"},
        {1028.0001f, "1028"},
        {1028.3f, "1028.3"},
        {1028.32f, "1028.32"},
        {1028.329f, "1028.329"},
        {1028.3217f, "1028.322"},
        {1000000, "1000000"},
        {-100.0f, "-100"},
        {NAN, "nan"},
        {INFINITY, "inf"},
        {-INFINITY, "-inf"},
        {2311380, "2311380"},
        {16777216, "16777216"}, /* Maximum integer value of float */
        {-16777216, "-16777216"},
        {INT32_MIN, "-2147483648"},
        {(float) INT32_MAX, "2147483648"},
        {(float) DBL_MAX, "inf",},
        {(float) DBL_MIN, "0",},
        {FLT_MIN, "0",},
        /* TODO test extreme value here, that push against the limits of f32 */
        // FAILS https://github.com/jeromekelleher/vcztools/issues/21
        /* {FLT_MAX, "340282346638528859811704183484516925440",}, */
    };
    // clang-format on
    int len;
    size_t j;
    char buf[1024];

    for (j = 0; j < sizeof(cases) / sizeof(*cases); j++) {
        len = vcz_ftoa(buf, cases[j].val);
        /* printf("j = %d %f->%s=='%s'\n", (int) j, cases[j].val, cases[j].expected,
         * buf); */
        CU_ASSERT_EQUAL_FATAL(len, strlen(cases[j].expected));
        CU_ASSERT_STRING_EQUAL_FATAL(buf, cases[j].expected);
    }
}

/*=================================================
  Test suite management
  =================================================
*/

static int
vcz_suite_init(void)
{
    _devnull = fopen("/dev/null", "w");
    if (_devnull == NULL) {
        return CUE_SINIT_FAILED;
    }
    return CUE_SUCCESS;
}

static int
vcz_suite_cleanup(void)
{
    if (_devnull != NULL) {
        fclose(_devnull);
    }
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
            .pName = "vcz",
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
        suite = CU_get_suite_by_name("vcz", CU_get_registry());
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
        { "test_string_field_1d", test_string_field_1d },
        { "test_string_field_1d_overflow", test_string_field_1d_overflow },
        { "test_string_field_2d", test_string_field_2d },
        { "test_int_field_1d", test_int_field_1d },
        { "test_int_field_1d_overflow", test_int_field_1d_overflow },
        { "test_int_field_2d", test_int_field_2d },
        { "test_float_field_1d", test_float_field_1d },
        { "test_float_field_1d_overflow", test_float_field_1d_overflow },
        { "test_variant_encoder_minimal", test_variant_encoder_minimal },
        { "test_variant_fields_all_missing", test_variant_encoder_fields_all_missing },
        { "test_itoa_small", test_itoa_small },
        { "test_itoa_pow10", test_itoa_pow10 },
        { "test_itoa_boundary", test_itoa_boundary },
        { "test_ftoa", test_ftoa },
        { NULL, NULL },
    };
    return test_main(tests, argc, argv);
}
