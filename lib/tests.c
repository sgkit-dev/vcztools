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
validate_field(const vcz_field_t *field, size_t num_rows, const char **expected)
{
    char *buf;
    int64_t ret, buflen;
    size_t j;

    for (j = 0; j < num_rows; j++) {
        /* printf("expected: %s\n", expected[j]); */
        for (buflen = 0; buflen < (int64_t) strlen(expected[j]); buflen++) {
            buf = malloc((size_t) buflen);
            CU_ASSERT_FATAL(buf != NULL);
            ret = vcz_field_write_1d(field, j, buf, buflen, 0);
            free(buf);
            CU_ASSERT_FATAL(ret == VCZ_ERR_BUFFER_OVERFLOW);
        }
        buflen = (int64_t) strlen(expected[j]);
        buf = malloc((size_t) buflen);
        CU_ASSERT_FATAL(buf != NULL);
        ret = vcz_field_write_1d(field, j, buf, buflen, 0);
        /* printf("ret = %d\n", (int)ret); */
        /* printf("'%.*s': %s\n", (int) ret, buf, expected[j]); */
        CU_ASSERT_EQUAL_FATAL(ret, strlen(expected[j]));
        CU_ASSERT_NSTRING_EQUAL(buf, expected[j], ret);
        free(buf);
    }
}

static void
validate_encoder(
    const vcz_variant_encoder_t *encoder, size_t num_rows, const char **expected)
{
    char *buf;
    int64_t ret, buflen, min_len;
    size_t j;

    vcz_variant_encoder_print_state(encoder, _devnull);
    /* printf("\n"); */
    /* vcz_variant_encoder_print_state(encoder, stdout); */

    for (j = 0; j < num_rows; j++) {
        /* We need space for the NULL byte as well */
        min_len = (int64_t) strlen(expected[j]) + 1;
        /* printf("expected: %s\n", expected[j]); */

        for (buflen = 0; buflen < min_len; buflen++) {
            buf = malloc((size_t) buflen);
            CU_ASSERT_FATAL(buf != NULL);
            ret = vcz_variant_encoder_encode(encoder, j, buf, (size_t) buflen);
            free(buf);
            CU_ASSERT_FATAL(ret == VCZ_ERR_BUFFER_OVERFLOW);
        }
        buflen = min_len;
        buf = malloc((size_t) buflen);
        CU_ASSERT_FATAL(buf != NULL);
        ret = vcz_variant_encoder_encode(encoder, j, buf, (size_t) buflen);
        /* printf("ret = %d\n", (int) ret); */
        /* printf("GOT:'%s'\n", buf); */
        /* printf("EXP:'%s'\n", expected[j]); */
        /* printf("GOT:%d\n", (int) strlen(buf)); */
        /* printf("EXP:%d\n", (int) strlen(expected[j])); */
        /* int64_t c; */
        /* for (c = 0; c < ret; c++) { */
        /*     if (buf[c] != expected[j][c]) { */
        /*         printf("Mismatch at %d: %c != %c\n", (int) c, buf[c], expected[j][c]);
         */

        /*     } */
        /* } */
        CU_ASSERT_EQUAL_FATAL(ret, strlen(expected[j]));
        CU_ASSERT_NSTRING_EQUAL_FATAL(buf, expected[j], ret);
        free(buf);
    }
    for (j = num_rows; j < num_rows + 5; j++) {
        ret = vcz_variant_encoder_encode(encoder, j, buf, 0);
        CU_ASSERT_EQUAL_FATAL(ret, VCZ_ERR_VARIANT_OUT_OF_BOUNDS);
    }
}

static void
test_field_name_too_long(void)
{
    vcz_field_t field;
    char *long_name = malloc(VCZ_MAX_FIELD_NAME_LEN + 2);
    int ret;

    CU_ASSERT_FATAL(long_name != NULL);
    memset(long_name, 'A', VCZ_MAX_FIELD_NAME_LEN + 1);
    long_name[VCZ_MAX_FIELD_NAME_LEN + 1] = '\0';
    CU_ASSERT_EQUAL_FATAL(strlen(long_name), VCZ_MAX_FIELD_NAME_LEN + 1);
    ret = vcz_field_init(&field, long_name, VCZ_TYPE_INT, 1, 1, NULL);
    CU_ASSERT_EQUAL(ret, VCZ_ERR_FIELD_NAME_TOO_LONG);

    long_name[VCZ_MAX_FIELD_NAME_LEN] = '\0';
    CU_ASSERT_EQUAL_FATAL(strlen(long_name), VCZ_MAX_FIELD_NAME_LEN);
    ret = vcz_field_init(&field, long_name, VCZ_TYPE_INT, 4, 1, NULL);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(field.name_length, VCZ_MAX_FIELD_NAME_LEN);
    free(long_name);
}

static void
test_field_bad_type(void)
{
    vcz_field_t field;
    int cases[] = { -1, 0, 5, 100 };
    int ret;
    size_t j;

    for (j = 0; j < sizeof(cases) / sizeof(*cases); j++) {
        ret = vcz_field_init(&field, "NAME", cases[j], 1, 1, NULL);
        CU_ASSERT_EQUAL(ret, VCZ_ERR_FIELD_UNSUPPORTED_TYPE);
    }
}

static void
test_field_bad_item_size(void)
{
    vcz_field_t field;
    struct test_case {
        int type;
        size_t item_size;
    };
    struct test_case cases[] = {
        { VCZ_TYPE_INT, 0 },
        { VCZ_TYPE_BOOL, 0 },
        { VCZ_TYPE_STRING, 0 },
        { VCZ_TYPE_FLOAT, 0 },
        { VCZ_TYPE_INT, 3 },
        { VCZ_TYPE_INT, 5 },
        { VCZ_TYPE_INT, 6 },
        { VCZ_TYPE_INT, 7 },
        { VCZ_TYPE_INT, 8 },
        { VCZ_TYPE_INT, 100 },
        { VCZ_TYPE_FLOAT, 1 },
        { VCZ_TYPE_FLOAT, 2 },
        { VCZ_TYPE_FLOAT, 3 },
        { VCZ_TYPE_FLOAT, 7 },
        { VCZ_TYPE_FLOAT, 8 },
        { VCZ_TYPE_FLOAT, 100 },
        { VCZ_TYPE_BOOL, 2 },
        { VCZ_TYPE_BOOL, 3 },
        { VCZ_TYPE_BOOL, 4 },
        { VCZ_TYPE_BOOL, 5 },
        { VCZ_TYPE_BOOL, 6 },
        { VCZ_TYPE_BOOL, 8 },
        { VCZ_TYPE_BOOL, 100 },
    };
    int ret;
    size_t j;

    for (j = 0; j < sizeof(cases) / sizeof(*cases); j++) {
        ret = vcz_field_init(&field, "NAME", cases[j].type, cases[j].item_size, 1, NULL);
        CU_ASSERT_EQUAL(ret, VCZ_ERR_FIELD_UNSUPPORTED_ITEM_SIZE);
    }
}

static void
test_field_bad_num_columns(void)
{
    vcz_field_t field;
    struct test_case {
        int type;
        size_t num_columns;
    };
    struct test_case cases[] = {
        { VCZ_TYPE_INT, 0 },
        { VCZ_TYPE_BOOL, 0 },
        { VCZ_TYPE_STRING, 0 },
        { VCZ_TYPE_FLOAT, 0 },
    };
    int ret;
    size_t j;

    for (j = 0; j < sizeof(cases) / sizeof(*cases); j++) {
        ret = vcz_field_init(
            &field, "NAME", cases[j].type, 1, cases[j].num_columns, NULL);
        CU_ASSERT_EQUAL(ret, VCZ_ERR_FIELD_UNSUPPORTED_NUM_COLUMNS);
    }
}

static void
test_int8_field_1d(void)
{
    const int8_t data[] = { 1, 2, 127, -1, -100 };
    const char *expected[] = { "1", "2", "127", ".", "-100" };
    vcz_field_t field = { .name = "test",
        .type = VCZ_TYPE_INT,
        .item_size = 1,
        .num_columns = 1,
        .data = (const char *) data };

    validate_field(&field, sizeof(data) / sizeof(*data), expected);
}

static void
test_int8_field_2d(void)
{
    const int8_t data[] = { 1, 2, 3, 123, 127, -2, -1, -2, -2, -2, -2, -2 };
    vcz_field_t field = { .name = "test",
        .type = VCZ_TYPE_INT,
        .item_size = 1,
        .num_columns = 3,
        .data = (const char *) data };
    const char *expected[] = { "1,2,3", "123,127", ".", "" };

    validate_field(&field, 4, expected);
}

static void
test_int16_field_1d(void)
{
    const int16_t data[] = { 1, 2, 127, -1, -100 };
    const char *expected[] = { "1", "2", "127", ".", "-100" };
    vcz_field_t field = { .name = "test",
        .type = VCZ_TYPE_INT,
        .item_size = 2,
        .num_columns = 1,
        .data = (const char *) data };

    validate_field(&field, sizeof(data) / sizeof(*data), expected);
}

static void
test_int16_field_2d(void)
{
    const int16_t data[] = { 1, 2, 3, 123, 127, -2, -1, -2, -2, -2, -2, -2 };
    vcz_field_t field = { .name = "test",
        .type = VCZ_TYPE_INT,
        .item_size = 2,
        .num_columns = 3,
        .data = (const char *) data };
    const char *expected[] = { "1,2,3", "123,127", ".", "" };

    validate_field(&field, 4, expected);
}

static void
test_int32_field_1d(void)
{
    const int32_t data[] = { 1, 2, 12345789, -1, -100 };
    const char *expected[] = { "1", "2", "12345789", ".", "-100" };
    vcz_field_t field = { .name = "test",
        .type = VCZ_TYPE_INT,
        .item_size = 4,
        .num_columns = 1,
        .data = (const char *) data };

    validate_field(&field, 5, expected);
}

static void
test_int32_field_2d(void)
{
    const int32_t data[] = { 1, 2, 3, 1234, 5678, -2, -1, -2, -2, -2, -2, -2 };
    vcz_field_t field = { .name = "test",
        .type = VCZ_TYPE_INT,
        .item_size = 4,
        .num_columns = 3,
        .data = (const char *) data };
    const char *expected[] = { "1,2,3", "1234,5678", ".", "" };

    validate_field(&field, 4, expected);
}

static void
test_float_field_1d(void)
{
    float data[] = { 0, 1.0f, 2.1f, INT32_MIN, 12345789.0f, -1, -100.123f, FLT_MAX };

    const size_t num_rows = sizeof(data) / sizeof(*data);
    vcz_field_t field = { .name = "test",
        .type = VCZ_TYPE_FLOAT,
        .item_size = 4,
        .num_columns = 1,
        .data = (const char *) data };
    const char *expected[] = { ".", "1", "2.1", "-2147483648", "12345789", "-1",
        "-100.123", "340282346638528859811704183484516925440.000" };
    int32_t *int_data = (int32_t *) data;

    int_data[0] = VCZ_FLOAT32_MISSING_AS_INT32;

    validate_field(&field, num_rows, expected);
}

static void
test_float_field_2d(void)
{
    // clang-format off
    int32_t data[] = {
        VCZ_FLOAT32_MISSING_AS_INT32, VCZ_FLOAT32_MISSING_AS_INT32, VCZ_FLOAT32_MISSING_AS_INT32,
        VCZ_FLOAT32_MISSING_AS_INT32, VCZ_FLOAT32_MISSING_AS_INT32, VCZ_FLOAT32_FILL_AS_INT32,
        VCZ_FLOAT32_MISSING_AS_INT32, VCZ_FLOAT32_FILL_AS_INT32, VCZ_FLOAT32_FILL_AS_INT32,
        VCZ_FLOAT32_FILL_AS_INT32, VCZ_FLOAT32_FILL_AS_INT32, VCZ_FLOAT32_FILL_AS_INT32,
        VCZ_FLOAT32_FILL_AS_INT32, VCZ_FLOAT32_FILL_AS_INT32, VCZ_FLOAT32_FILL_AS_INT32
    };
    // clang-format on
    const char *expected[] = { ".,.,.", ".,.", ".", "1.25,3", "3" };
    vcz_field_t field = { .name = "test",
        .type = VCZ_TYPE_FLOAT,
        .item_size = 4,
        .num_columns = 3,
        .data = (const char *) data };
    float *float_data = (float *) data;

    float_data[9] = 1.25;
    float_data[10] = 3;
    float_data[12] = 3;

    validate_field(&field, 5, expected);
}

static void
test_string_field_1d(void)
{
    const char data[] = "X\0\0"  /* X */
                        "XX\0"   /* XX*/
                        "XXX"    /* XXX, */
                        ".\0\0"; /*. */

    vcz_field_t field = { .name = "test",
        .type = VCZ_TYPE_STRING,
        .item_size = 3,
        .num_columns = 1,
        .data = data };
    const char *expected[] = { "X", "XX", "XXX", "." };

    CU_ASSERT_EQUAL_FATAL(sizeof(data), 13);
    validate_field(&field, 4, expected);
}

static void
test_string_field_2d(void)
{
    const char data[] = "X\0\0Y\0\0\0\0\0" /* [X, Y] */
                        "XX\0YY\0Z\0\0"    /* [XX, YY, Z], */
                        ".\0\0.\0\0\0\0\0" /* [., .], */
                        "XXX\0\0\0\0\0";   /* [XXX], */
    vcz_field_t field = { .name = "test",
        .type = VCZ_TYPE_STRING,
        .item_size = 3,
        .num_columns = 3,
        .data = data };
    const char *expected[] = { "X,Y", "XX,YY,Z", ".,.", "XXX" };

    CU_ASSERT_EQUAL_FATAL(sizeof(data), 36);
    validate_field(&field, 4, expected);
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
    vcz_variant_encoder_t writer;
    const char *expected[] = {
        "X\t123\tRS1\tA\tT\t9\tPASS\tAA=G\tGT:HQ:GL\t0/0:10,15:1,2\t0|1:7,12:3,4",
        "YY\t45678\tRS2\tG\t.\t12.1\tFILT1\tAN=9;FLAG\tGT:GL\t1|1:1.1,1.2\t1/0:1.3,1.4",
    };

    ret = vcz_variant_encoder_init(&writer, 2, 2);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_chrom_field(&writer, 2, contig_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_pos_field(&writer, pos_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_id_field(&writer, 3, 1, id_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_ref_field(&writer, 1, ref_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_alt_field(&writer, 1, 1, alt_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_qual_field(&writer, qual_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_filter_field(
        &writer, 5, 2, filter_id_data, filter_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_gt_field(&writer, 4, 2, gt_data, gt_phased_data);
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

    validate_encoder(&writer, num_rows, expected);
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
    const int8_t filter_data[] = { 0 };
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
    vcz_variant_encoder_t writer;
    const char *expected[] = {
        "X\t123\t.\tA\tT\t9\t.\t.\t.\t.\t.",
    };

    ret = vcz_variant_encoder_init(&writer, num_rows, 2);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_chrom_field(&writer, 1, contig_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_pos_field(&writer, pos_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_id_field(&writer, 1, 1, id_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_ref_field(&writer, 1, ref_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_alt_field(&writer, 1, 1, alt_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_qual_field(&writer, qual_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_filter_field(
        &writer, 4, 1, filter_id_data, filter_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_gt_field(&writer, 4, 2, gt_data, gt_phased_data);
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

    validate_encoder(&writer, num_rows, expected);
    vcz_variant_encoder_free(&writer);
}

/* NOTE: the duplication in the next three tests is pretty ugly, but it's
 * an effective way to make sure we are definely using the correct size
 * pointers. We can always write other tests that are not so ugly. */
static void
test_variant_encoder_int8_fields(void)
{
    const size_t num_rows = 4;
    const size_t num_samples = 1;
    const char contig_data[] = "1234";
    const int32_t pos_data[] = { 1, 2, 3, 4 };
    const char id_data[] = "1234";
    const char ref_data[] = "1234";
    const char alt_data[] = "1234";
    const float qual_data[] = { 1, 2, 3, 4 };
    const char filter_id_data[] = "PASS";
    const int8_t filter_data[] = { 1, 1, 1, 1 };
    const int8_t gt_data[] = { 0, 1, 2, 3, -1, 5, -1, -1 };
    const int8_t gt_phased_data[] = { 1, 1, 1, 1, 1, 1, 1, 1 };
    const int8_t ii1_data[] = { 1, 2, 3, -1 };
    const int8_t if1_data[] = { 1, 2, 3, -1 };
    const int8_t ii2_data[] = { 1, 2, 3, 4, 5, -1, -1, -1 };
    const int8_t if2_data[] = { 1, 2, 3, 4, -1, 6, -1, -1 };
    int64_t ret;
    vcz_variant_encoder_t writer;
    const char *expected[] = {
        "1\t1\t1\t1\t1\t1\tPASS\tII1=1;II2=1,2\tGT:IF1:IF2\t0|1:1:1,2",
        "2\t2\t2\t2\t2\t2\tPASS\tII1=2;II2=3,4\tGT:IF1:IF2\t2|3:2:3,4",
        "3\t3\t3\t3\t3\t3\tPASS\tII1=3;II2=5,.\tGT:IF1:IF2\t.|5:3:.,6",
        "4\t4\t4\t4\t4\t4\tPASS\t.\t.\t.",
    };

    ret = vcz_variant_encoder_init(&writer, num_rows, num_samples);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_chrom_field(&writer, 1, contig_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_pos_field(&writer, pos_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_id_field(&writer, 1, 1, id_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_ref_field(&writer, 1, ref_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_alt_field(&writer, 1, 1, alt_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_qual_field(&writer, qual_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_filter_field(
        &writer, 4, 1, filter_id_data, filter_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = vcz_variant_encoder_add_gt_field(&writer, 1, 2, gt_data, gt_phased_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_info_field(
        &writer, "II1", VCZ_TYPE_INT, 1, 1, ii1_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_info_field(
        &writer, "II2", VCZ_TYPE_INT, 1, 2, ii2_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_format_field(
        &writer, "IF1", VCZ_TYPE_INT, 1, 1, if1_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_format_field(
        &writer, "IF2", VCZ_TYPE_INT, 1, 2, if2_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    validate_encoder(&writer, num_rows, expected);
    vcz_variant_encoder_free(&writer);
}

static void
test_variant_encoder_int16_fields(void)
{
    const size_t num_rows = 4;
    const size_t num_samples = 1;
    const char contig_data[] = "1234";
    const int32_t pos_data[] = { 1, 2, 3, 4 };
    const char id_data[] = "1234";
    const char ref_data[] = "1234";
    const char alt_data[] = "1234";
    const float qual_data[] = { 1, 2, 3, 4 };
    const char filter_id_data[] = "PASS";
    const int8_t filter_data[] = { 1, 1, 1, 1 };
    const int8_t gt_phased_data[] = { 1, 1, 1, 1, 1, 1, 1, 1 };
    const int16_t gt_data[] = { 0, 1, 2, 3, -1, 5, -1, -1 };
    const int16_t ii1_data[] = { 1, 2, 3, -1 };
    const int16_t if1_data[] = { 1, 2, 3, -1 };
    const int16_t ii2_data[] = { 1, 2, 3, 4, 5, -1, -1, -1 };
    const int16_t if2_data[] = { 1, 2, 3, 4, -1, 6, -1, -1 };
    int64_t ret;
    vcz_variant_encoder_t writer;
    const char *expected[] = {
        "1\t1\t1\t1\t1\t1\tPASS\tII1=1;II2=1,2\tGT:IF1:IF2\t0|1:1:1,2",
        "2\t2\t2\t2\t2\t2\tPASS\tII1=2;II2=3,4\tGT:IF1:IF2\t2|3:2:3,4",
        "3\t3\t3\t3\t3\t3\tPASS\tII1=3;II2=5,.\tGT:IF1:IF2\t.|5:3:.,6",
        "4\t4\t4\t4\t4\t4\tPASS\t.\t.\t.",
    };

    ret = vcz_variant_encoder_init(&writer, num_rows, num_samples);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_chrom_field(&writer, 1, contig_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_pos_field(&writer, pos_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_id_field(&writer, 1, 1, id_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_ref_field(&writer, 1, ref_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_alt_field(&writer, 1, 1, alt_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_qual_field(&writer, qual_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_filter_field(
        &writer, 4, 1, filter_id_data, filter_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = vcz_variant_encoder_add_gt_field(&writer, 2, 2, gt_data, gt_phased_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_info_field(
        &writer, "II1", VCZ_TYPE_INT, 2, 1, ii1_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_info_field(
        &writer, "II2", VCZ_TYPE_INT, 2, 2, ii2_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_format_field(
        &writer, "IF1", VCZ_TYPE_INT, 2, 1, if1_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_format_field(
        &writer, "IF2", VCZ_TYPE_INT, 2, 2, if2_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    validate_encoder(&writer, num_rows, expected);
    vcz_variant_encoder_free(&writer);
}

static void
test_variant_encoder_int32_fields(void)
{
    const size_t num_rows = 4;
    const size_t num_samples = 1;
    const char contig_data[] = "1234";
    const int32_t pos_data[] = { 1, 2, 3, 4 };
    const char id_data[] = "1234";
    const char ref_data[] = "1234";
    const char alt_data[] = "1234";
    const float qual_data[] = { 1, 2, 3, 4 };
    const char filter_id_data[] = "PASS";
    const int8_t filter_data[] = { 1, 1, 1, 1 };
    const int8_t gt_phased_data[] = { 1, 1, 1, 1, 1, 1, 1, 1 };
    const int32_t gt_data[] = { 0, 1, 2, 3, -1, 5, -1, -1 };
    const int32_t ii1_data[] = { 1, 2, 3, -1 };
    const int32_t if1_data[] = { 1, 2, 3, -1 };
    const int32_t ii2_data[] = { 1, 2, 3, 4, 5, -1, -1, -1 };
    const int32_t if2_data[] = { 1, 2, 3, 4, -1, 6, -1, -1 };
    int64_t ret;
    vcz_variant_encoder_t writer;
    const char *expected[] = {
        "1\t1\t1\t1\t1\t1\tPASS\tII1=1;II2=1,2\tGT:IF1:IF2\t0|1:1:1,2",
        "2\t2\t2\t2\t2\t2\tPASS\tII1=2;II2=3,4\tGT:IF1:IF2\t2|3:2:3,4",
        "3\t3\t3\t3\t3\t3\tPASS\tII1=3;II2=5,.\tGT:IF1:IF2\t.|5:3:.,6",
        "4\t4\t4\t4\t4\t4\tPASS\t.\t.\t.",
    };

    ret = vcz_variant_encoder_init(&writer, num_rows, num_samples);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_chrom_field(&writer, 1, contig_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_pos_field(&writer, pos_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_id_field(&writer, 1, 1, id_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_ref_field(&writer, 1, ref_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_alt_field(&writer, 1, 1, alt_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_qual_field(&writer, qual_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_filter_field(
        &writer, 4, 1, filter_id_data, filter_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = vcz_variant_encoder_add_gt_field(&writer, 4, 2, gt_data, gt_phased_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_info_field(
        &writer, "II1", VCZ_TYPE_INT, 4, 1, ii1_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_info_field(
        &writer, "II2", VCZ_TYPE_INT, 4, 2, ii2_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_format_field(
        &writer, "IF1", VCZ_TYPE_INT, 4, 1, if1_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = vcz_variant_encoder_add_format_field(
        &writer, "IF2", VCZ_TYPE_INT, 4, 2, if2_data);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    validate_encoder(&writer, num_rows, expected);
    vcz_variant_encoder_free(&writer);
}

static void
test_variant_encoder_bad_fields(void)
{
    vcz_variant_encoder_t writer;
    int ret;

    ret = vcz_variant_encoder_init(&writer, 0, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = vcz_variant_encoder_add_info_field(&writer, "FIELD", 0, 1, 1, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, VCZ_ERR_FIELD_UNSUPPORTED_TYPE);
    ret = vcz_variant_encoder_add_info_field(&writer, "FIELD", VCZ_TYPE_INT, 3, 1, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, VCZ_ERR_FIELD_UNSUPPORTED_ITEM_SIZE);
    ret = vcz_variant_encoder_add_info_field(&writer, "FIELD", VCZ_TYPE_INT, 4, 0, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, VCZ_ERR_FIELD_UNSUPPORTED_NUM_COLUMNS);
    CU_ASSERT_EQUAL_FATAL(writer.num_info_fields, 0);

    ret = vcz_variant_encoder_add_format_field(&writer, "FIELD", 0, 1, 1, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, VCZ_ERR_FIELD_UNSUPPORTED_TYPE);
    ret = vcz_variant_encoder_add_format_field(
        &writer, "FIELD", VCZ_TYPE_INT, 3, 1, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, VCZ_ERR_FIELD_UNSUPPORTED_ITEM_SIZE);
    ret = vcz_variant_encoder_add_format_field(
        &writer, "FIELD", VCZ_TYPE_INT, 4, 0, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, VCZ_ERR_FIELD_UNSUPPORTED_NUM_COLUMNS);
    CU_ASSERT_EQUAL_FATAL(writer.num_format_fields, 0);

    vcz_variant_encoder_free(&writer);
}

static void
test_variant_encoder_many_fields(void)
{
    vcz_variant_encoder_t writer;
    int ret;
    size_t j;

    ret = vcz_variant_encoder_init(&writer, 0, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    /* We're basically just testing the realloc behavior here and making sure we don't
     * leak memory */
    for (j = 0; j < 3 * writer.field_array_size_increment; j++) {
        /* We don't check for uniqueness of names */
        ret = vcz_variant_encoder_add_info_field(
            &writer, "FIELD", VCZ_TYPE_INT, 4, 1, NULL);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        /* vcz_variant_encoder_print_state(&writer, stdout); */
        CU_ASSERT_EQUAL_FATAL(writer.num_info_fields, j + 1);
        ret = vcz_variant_encoder_add_format_field(
            &writer, "FIELD", VCZ_TYPE_INT, 4, 1, NULL);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL_FATAL(writer.num_format_fields, j + 1);
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
        {2 * (float) INT32_MAX, "4294967296.000"},
        {(float) DBL_MAX, "inf",},
        {(float) DBL_MIN, "0",},
        {FLT_MIN, "0",},
        {FLT_MAX, "340282346638528859811704183484516925440.000",},
        {-FLT_MAX, "-340282346638528859811704183484516925440.000",},
    };
    // clang-format on
    int len;
    size_t j;
    char buf[1024];

    for (j = 0; j < sizeof(cases) / sizeof(*cases); j++) {
        len = vcz_ftoa(buf, cases[j].val);
        /* printf("j = %d %f->%s=='%s'\n", (int) j, cases[j].val, cases[j].expected, buf); */
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
        { "test_field_name_too_long", test_field_name_too_long },
        { "test_field_bad_type", test_field_bad_type },
        { "test_field_bad_item_size", test_field_bad_item_size },
        { "test_field_bad_num_columns", test_field_bad_num_columns },
        { "test_int8_field_1d", test_int8_field_1d },
        { "test_int8_field_2d", test_int8_field_2d },
        { "test_int16_field_1d", test_int16_field_1d },
        { "test_int16_field_2d", test_int16_field_2d },
        { "test_int32_field_1d", test_int32_field_1d },
        { "test_int32_field_2d", test_int32_field_2d },
        { "test_float_field_1d", test_float_field_1d },
        { "test_float_field_2d", test_float_field_2d },
        { "test_string_field_1d", test_string_field_1d },
        { "test_string_field_2d", test_string_field_2d },
        { "test_variant_encoder_minimal", test_variant_encoder_minimal },
        { "test_variant_encoder_fields_all_missing",
            test_variant_encoder_fields_all_missing },
        { "test_variant_encoder_int8_fields", test_variant_encoder_int8_fields },
        { "test_variant_encoder_int16_fields", test_variant_encoder_int16_fields },
        { "test_variant_encoder_int32_fields", test_variant_encoder_int32_fields },
        { "test_variant_encoder_bad_fields", test_variant_encoder_bad_fields },
        { "test_variant_encoder_many_fields", test_variant_encoder_many_fields },
        { "test_itoa_small", test_itoa_small },
        { "test_itoa_pow10", test_itoa_pow10 },
        { "test_itoa_boundary", test_itoa_boundary },
        { "test_ftoa", test_ftoa },
        { NULL, NULL },
    };
    return test_main(tests, argc, argv);
}
