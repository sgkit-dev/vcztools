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
        /* printf("expected: %s\n", expected[j]); */

        /* We need space for the NULL byte as well */
        min_len = (int64_t) strlen(expected[j]) + 1;
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
        /*
        printf("ret = %d\n", (int) ret);
        printf("GOT:'%s'\n", buf);
        printf("EXP:'%s'\n", expected[j]);
        printf("GOT:%d\n", (int) strlen(buf));
        printf("EXP:%d\n", (int) strlen(expected[j]));
        int64_t c;
        for (c = 0; c < ret; c++) {
            if (buf[c] != expected[j][c]) {
                printf("Mismatch at %d: %c != %c\n", (int) c, buf[c], expected[j][c]);

            }
        }
        */
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
    const int8_t filter_data[] = { 1, 1, 0, 1 };
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
        "X\t123\tRS1\tA\tT\t9\tPASS;FILT1\tAA=G\tGT:HQ:GL\t0/0:10,15:1,2\t0|1:7,12:3,4",
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
    /* Flag fields must have itemsize 1, and 1 column. */
    ret = vcz_variant_encoder_add_info_field(
        &writer, "FIELD", VCZ_TYPE_BOOL, 1, 2, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, VCZ_ERR_FIELD_UNSUPPORTED_NUM_COLUMNS);
    ret = vcz_variant_encoder_add_info_field(
        &writer, "FIELD", VCZ_TYPE_BOOL, 2, 1, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, VCZ_ERR_FIELD_UNSUPPORTED_ITEM_SIZE);

    CU_ASSERT_EQUAL_FATAL(writer.num_info_fields, 0);

    ret = vcz_variant_encoder_add_format_field(&writer, "FIELD", 0, 1, 1, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, VCZ_ERR_FIELD_UNSUPPORTED_TYPE);
    ret = vcz_variant_encoder_add_format_field(
        &writer, "FIELD", VCZ_TYPE_INT, 3, 1, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, VCZ_ERR_FIELD_UNSUPPORTED_ITEM_SIZE);
    ret = vcz_variant_encoder_add_format_field(
        &writer, "FIELD", VCZ_TYPE_INT, 4, 0, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, VCZ_ERR_FIELD_UNSUPPORTED_NUM_COLUMNS);
    /* Cannot have flags format fields */
    ret = vcz_variant_encoder_add_format_field(
        &writer, "FIELD", VCZ_TYPE_BOOL, 1, 1, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, VCZ_ERR_FIELD_UNSUPPORTED_TYPE);

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
        /* printf("j = %d %f->%s=='%s'\n", (int) j, cases[j].val, cases[j].expected,
         * buf); */
        CU_ASSERT_EQUAL_FATAL(len, strlen(cases[j].expected));
        CU_ASSERT_STRING_EQUAL_FATAL(buf, cases[j].expected);
    }
}

static void
test_encode_plink_single_genotype(void)
{
    struct test_case {
        int8_t genotype[2];
        char expected;
    };
    // clang-format off
    struct test_case cases[] = {
        {{-1, -1}, VCZ_PLINK_MISSING},
        {{-2, -1}, VCZ_PLINK_MISSING},
        {{-1, -2}, VCZ_PLINK_MISSING},
        {{-2, -2}, VCZ_PLINK_MISSING},
        /* Unknown alleles are treated as missing */
        {{2, 2}, VCZ_PLINK_MISSING},
        {{-1, 2}, VCZ_PLINK_MISSING},
        {{2, -1}, VCZ_PLINK_MISSING},
        /* Half-calls are homozygous */
        {{0, -2}, VCZ_PLINK_HOM_A2},
        {{1, -2}, VCZ_PLINK_HOM_A1},
        /* Nominal cases */
        {{1, 0}, VCZ_PLINK_HET},
        {{0, 1}, VCZ_PLINK_HET},
        {{0, 0}, VCZ_PLINK_HOM_A2},
        {{1, 1}, VCZ_PLINK_HOM_A1},
    };
    // clang-format on
    size_t j;
    char buf;

    for (j = 0; j < sizeof(cases) / sizeof(*cases); j++) {
        vcz_encode_plink(1, 1, cases[j].genotype, &buf);
        CU_ASSERT_EQUAL_FATAL(buf, cases[j].expected);
    }
}

static void
test_encode_plink_example(void)
{

    const size_t num_variants = 3;
    const size_t num_samples = 3;
    int j;
    // clang-format off
    int8_t genotypes[] = {
        0, 0, 0, 1, 0, 0,
        1, 0, 1, 1, 0, -2,
        1, 1, 0, 1, -1,-1,
    };
    // clang-format on
    int8_t expected[] = { 59, 50, 24 };
    char buf[3];

    vcz_encode_plink(num_variants, num_samples, genotypes, buf);
    for (j = 0; j < 3; j++) {
        CU_ASSERT_EQUAL_FATAL(buf[j], expected[j]);
    }
}

static void
test_encode_plink_all_zeros_instance(size_t num_variants, size_t num_samples)
{
    int8_t *genotypes = calloc(num_variants * num_samples, 2);
    char *buf = malloc(num_variants * num_samples / 4);
    size_t j;

    CU_ASSERT_FATAL(num_samples % 4 == 0);
    CU_ASSERT_FATAL(genotypes != NULL);
    CU_ASSERT_FATAL(buf != NULL);

    /* All-zero genotypes encode as HOM_A2 (REF/REF) -> 0xff per byte. */
    vcz_encode_plink(num_variants, num_samples, genotypes, buf);
    for (j = 0; j < num_variants * num_samples / 4; j++) {
        CU_ASSERT_EQUAL_FATAL(buf[j], -1);
    }

    free(genotypes);
    free(buf);
}

static void
test_encode_plink_all_zeros(void)
{
    test_encode_plink_all_zeros_instance(1, 4);
    test_encode_plink_all_zeros_instance(10, 4);
    test_encode_plink_all_zeros_instance(1, 400);
    test_encode_plink_all_zeros_instance(100, 400);
}

/* Diploid single-sample test vectors: both alleles non-(-2). The kernel
 * emits two probability bytes per sample for these (K=2). */
struct bgen_single_diploid_case {
    int8_t a;
    int8_t b;
    uint8_t ploidy;
    uint8_t b0_unphased;
    uint8_t b1_unphased;
    uint8_t b0_phased;
    uint8_t b1_phased;
};

static struct bgen_single_diploid_case bgen_single_diploid_cases[] = {
    /* a, b,  ploidy,                            B0u,  B1u,  B0p,  B1p */
    { 0, 0, VCZ_BGEN_PLOIDY_DIPLOID, 0xFF, 0x00, 0xFF, 0xFF },
    { 0, 1, VCZ_BGEN_PLOIDY_DIPLOID, 0x00, 0xFF, 0xFF, 0x00 },
    { 1, 0, VCZ_BGEN_PLOIDY_DIPLOID, 0x00, 0xFF, 0x00, 0xFF },
    { 1, 1, VCZ_BGEN_PLOIDY_DIPLOID, 0x00, 0x00, 0x00, 0x00 },
    /* -1 on either diploid allele -> missing diploid, prob bytes zeroed. */
    { -1, 0, VCZ_BGEN_PLOIDY_MISSING_DIPLOID, 0x00, 0x00, 0x00, 0x00 },
    { 0, -1, VCZ_BGEN_PLOIDY_MISSING_DIPLOID, 0x00, 0x00, 0x00, 0x00 },
    { -1, -1, VCZ_BGEN_PLOIDY_MISSING_DIPLOID, 0x00, 0x00, 0x00, 0x00 },
};

/* Haploid single-sample test vectors: slot 1 = -2 sentinel. The kernel
 * emits one probability byte per sample (K=1); phased and unphased are
 * identical for biallelic K=1. */
struct bgen_single_haploid_case {
    int8_t a;
    uint8_t ploidy;
    uint8_t b0;
};

static struct bgen_single_haploid_case bgen_single_haploid_cases[] = {
    { 0, VCZ_BGEN_PLOIDY_HAPLOID, 0xFF },
    { 1, VCZ_BGEN_PLOIDY_HAPLOID, 0x00 },
    { -1, VCZ_BGEN_PLOIDY_MISSING_HAPLOID, 0x00 },
};

/* Invalid single-sample inputs: -2 in slot 0 is zero-ploidy and not
 * representable in BGEN; the kernel returns
 * VCZ_ERR_BGEN_INVALID_PLOIDY. */
struct bgen_single_invalid_case {
    int8_t a;
    int8_t b;
};

static struct bgen_single_invalid_case bgen_single_invalid_ploidy_cases[] = {
    { -2, 0 },
    { -2, -2 },
    { -2, -1 },
    { -2, 1 },
};

/* Invalid-allele single-sample inputs: BGEN is biallelic and only
 * accepts {-2, -1, 0, 1}. Anything else is a data-quality error and
 * the kernel returns VCZ_ERR_BGEN_INVALID_ALLELE. */
static struct bgen_single_invalid_case bgen_single_invalid_allele_cases[] = {
    /* Diploid slot positives out of range. */
    { 2, 0 },
    { 0, 2 },
    { 2, 2 },
    { 1, 2 },
    { 2, 1 },
    /* Haploid slot 0 out of range (slot 1 = -2 sentinel). */
    { 2, -2 },
    { 127, -2 },
    /* Below -2 in either slot. */
    { -3, 0 },
    { 0, -3 },
    { -128, -128 },
    /* Mixed: invalid allele combined with otherwise-valid sentinel. */
    { 3, -1 },
    { -1, 5 },
};

static void
check_single_sample_header(
    const uint8_t *buf, uint8_t expected_phased, uint8_t pmin, uint8_t pmax)
{
    /* Header: N=1 LE, K=2, P_min, P_max */
    CU_ASSERT_EQUAL_FATAL(buf[0], 1);
    CU_ASSERT_EQUAL_FATAL(buf[1], 0);
    CU_ASSERT_EQUAL_FATAL(buf[2], 0);
    CU_ASSERT_EQUAL_FATAL(buf[3], 0);
    CU_ASSERT_EQUAL_FATAL(buf[4], 2);
    CU_ASSERT_EQUAL_FATAL(buf[5], 0);
    CU_ASSERT_EQUAL_FATAL(buf[6], pmin);
    CU_ASSERT_EQUAL_FATAL(buf[7], pmax);
    /* buf[8] is the ploidy byte (variable per case) */
    CU_ASSERT_EQUAL_FATAL(buf[9], expected_phased);
    CU_ASSERT_EQUAL_FATAL(buf[10], VCZ_BGEN_BITS_PER_PROB);
}

static void
test_bgen_geno_blocks_single_sample_diploid_unphased(void)
{
    int8_t genotypes[2];
    uint8_t phased = 0;
    uint8_t buf[13]; /* 10 + 3*1 */
    uint32_t lens[1];
    size_t j, num_cases;
    int ret;

    num_cases = sizeof(bgen_single_diploid_cases) / sizeof(*bgen_single_diploid_cases);
    for (j = 0; j < num_cases; j++) {
        genotypes[0] = bgen_single_diploid_cases[j].a;
        genotypes[1] = bgen_single_diploid_cases[j].b;
        memset(buf, 0xCC, sizeof(buf));
        ret = vcz_encode_bgen_geno_blocks(
            1, 1, genotypes, &phased, buf, sizeof(buf), lens);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL_FATAL(lens[0], 13);
        check_single_sample_header(buf, 0, 2, 2);
        CU_ASSERT_EQUAL_FATAL(buf[8], bgen_single_diploid_cases[j].ploidy);
        CU_ASSERT_EQUAL_FATAL(buf[11], bgen_single_diploid_cases[j].b0_unphased);
        CU_ASSERT_EQUAL_FATAL(buf[12], bgen_single_diploid_cases[j].b1_unphased);
    }
}

static void
test_bgen_geno_blocks_single_sample_diploid_phased(void)
{
    int8_t genotypes[2];
    uint8_t phased = 1;
    uint8_t buf[13];
    uint32_t lens[1];
    size_t j, num_cases;
    int ret;

    num_cases = sizeof(bgen_single_diploid_cases) / sizeof(*bgen_single_diploid_cases);
    for (j = 0; j < num_cases; j++) {
        genotypes[0] = bgen_single_diploid_cases[j].a;
        genotypes[1] = bgen_single_diploid_cases[j].b;
        memset(buf, 0xCC, sizeof(buf));
        ret = vcz_encode_bgen_geno_blocks(
            1, 1, genotypes, &phased, buf, sizeof(buf), lens);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_EQUAL_FATAL(lens[0], 13);
        check_single_sample_header(buf, 1, 2, 2);
        CU_ASSERT_EQUAL_FATAL(buf[8], bgen_single_diploid_cases[j].ploidy);
        CU_ASSERT_EQUAL_FATAL(buf[11], bgen_single_diploid_cases[j].b0_phased);
        CU_ASSERT_EQUAL_FATAL(buf[12], bgen_single_diploid_cases[j].b1_phased);
    }
}

static void
test_bgen_geno_blocks_single_sample_haploid(void)
{
    int8_t genotypes[2];
    uint8_t buf[13]; /* size to worst-case-diploid stride */
    uint32_t lens[1];
    uint8_t phased_flags[2] = { 0, 1 };
    size_t p, j, num_cases;
    int ret;

    num_cases = sizeof(bgen_single_haploid_cases) / sizeof(*bgen_single_haploid_cases);
    for (p = 0; p < 2; p++) {
        for (j = 0; j < num_cases; j++) {
            genotypes[0] = bgen_single_haploid_cases[j].a;
            genotypes[1] = VCZ_INT_FILL;
            memset(buf, 0xCC, sizeof(buf));
            ret = vcz_encode_bgen_geno_blocks(
                1, 1, genotypes, &phased_flags[p], buf, sizeof(buf), lens);
            CU_ASSERT_EQUAL_FATAL(ret, 0);
            /* Haploid row length: 8 header + 1 ploidy + 2 flags + 1 prob = 12. */
            CU_ASSERT_EQUAL_FATAL(lens[0], 12);
            check_single_sample_header(buf, phased_flags[p], 1, 1);
            CU_ASSERT_EQUAL_FATAL(buf[8], bgen_single_haploid_cases[j].ploidy);
            CU_ASSERT_EQUAL_FATAL(buf[11], bgen_single_haploid_cases[j].b0);
        }
    }
}

static void
test_bgen_geno_blocks_single_sample_invalid_ploidy(void)
{
    int8_t genotypes[2];
    uint8_t buf[13];
    uint32_t lens[1];
    uint8_t phased_flags[2] = { 0, 1 };
    size_t p, j, num_cases;
    int ret;

    num_cases = sizeof(bgen_single_invalid_ploidy_cases)
                / sizeof(*bgen_single_invalid_ploidy_cases);
    for (p = 0; p < 2; p++) {
        for (j = 0; j < num_cases; j++) {
            genotypes[0] = bgen_single_invalid_ploidy_cases[j].a;
            genotypes[1] = bgen_single_invalid_ploidy_cases[j].b;
            ret = vcz_encode_bgen_geno_blocks(
                1, 1, genotypes, &phased_flags[p], buf, sizeof(buf), lens);
            CU_ASSERT_EQUAL_FATAL(ret, VCZ_ERR_BGEN_INVALID_PLOIDY);
        }
    }
}

static void
test_bgen_geno_blocks_single_sample_invalid_allele(void)
{
    int8_t genotypes[2];
    uint8_t buf[13];
    uint32_t lens[1];
    uint8_t phased_flags[2] = { 0, 1 };
    size_t p, j, num_cases;
    int ret;

    num_cases = sizeof(bgen_single_invalid_allele_cases)
                / sizeof(*bgen_single_invalid_allele_cases);
    for (p = 0; p < 2; p++) {
        for (j = 0; j < num_cases; j++) {
            genotypes[0] = bgen_single_invalid_allele_cases[j].a;
            genotypes[1] = bgen_single_invalid_allele_cases[j].b;
            ret = vcz_encode_bgen_geno_blocks(
                1, 1, genotypes, &phased_flags[p], buf, sizeof(buf), lens);
            CU_ASSERT_EQUAL_FATAL(ret, VCZ_ERR_BGEN_INVALID_ALLELE);
        }
    }
}

static void
test_bgen_geno_blocks_invalid_allele_mid_chunk(void)
{
    /* Invalid allele on sample 1 (slot 0) of variant 2 of a 3-variant
     * chunk: the error must surface from the kernel even when the
     * earlier rows look fine. */
    int8_t genotypes[3 * 4 * 2];
    uint8_t phased[3] = { 0, 0, 0 };
    uint8_t buf[3 * 22];
    uint32_t lens[3];
    int ret;

    memset(genotypes, 0, sizeof(genotypes));
    /* Plant a 2 in variant 2, sample 1, slot 0. */
    genotypes[2 * 4 * 2 + 1 * 2] = 2;
    ret = vcz_encode_bgen_geno_blocks(3, 4, genotypes, phased, buf, 22, lens);
    CU_ASSERT_EQUAL_FATAL(ret, VCZ_ERR_BGEN_INVALID_ALLELE);
}

static void
check_header_for_size(size_t num_samples)
{
    int8_t *gt;
    uint8_t phased = 0;
    uint8_t *buf;
    uint32_t lens[1];
    size_t row_max;
    int ret;

    row_max = vcz_bgen_geno_block_row_max_size(num_samples);
    gt = calloc(num_samples > 0 ? num_samples : 1, 2 * sizeof(int8_t));
    buf = malloc(row_max);
    CU_ASSERT_FATAL(gt != NULL);
    CU_ASSERT_FATAL(buf != NULL);

    ret = vcz_encode_bgen_geno_blocks(1, num_samples, gt, &phased, buf, row_max, lens);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    /* All-zero diploid: every row is the worst-case width. */
    CU_ASSERT_EQUAL_FATAL(lens[0], (uint32_t) row_max);
    /* N as uint32 LE */
    CU_ASSERT_EQUAL_FATAL(buf[0], (uint8_t) (num_samples & 0xFF));
    CU_ASSERT_EQUAL_FATAL(buf[1], (uint8_t) ((num_samples >> 8) & 0xFF));
    CU_ASSERT_EQUAL_FATAL(buf[2], (uint8_t) ((num_samples >> 16) & 0xFF));
    CU_ASSERT_EQUAL_FATAL(buf[3], (uint8_t) ((num_samples >> 24) & 0xFF));
    /* K=2 little-endian, P_min=2, P_max=2 (all diploid). */
    CU_ASSERT_EQUAL_FATAL(buf[4], 2);
    CU_ASSERT_EQUAL_FATAL(buf[5], 0);
    CU_ASSERT_EQUAL_FATAL(buf[6], 2);
    CU_ASSERT_EQUAL_FATAL(buf[7], 2);
    /* Phased flag and B byte sit at the expected offsets. */
    CU_ASSERT_EQUAL_FATAL(buf[8 + num_samples], 0);
    CU_ASSERT_EQUAL_FATAL(buf[8 + num_samples + 1], VCZ_BGEN_BITS_PER_PROB);

    free(gt);
    free(buf);
}

static void
test_bgen_geno_blocks_header_bytes(void)
{
    check_header_for_size(0);
    check_header_for_size(1);
    check_header_for_size(4);
    check_header_for_size(255);
    check_header_for_size(256);
    check_header_for_size(65536);
    check_header_for_size(1000000);
}

static void
test_bgen_geno_blocks_zero_samples(void)
{
    /* Three variants, no samples: every row is exactly the 10 header
     * bytes (8) + phased flag (1) + B byte (1). Pmin/Pmax default to
     * 2 when there are no samples. */
    int8_t genotypes[1] = { 0 }; /* unused but must be a valid pointer */
    uint8_t phased[3] = { 0, 1, 0 };
    uint8_t buf[30];
    uint32_t lens[3];
    size_t v, off;
    int ret;

    memset(buf, 0xCC, sizeof(buf));
    ret = vcz_encode_bgen_geno_blocks(3, 0, genotypes, phased, buf, 10, lens);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    for (v = 0; v < 3; v++) {
        off = v * 10;
        CU_ASSERT_EQUAL_FATAL(lens[v], 10);
        CU_ASSERT_EQUAL_FATAL(buf[off + 0], 0);
        CU_ASSERT_EQUAL_FATAL(buf[off + 1], 0);
        CU_ASSERT_EQUAL_FATAL(buf[off + 2], 0);
        CU_ASSERT_EQUAL_FATAL(buf[off + 3], 0);
        CU_ASSERT_EQUAL_FATAL(buf[off + 4], 2);
        CU_ASSERT_EQUAL_FATAL(buf[off + 5], 0);
        CU_ASSERT_EQUAL_FATAL(buf[off + 6], 2);
        CU_ASSERT_EQUAL_FATAL(buf[off + 7], 2);
        CU_ASSERT_EQUAL_FATAL(buf[off + 8], phased[v]);
        CU_ASSERT_EQUAL_FATAL(buf[off + 9], VCZ_BGEN_BITS_PER_PROB);
    }
}

static void
test_bgen_geno_blocks_zero_variants(void)
{
    /* Zero variants must not touch buf, regardless of num_samples. */
    int8_t genotypes[1] = { 0 };
    uint8_t phased[1] = { 0 };
    uint8_t buf[64];
    uint32_t lens[1] = { 0 };
    size_t j;
    int ret;

    memset(buf, 0xCC, sizeof(buf));
    ret = vcz_encode_bgen_geno_blocks(0, 5, genotypes, phased, buf, 25, lens);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    for (j = 0; j < sizeof(buf); j++) {
        CU_ASSERT_EQUAL_FATAL(buf[j], 0xCC);
    }
}

static void
test_bgen_geno_blocks_example_unphased(void)
{
    /* Mirrors tests/test_bgen.py::test_unphased_basic. */
    int8_t genotypes[] = { 0, 0, 0, 1, 1, 1 }; /* 1 variant, 3 samples */
    uint8_t phased = 0;
    uint8_t buf[19]; /* 10 + 3*3 */
    uint32_t lens[1];
    /* Header (8) + ploidy (3) + phased + B + probs (6) = 19 */
    uint8_t expected[] = {
        3, 0, 0, 0, 2, 0, 2, 2,             /* header: N=3, K=2, P_min=2, P_max=2 */
        0x02, 0x02, 0x02,                   /* ploidy: 3 diploids */
        0x00,                               /* phased flag */
        VCZ_BGEN_BITS_PER_PROB,             /* B = 8 */
        0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, /* hom-ref, het, hom-alt */
    };
    size_t j;

    vcz_encode_bgen_geno_blocks(1, 3, genotypes, &phased, buf, sizeof(buf), lens);
    CU_ASSERT_EQUAL_FATAL(lens[0], sizeof(expected));
    for (j = 0; j < sizeof(expected); j++) {
        CU_ASSERT_EQUAL_FATAL(buf[j], expected[j]);
    }
}

static void
test_bgen_geno_blocks_example_phased(void)
{
    /* Mirrors tests/test_bgen.py::test_phased. */
    // clang-format off
    int8_t genotypes[] = {
        0, 0, 0, 1, 1, 0, 1, 1,
    }; /* 1 variant, 4 samples */
    uint8_t phased = 1;
    uint8_t buf[22]; /* 10 + 3*4 */
    uint32_t lens[1];
    uint8_t expected[] = {
        4, 0, 0, 0, 2, 0, 2, 2,                         /* header: N=4 */
        0x02, 0x02, 0x02, 0x02,                         /* ploidy: all diploid */
        0x01,                                           /* phased flag */
        VCZ_BGEN_BITS_PER_PROB,
        0xFF, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00,
    };
    // clang-format on
    size_t j;

    vcz_encode_bgen_geno_blocks(1, 4, genotypes, &phased, buf, sizeof(buf), lens);
    CU_ASSERT_EQUAL_FATAL(lens[0], sizeof(expected));
    for (j = 0; j < sizeof(expected); j++) {
        CU_ASSERT_EQUAL_FATAL(buf[j], expected[j]);
    }
}

static void
test_bgen_geno_blocks_example_missing(void)
{
    /* Mirrors tests/test_bgen.py::test_missing_genotype. */
    int8_t genotypes[] = { -1, -1, 0, -1, 0, 1 }; /* 1 variant, 3 samples */
    uint8_t phased = 0;
    uint8_t buf[19];
    uint32_t lens[1];
    // clang-format off
    uint8_t expected[] = {
        3, 0, 0, 0, 2, 0, 2, 2,
        0x82, 0x82, 0x02,                /* missing, missing, diploid */
        0x00,
        VCZ_BGEN_BITS_PER_PROB,
        0x00, 0x00, 0x00, 0x00, 0x00, 0xFF,
    };
    // clang-format on
    size_t j;

    vcz_encode_bgen_geno_blocks(1, 3, genotypes, &phased, buf, sizeof(buf), lens);
    CU_ASSERT_EQUAL_FATAL(lens[0], sizeof(expected));
    for (j = 0; j < sizeof(expected); j++) {
        CU_ASSERT_EQUAL_FATAL(buf[j], expected[j]);
    }
}

static void
test_bgen_geno_blocks_example_haploid(void)
{
    /* 1 variant, 3 haploid samples: alleles (0, 1, -1) with second slot
     * filled with -2. Expected layout: 8 header + 3 ploidy + 2 flags +
     * 3 prob = 16 bytes; Pmin=Pmax=1. */
    int8_t genotypes[] = { 0, -2, 1, -2, -1, -2 };
    uint8_t phased = 0;
    uint8_t buf[19]; /* worst-case stride for safety */
    uint32_t lens[1];
    uint8_t expected[] = {
        3,
        0,
        0,
        0,
        2,
        0,
        1,
        1,
        VCZ_BGEN_PLOIDY_HAPLOID,
        VCZ_BGEN_PLOIDY_HAPLOID,
        VCZ_BGEN_PLOIDY_MISSING_HAPLOID,
        0x00,
        VCZ_BGEN_BITS_PER_PROB,
        0xFF,
        0x00,
        0x00,
    };
    size_t j;

    vcz_encode_bgen_geno_blocks(1, 3, genotypes, &phased, buf, sizeof(buf), lens);
    CU_ASSERT_EQUAL_FATAL(lens[0], sizeof(expected));
    for (j = 0; j < sizeof(expected); j++) {
        CU_ASSERT_EQUAL_FATAL(buf[j], expected[j]);
    }
}

static void
test_bgen_geno_blocks_example_mixed_ploidy(void)
{
    /* 1 variant, 4 samples: diploid (0,0), haploid (1, -2), missing
     * diploid (-1, -1), missing haploid (-1, -2). Length:
     * 8 + 4 + 2 + (2+1+2+1) = 20. Pmin=1, Pmax=2. */
    int8_t genotypes[] = { 0, 0, 1, -2, -1, -1, -1, -2 };
    uint8_t phased = 0;
    uint8_t buf[22]; /* worst-case stride 10 + 3*4 */
    uint32_t lens[1];
    uint8_t expected[] = {
        4,
        0,
        0,
        0,
        2,
        0,
        1,
        2,
        VCZ_BGEN_PLOIDY_DIPLOID,
        VCZ_BGEN_PLOIDY_HAPLOID,
        VCZ_BGEN_PLOIDY_MISSING_DIPLOID,
        VCZ_BGEN_PLOIDY_MISSING_HAPLOID,
        0x00,
        VCZ_BGEN_BITS_PER_PROB,
        /* sample 0 diploid (0,0) unphased: 0xFF, 0x00 */
        0xFF,
        0x00,
        /* sample 1 haploid allele 1: 0x00 */
        0x00,
        /* sample 2 missing diploid: 0x00, 0x00 */
        0x00,
        0x00,
        /* sample 3 missing haploid: 0x00 */
        0x00,
    };
    size_t j;

    vcz_encode_bgen_geno_blocks(1, 4, genotypes, &phased, buf, sizeof(buf), lens);
    CU_ASSERT_EQUAL_FATAL(lens[0], sizeof(expected));
    for (j = 0; j < sizeof(expected); j++) {
        CU_ASSERT_EQUAL_FATAL(buf[j], expected[j]);
    }
}

static void
test_bgen_geno_blocks_invalid_ploidy_error(void)
{
    /* A single sample with -2 in slot 0 anywhere in the chunk must
     * surface as VCZ_ERR_BGEN_INVALID_PLOIDY. */
    int8_t genotypes[] = { 0, 0, -2, -2, 0, 0 };
    uint8_t phased = 0;
    uint8_t buf[3 * 13];
    uint32_t lens[3];
    int ret;

    ret = vcz_encode_bgen_geno_blocks(1, 3, genotypes, &phased, buf, 13, lens);
    CU_ASSERT_EQUAL_FATAL(ret, VCZ_ERR_BGEN_INVALID_PLOIDY);
}

static void
test_bgen_geno_blocks_mixed_phase(void)
{
    /* Same 3-sample genotype repeated for 4 variants with alternating
     * phase flags; verifies the per-variant flag drives the per-row
     * branch and that variant rows are independent. */
    int8_t single[] = { 0, 0, 1, 0, 1, 1 }; /* 3 samples: (0,0),(1,0),(1,1) */
    int8_t genotypes[24];
    uint8_t phased[4] = { 0, 1, 0, 1 };
    uint8_t buf[4 * 19];
    uint32_t lens[4];
    /* Unphased: (0,0)=homref => B0=FF,B1=00; (1,0)=het => B0=00,B1=FF;
     * (1,1) => B0=00,B1=00.
     * Phased: (0,0) => B0=FF,B1=FF; (1,0) => B0=00,B1=FF;
     * (1,1) => B0=00,B1=00.
     */
    uint8_t prob_unphased[] = { 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00 };
    uint8_t prob_phased[] = { 0xFF, 0xFF, 0x00, 0xFF, 0x00, 0x00 };
    size_t v, j;

    for (v = 0; v < 4; v++) {
        memcpy(genotypes + v * 6, single, sizeof(single));
    }
    vcz_encode_bgen_geno_blocks(4, 3, genotypes, phased, buf, 19, lens);

    for (v = 0; v < 4; v++) {
        const uint8_t *row = buf + v * 19;
        CU_ASSERT_EQUAL_FATAL(lens[v], 19);
        /* Common header */
        CU_ASSERT_EQUAL_FATAL(row[0], 3);
        CU_ASSERT_EQUAL_FATAL(row[1], 0);
        CU_ASSERT_EQUAL_FATAL(row[2], 0);
        CU_ASSERT_EQUAL_FATAL(row[3], 0);
        CU_ASSERT_EQUAL_FATAL(row[4], 2);
        CU_ASSERT_EQUAL_FATAL(row[5], 0);
        CU_ASSERT_EQUAL_FATAL(row[6], 2);
        CU_ASSERT_EQUAL_FATAL(row[7], 2);
        /* All samples non-missing => ploidy 0x02 across the board */
        CU_ASSERT_EQUAL_FATAL(row[8], 0x02);
        CU_ASSERT_EQUAL_FATAL(row[9], 0x02);
        CU_ASSERT_EQUAL_FATAL(row[10], 0x02);
        CU_ASSERT_EQUAL_FATAL(row[11], phased[v]);
        CU_ASSERT_EQUAL_FATAL(row[12], VCZ_BGEN_BITS_PER_PROB);
        for (j = 0; j < 6; j++) {
            CU_ASSERT_EQUAL_FATAL(
                row[13 + j], phased[v] ? prob_phased[j] : prob_unphased[j]);
        }
    }
}

/* Independent reference used to validate the large parameter-sweep case.
 * Spec-derived; not a refactor of the kernel under test. Handles
 * per-sample ploidy {1, 2} and missing variants of both. Writes one
 * row at `out`; returns the number of bytes written. */
static uint32_t
build_expected_bgen_row(
    uint8_t *out, size_t num_samples, const int8_t *gt, uint8_t variant_phased)
{
    size_t s;
    int8_t a, b;
    uint8_t pmin, pmax;
    size_t prob_offset;
    uint8_t *prob_out;

    pmin = 2;
    pmax = 1;
    prob_offset = 0;
    prob_out = out + 8 + num_samples + 2;

    for (s = 0; s < num_samples; s++) {
        a = gt[2 * s];
        b = gt[2 * s + 1];
        if (b == VCZ_INT_FILL) {
            if (a == VCZ_INT_MISSING) {
                out[8 + s] = VCZ_BGEN_PLOIDY_MISSING_HAPLOID;
                prob_out[prob_offset] = 0x00;
            } else {
                out[8 + s] = VCZ_BGEN_PLOIDY_HAPLOID;
                prob_out[prob_offset] = (a == 0) ? 0xFF : 0x00;
            }
            prob_offset += 1;
            if (pmin > 1) {
                pmin = 1;
            }
        } else if (a < 0 || b < 0) {
            out[8 + s] = VCZ_BGEN_PLOIDY_MISSING_DIPLOID;
            prob_out[prob_offset] = 0x00;
            prob_out[prob_offset + 1] = 0x00;
            prob_offset += 2;
            pmax = 2;
        } else {
            out[8 + s] = VCZ_BGEN_PLOIDY_DIPLOID;
            if (variant_phased) {
                prob_out[prob_offset] = (a == 0) ? 0xFF : 0x00;
                prob_out[prob_offset + 1] = (b == 0) ? 0xFF : 0x00;
            } else {
                prob_out[prob_offset] = (a == 0 && b == 0) ? 0xFF : 0x00;
                prob_out[prob_offset + 1]
                    = ((a == 0 && b == 1) || (a == 1 && b == 0)) ? 0xFF : 0x00;
            }
            prob_offset += 2;
            pmax = 2;
        }
    }

    if (num_samples == 0) {
        pmin = 2;
        pmax = 2;
    }

    out[0] = (uint8_t) (num_samples & 0xFF);
    out[1] = (uint8_t) ((num_samples >> 8) & 0xFF);
    out[2] = (uint8_t) ((num_samples >> 16) & 0xFF);
    out[3] = (uint8_t) ((num_samples >> 24) & 0xFF);
    out[4] = 2;
    out[5] = 0;
    out[6] = pmin;
    out[7] = pmax;
    out[8 + num_samples] = variant_phased;
    out[8 + num_samples + 1] = VCZ_BGEN_BITS_PER_PROB;

    return (uint32_t) (8 + num_samples + 2 + prob_offset);
}

static void
test_bgen_geno_blocks_large(void)
{
    /* 100 variants x 100 samples sweep. Deterministic content covers
     * the missing/normal/het/half-call and haploid branches in both
     * phasings. Verified row-by-row against a separately-coded spec
     * reference. */
    const size_t num_variants = 100;
    const size_t num_samples = 100;
    const size_t row_max = 10 + 3 * num_samples;
    int8_t *genotypes;
    uint8_t *phased;
    uint8_t *buf;
    uint32_t *lens;
    uint8_t *expected;
    size_t v, s, j;
    int8_t a, b;
    uint32_t expected_len;
    int ret;

    genotypes = malloc(num_variants * num_samples * 2);
    phased = malloc(num_variants);
    buf = malloc(num_variants * row_max);
    lens = malloc(num_variants * sizeof(uint32_t));
    expected = malloc(row_max);
    CU_ASSERT_FATAL(genotypes != NULL);
    CU_ASSERT_FATAL(phased != NULL);
    CU_ASSERT_FATAL(buf != NULL);
    CU_ASSERT_FATAL(lens != NULL);
    CU_ASSERT_FATAL(expected != NULL);

    for (v = 0; v < num_variants; v++) {
        phased[v] = (uint8_t) (v % 2);
        for (s = 0; s < num_samples; s++) {
            /* a in {-1, 0, 1} (omitting -2 -> zero-ploidy error;
             * omitting 2+ -> invalid-allele error).
             * b in {-2, -1, 0, 1}, so haploid (b=-2), missing,
             * half-missing, het and hom all appear. */
            a = (int8_t) (((v + s) % 3) - 1);
            b = (int8_t) (((v + 2 * s) % 4) - 2);
            genotypes[(v * num_samples + s) * 2] = a;
            genotypes[(v * num_samples + s) * 2 + 1] = b;
        }
    }

    ret = vcz_encode_bgen_geno_blocks(
        num_variants, num_samples, genotypes, phased, buf, row_max, lens);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    for (v = 0; v < num_variants; v++) {
        memset(expected, 0xAB, row_max);
        expected_len = build_expected_bgen_row(
            expected, num_samples, genotypes + v * num_samples * 2, phased[v]);
        CU_ASSERT_EQUAL_FATAL(lens[v], expected_len);
        for (j = 0; j < expected_len; j++) {
            CU_ASSERT_EQUAL_FATAL(buf[v * row_max + j], expected[j]);
        }
    }

    free(genotypes);
    free(phased);
    free(buf);
    free(lens);
    free(expected);
}

/* Decode a little-endian uint16/uint32 from a byte buffer. Match the
 * encode_u32_le / uint16-LE byte order the kernel writes. */
static uint16_t
decode_u16_le(const uint8_t *buf)
{
    return (uint16_t) (buf[0] | ((uint16_t) buf[1] << 8));
}

static uint32_t
decode_u32_le(const uint8_t *buf)
{
    return (uint32_t) buf[0] | ((uint32_t) buf[1] << 8) | ((uint32_t) buf[2] << 16)
           | ((uint32_t) buf[3] << 24);
}

/* Inputs for vcz_encode_bgen_chunk_slice_level0 tests. Caller stages the
 * arrays; helper kicks the kernel and returns its rc. Use scope-block
 * decls only at function tops to satisfy the house style. */
typedef struct {
    size_t num_variants;
    size_t num_samples;
    size_t uniform_ploidy;
    size_t varid_max;
    size_t rsid_max;
    size_t chrom_max;
    size_t allele_max;
    const uint8_t *varid;
    const uint8_t *rsid;
    const uint8_t *chrom;
    const uint8_t *allele1;
    const uint8_t *allele2;
    const int32_t *position;
    const int8_t *genotypes;
    const uint8_t *phased;
} chunk_slice_args_t;

static int
call_chunk_slice(const chunk_slice_args_t *a, uint8_t *out_buf)
{
    return vcz_encode_bgen_chunk_slice_level0(a->num_variants, a->num_samples,
        a->uniform_ploidy, a->varid, a->varid_max, a->rsid, a->rsid_max, a->chrom,
        a->chrom_max, a->allele1, a->allele2, a->allele_max, a->position, a->genotypes,
        a->phased, out_buf);
}

static void
test_bgen_chunk_slice_zero_variants(void)
{
    /* num_variants=0: kernel must not touch out_buf. */
    uint8_t out_buf[64];
    chunk_slice_args_t args
        = { 0, 5, 2, 1, 1, 1, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
    size_t j;
    int rc;

    memset(out_buf, 0xCC, sizeof(out_buf));
    rc = call_chunk_slice(&args, out_buf);
    CU_ASSERT_EQUAL_FATAL(rc, 0);
    for (j = 0; j < sizeof(out_buf); j++) {
        CU_ASSERT_EQUAL_FATAL(out_buf[j], 0xCC);
    }
}

static void
test_bgen_chunk_slice_single_variant_diploid(void)
{
    /* 1 variant x 1 sample, hom-ref diploid, unphased. Drives the
     * variant-header layout, the K=2 marker, the C/D framing and the
     * vcz_compress2 stored-zlib envelope. */
    uint8_t varid[2] = { 'r', 's' };
    uint8_t rsid[2] = { 'r', 's' };
    uint8_t chrom[4] = { 'c', 'h', 'r', '1' };
    uint8_t allele1[1] = { 'A' };
    uint8_t allele2[1] = { 'T' };
    int32_t position[1] = { 100 };
    int8_t genotypes[2] = { 0, 0 };
    uint8_t phased[1] = { 0 };
    chunk_slice_args_t args = { 1, 1, 2, 2, 2, 4, 1, varid, rsid, chrom, allele1,
        allele2, position, genotypes, phased };
    size_t geno_size = vcz_bgen_geno_block_size(1, 2);
    size_t payload_size = vcz_compress_bound(geno_size);
    size_t bpv = vcz_bgen_variant_block_size(1, 2, 2, 2, 4, 1);
    uint8_t *out = malloc(bpv);
    const uint8_t *p;
    int rc;

    CU_ASSERT_FATAL(out != NULL);
    rc = call_chunk_slice(&args, out);
    CU_ASSERT_EQUAL_FATAL(rc, 0);

    p = out;
    /* varid: u16 LE length + bytes */
    CU_ASSERT_EQUAL_FATAL(decode_u16_le(p), 2);
    CU_ASSERT_EQUAL_FATAL(memcmp(p + 2, varid, 2), 0);
    p += 2 + 2;
    /* rsid */
    CU_ASSERT_EQUAL_FATAL(decode_u16_le(p), 2);
    CU_ASSERT_EQUAL_FATAL(memcmp(p + 2, rsid, 2), 0);
    p += 2 + 2;
    /* chrom */
    CU_ASSERT_EQUAL_FATAL(decode_u16_le(p), 4);
    CU_ASSERT_EQUAL_FATAL(memcmp(p + 2, chrom, 4), 0);
    p += 2 + 4;
    /* position */
    CU_ASSERT_EQUAL_FATAL(decode_u32_le(p), 100);
    p += 4;
    /* K = 2 */
    CU_ASSERT_EQUAL_FATAL(decode_u16_le(p), 2);
    p += 2;
    /* allele1 */
    CU_ASSERT_EQUAL_FATAL(decode_u32_le(p), 1);
    CU_ASSERT_EQUAL_FATAL(p[4], 'A');
    p += 4 + 1;
    /* allele2 */
    CU_ASSERT_EQUAL_FATAL(decode_u32_le(p), 1);
    CU_ASSERT_EQUAL_FATAL(p[4], 'T');
    p += 4 + 1;
    /* C, D */
    CU_ASSERT_EQUAL_FATAL(decode_u32_le(p), (uint32_t) (4 + payload_size));
    CU_ASSERT_EQUAL_FATAL(decode_u32_le(p + 4), (uint32_t) geno_size);
    p += 8;
    /* Stored zlib envelope. Spec compress2(level=0) header: 0x78 0x01;
     * one stored block (BFINAL=1, BTYPE=00) for geno_size < 65535. */
    CU_ASSERT_EQUAL_FATAL(p[0], 0x78);
    CU_ASSERT_EQUAL_FATAL(p[1], 0x01);
    CU_ASSERT_EQUAL_FATAL(p[2], 0x01); /* BFINAL=1, BTYPE=00 */
    CU_ASSERT_EQUAL_FATAL(decode_u16_le(p + 3), (uint16_t) geno_size);
    CU_ASSERT_EQUAL_FATAL(decode_u16_le(p + 5), (uint16_t) ~geno_size);
    /* Geno block at p + 2 + 5: N=1, K=2, Pmin=2, Pmax=2. */
    CU_ASSERT_EQUAL_FATAL(decode_u32_le(p + 7), 1);
    CU_ASSERT_EQUAL_FATAL(p[7 + 4], 2); /* K low */
    CU_ASSERT_EQUAL_FATAL(p[7 + 5], 0); /* K high */
    CU_ASSERT_EQUAL_FATAL(p[7 + 6], 2); /* Pmin */
    CU_ASSERT_EQUAL_FATAL(p[7 + 7], 2); /* Pmax */
    CU_ASSERT_EQUAL_FATAL(p[7 + 8], VCZ_BGEN_PLOIDY_DIPLOID);
    CU_ASSERT_EQUAL_FATAL(p[7 + 9], 0); /* phased flag */
    CU_ASSERT_EQUAL_FATAL(p[7 + 10], VCZ_BGEN_BITS_PER_PROB);
    CU_ASSERT_EQUAL_FATAL(p[7 + 11], 0xFF); /* hom-ref unphased */
    CU_ASSERT_EQUAL_FATAL(p[7 + 12], 0x00);

    free(out);
}

static void
test_bgen_chunk_slice_multi_variant_offsets(void)
{
    /* Three variants land at deterministic byte offsets bpv apart;
     * confirm position[v] surfaces at the right offset in each. */
    uint8_t varid[3] = { 'a', 'b', 'c' };
    uint8_t rsid[3] = { 'a', 'b', 'c' };
    uint8_t chrom[3] = { '1', '1', '1' };
    uint8_t allele1[3] = { 'A', 'C', 'G' };
    uint8_t allele2[3] = { 'T', 'G', 'C' };
    int32_t position[3] = { 100, 200, 300 };
    int8_t genotypes[3 * 2 * 2]
        = { 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0 }; /* 3 variants x 2 samples */
    uint8_t phased[3] = { 0, 0, 0 };
    chunk_slice_args_t args = { 3, 2, 2, 1, 1, 1, 1, varid, rsid, chrom, allele1,
        allele2, position, genotypes, phased };
    size_t bpv = vcz_bgen_variant_block_size(2, 2, 1, 1, 1, 1);
    size_t v;
    uint8_t *out = malloc(3 * bpv);
    int rc;

    CU_ASSERT_FATAL(out != NULL);
    rc = call_chunk_slice(&args, out);
    CU_ASSERT_EQUAL_FATAL(rc, 0);

    for (v = 0; v < 3; v++) {
        const uint8_t *vp = out + v * bpv;
        /* position lives at offset 2+1 + 2+1 + 2+1 = 9 */
        CU_ASSERT_EQUAL_FATAL(decode_u32_le(vp + 9), (uint32_t) position[v]);
        /* allele1 single-byte payload after position(4)+K(2)+u32 length(4) = 10 */
        CU_ASSERT_EQUAL_FATAL(vp[9 + 4 + 2 + 4], allele1[v]);
    }
    free(out);
}

static void
test_bgen_chunk_slice_uniform_haploid(void)
{
    /* uniform_ploidy=1 with -2 in slot 1 for every sample: kernel
     * accepts and produces a haploid geno block. */
    uint8_t varid[1] = { '.' };
    uint8_t rsid[1] = { '.' };
    uint8_t chrom[1] = { '1' };
    uint8_t allele1[1] = { 'A' };
    uint8_t allele2[1] = { 'T' };
    int32_t position[1] = { 42 };
    int8_t genotypes[2 * 2] = { 0, -2, 1, -2 }; /* 1 variant x 2 haploid samples */
    uint8_t phased[1] = { 0 };
    chunk_slice_args_t args = { 1, 2, 1, 1, 1, 1, 1, varid, rsid, chrom, allele1,
        allele2, position, genotypes, phased };
    size_t geno_size = vcz_bgen_geno_block_size(2, 1);
    size_t bpv = vcz_bgen_variant_block_size(2, 1, 1, 1, 1, 1);
    uint8_t *out = malloc(bpv);
    const uint8_t *p;
    int rc;

    CU_ASSERT_FATAL(out != NULL);
    rc = call_chunk_slice(&args, out);
    CU_ASSERT_EQUAL_FATAL(rc, 0);

    /* D in the C/D framing must equal the haploid geno_size, not the
     * worst-case diploid stride. */
    p = out + (28 - 8) + 1 + 1 + 1 + 2; /* skip variant header */
    CU_ASSERT_EQUAL_FATAL(decode_u32_le(p + 4), (uint32_t) geno_size);
    free(out);
}

static void
test_bgen_chunk_slice_invalid_ploidy(void)
{
    /* -2 in slot 0 is "zero-ploidy" — not representable in BGEN.
     * Surfaces via VCZ_ERR_BGEN_INVALID_PLOIDY. */
    uint8_t varid[1] = { '.' };
    uint8_t rsid[1] = { '.' };
    uint8_t chrom[1] = { '1' };
    uint8_t allele1[1] = { 'A' };
    uint8_t allele2[1] = { 'T' };
    int32_t position[1] = { 1 };
    int8_t genotypes[2] = { -2, 0 };
    uint8_t phased[1] = { 0 };
    chunk_slice_args_t args = { 1, 1, 2, 1, 1, 1, 1, varid, rsid, chrom, allele1,
        allele2, position, genotypes, phased };
    size_t bpv = vcz_bgen_variant_block_size(1, 2, 1, 1, 1, 1);
    uint8_t *out = malloc(bpv);
    int rc;

    CU_ASSERT_FATAL(out != NULL);
    rc = call_chunk_slice(&args, out);
    CU_ASSERT_EQUAL_FATAL(rc, VCZ_ERR_BGEN_INVALID_PLOIDY);
    free(out);
}

static void
test_bgen_chunk_slice_invalid_allele(void)
{
    /* Multi-allelic index 2 in genotypes (beyond {-2,-1,0,1}) is
     * rejected with VCZ_ERR_BGEN_INVALID_ALLELE. */
    uint8_t varid[1] = { '.' };
    uint8_t rsid[1] = { '.' };
    uint8_t chrom[1] = { '1' };
    uint8_t allele1[1] = { 'A' };
    uint8_t allele2[1] = { 'T' };
    int32_t position[1] = { 1 };
    int8_t genotypes[2] = { 0, 2 };
    uint8_t phased[1] = { 0 };
    chunk_slice_args_t args = { 1, 1, 2, 1, 1, 1, 1, varid, rsid, chrom, allele1,
        allele2, position, genotypes, phased };
    size_t bpv = vcz_bgen_variant_block_size(1, 2, 1, 1, 1, 1);
    uint8_t *out = malloc(bpv);
    int rc;

    CU_ASSERT_FATAL(out != NULL);
    rc = call_chunk_slice(&args, out);
    CU_ASSERT_EQUAL_FATAL(rc, VCZ_ERR_BGEN_INVALID_ALLELE);
    free(out);
}

static void
test_bgen_chunk_slice_mixed_ploidy(void)
{
    /* Encoder configured uniform_ploidy=2 but one sample is haploid
     * (b == -2): mixed-ploidy detection kicks in via the geno_size
     * mismatch and the kernel returns VCZ_ERR_BGEN_MIXED_PLOIDY. */
    uint8_t varid[1] = { '.' };
    uint8_t rsid[1] = { '.' };
    uint8_t chrom[1] = { '1' };
    uint8_t allele1[1] = { 'A' };
    uint8_t allele2[1] = { 'T' };
    int32_t position[1] = { 1 };
    /* 1 variant x 2 samples: first sample is diploid (0/0), second is
     * haploid (0/-2). */
    int8_t genotypes[2 * 2] = { 0, 0, 0, -2 };
    uint8_t phased[1] = { 0 };
    chunk_slice_args_t args = { 1, 2, 2, 1, 1, 1, 1, varid, rsid, chrom, allele1,
        allele2, position, genotypes, phased };
    size_t bpv = vcz_bgen_variant_block_size(2, 2, 1, 1, 1, 1);
    uint8_t *out = malloc(bpv);
    int rc;

    CU_ASSERT_FATAL(out != NULL);
    rc = call_chunk_slice(&args, out);
    CU_ASSERT_EQUAL_FATAL(rc, VCZ_ERR_BGEN_MIXED_PLOIDY);
    free(out);
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
        { "test_encode_plink_single_genotype", test_encode_plink_single_genotype },
        { "test_encode_plink_example", test_encode_plink_example },
        { "test_encode_plink_all_zeros", test_encode_plink_all_zeros },
        { "test_bgen_geno_blocks_single_sample_diploid_unphased",
            test_bgen_geno_blocks_single_sample_diploid_unphased },
        { "test_bgen_geno_blocks_single_sample_diploid_phased",
            test_bgen_geno_blocks_single_sample_diploid_phased },
        { "test_bgen_geno_blocks_single_sample_haploid",
            test_bgen_geno_blocks_single_sample_haploid },
        { "test_bgen_geno_blocks_single_sample_invalid_ploidy",
            test_bgen_geno_blocks_single_sample_invalid_ploidy },
        { "test_bgen_geno_blocks_single_sample_invalid_allele",
            test_bgen_geno_blocks_single_sample_invalid_allele },
        { "test_bgen_geno_blocks_invalid_allele_mid_chunk",
            test_bgen_geno_blocks_invalid_allele_mid_chunk },
        { "test_bgen_geno_blocks_header_bytes", test_bgen_geno_blocks_header_bytes },
        { "test_bgen_geno_blocks_zero_samples", test_bgen_geno_blocks_zero_samples },
        { "test_bgen_geno_blocks_zero_variants", test_bgen_geno_blocks_zero_variants },
        { "test_bgen_geno_blocks_example_unphased",
            test_bgen_geno_blocks_example_unphased },
        { "test_bgen_geno_blocks_example_phased", test_bgen_geno_blocks_example_phased },
        { "test_bgen_geno_blocks_example_missing",
            test_bgen_geno_blocks_example_missing },
        { "test_bgen_geno_blocks_example_haploid",
            test_bgen_geno_blocks_example_haploid },
        { "test_bgen_geno_blocks_example_mixed_ploidy",
            test_bgen_geno_blocks_example_mixed_ploidy },
        { "test_bgen_geno_blocks_invalid_ploidy_error",
            test_bgen_geno_blocks_invalid_ploidy_error },
        { "test_bgen_geno_blocks_mixed_phase", test_bgen_geno_blocks_mixed_phase },
        { "test_bgen_geno_blocks_large", test_bgen_geno_blocks_large },
        { "test_bgen_chunk_slice_zero_variants", test_bgen_chunk_slice_zero_variants },
        { "test_bgen_chunk_slice_single_variant_diploid",
            test_bgen_chunk_slice_single_variant_diploid },
        { "test_bgen_chunk_slice_multi_variant_offsets",
            test_bgen_chunk_slice_multi_variant_offsets },
        { "test_bgen_chunk_slice_uniform_haploid",
            test_bgen_chunk_slice_uniform_haploid },
        { "test_bgen_chunk_slice_invalid_ploidy", test_bgen_chunk_slice_invalid_ploidy },
        { "test_bgen_chunk_slice_invalid_allele", test_bgen_chunk_slice_invalid_allele },
        { "test_bgen_chunk_slice_mixed_ploidy", test_bgen_chunk_slice_mixed_ploidy },
        { NULL, NULL },
    };
    return test_main(tests, argc, argv);
}
