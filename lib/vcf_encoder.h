#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#ifdef __GNUC__
#define VCZ_UNUSED(x) VCZ_UNUSED_##x __attribute__((__unused__))
#else
#define VCZ_UNUSED(x) VCZ_UNUSED_##x
/* Don't bother with restrict for MSVC */
#define restrict
#endif

/* /1* We assume CHAR_BIT == 8 when loading strings from 8-bit byte arrays *1/ */
/* #if CHAR_BIT != 8 */
/* #error CHAR_BIT MUST EQUAL 8 */
/* #endif */

#define VCZ_INT_MISSING -1
#define VCZ_INT_FILL -2
#define VCZ_STRING_MISSING '.'
#define VCZ_STRING_FILL '\0'
#define VCZ_FLOAT32_MISSING_AS_INT32 0x7F800001
#define VCZ_FLOAT32_FILL_AS_INT32 0x7F800002

#define VCZ_NUM_FIXED_FIELDS 6

#define VCZ_TYPE_INT 1
#define VCZ_TYPE_FLOAT 2
#define VCZ_TYPE_STRING 3
#define VCZ_TYPE_BOOL 4

// arbitrary - we can increase if needs be
#define VCZ_MAX_FIELD_NAME_LEN 255
#define VCZ_INT32_BUF_SIZE 12 // 10 digits, leading '-' and terminating NULL
// Safe limit, no point in trying to make it too tight as it's easy to represent
// certain very large numbers of floating point.
#define VCZ_FLOAT32_BUF_SIZE 256

#define VCZ_ERR_NO_MEMORY (-100)
#define VCZ_ERR_BUFFER_OVERFLOW (-101)
#define VCZ_ERR_VARIANT_OUT_OF_BOUNDS (-102)

/* Built-in-limitations */
#define VCZ_ERR_FIELD_NAME_TOO_LONG (-201)
#define VCZ_ERR_FIELD_UNSUPPORTED_TYPE (-202)
#define VCZ_ERR_FIELD_UNSUPPORTED_ITEM_SIZE (-203)
#define VCZ_ERR_FIELD_UNSUPPORTED_NUM_COLUMNS (-204)

typedef struct {
    // maximum length + 1 for NULL byte
    char name[VCZ_MAX_FIELD_NAME_LEN + 1];
    size_t name_length;
    int type;
    size_t item_size;
    size_t num_columns;
    const char *data;
} vcz_field_t;

int vcz_field_init(vcz_field_t *self, const char *name, int type, size_t item_size,
    size_t num_columns, const void *data);
int64_t vcz_field_write_1d(
    const vcz_field_t *self, size_t row, char *buf, int64_t buflen, int64_t offset);
void vcz_field_print_state(const vcz_field_t *self, FILE *out);

typedef struct {
    size_t num_variants;
    size_t num_samples;
    vcz_field_t fixed_fields[VCZ_NUM_FIXED_FIELDS];
    vcz_field_t filter_id;
    const int8_t *filter_data;
    vcz_field_t gt;
    const int8_t *gt_phased_data;
    size_t num_info_fields;
    size_t max_info_fields;
    vcz_field_t *info_fields;
    size_t num_format_fields;
    size_t max_format_fields;
    size_t field_array_size_increment;
    vcz_field_t *format_fields;
} vcz_variant_encoder_t;

int vcz_variant_encoder_init(
    vcz_variant_encoder_t *self, size_t num_variants, size_t num_samples);
void vcz_variant_encoder_free(vcz_variant_encoder_t *self);
void vcz_variant_encoder_print_state(const vcz_variant_encoder_t *self, FILE *out);

int vcz_variant_encoder_add_chrom_field(
    vcz_variant_encoder_t *self, size_t item_size, const char *data);
int vcz_variant_encoder_add_pos_field(vcz_variant_encoder_t *self, const int32_t *data);
int vcz_variant_encoder_add_qual_field(vcz_variant_encoder_t *self, const float *data);
int vcz_variant_encoder_add_ref_field(
    vcz_variant_encoder_t *self, size_t item_size, const char *data);
int vcz_variant_encoder_add_id_field(
    vcz_variant_encoder_t *self, size_t item_size, size_t num_columns, const char *data);
int vcz_variant_encoder_add_alt_field(
    vcz_variant_encoder_t *self, size_t item_size, size_t num_columns, const char *data);
int vcz_variant_encoder_add_filter_field(vcz_variant_encoder_t *self,
    size_t id_item_size, size_t id_num_columns, const char *id_data,
    const int8_t *filter_data);
int vcz_variant_encoder_add_gt_field(vcz_variant_encoder_t *self, size_t item_size,
    size_t num_columns, const void *data, const int8_t *phased_data);
int vcz_variant_encoder_add_info_field(vcz_variant_encoder_t *self, const char *name,
    int type, size_t item_size, size_t num_columns, const void *data);
int vcz_variant_encoder_add_format_field(vcz_variant_encoder_t *self, const char *name,
    int type, size_t item_size, size_t num_columns, const void *data);

int64_t vcz_variant_encoder_encode(
    const vcz_variant_encoder_t *self, size_t row, char *buf, size_t buflen);

int vcz_itoa(char *buf, int64_t v);
int vcz_ftoa(char *buf, float v);
