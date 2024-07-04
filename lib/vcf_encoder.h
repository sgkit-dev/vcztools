#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#define VCZ_INT_MISSING -1
#define VCZ_INT_FILL -2
#define VCZ_STRING_MISSING '.'
#define VCZ_STRING_FILL '\0'

#define VCZ_NUM_FIXED_FIELDS 6

#define VCZ_TYPE_INT 0
#define VCZ_TYPE_FLOAT 0
#define VCZ_TYPE_STRING 2

// arbitrary - we can increase if needs be
#define VCZ_MAX_FIELD_NAME_LEN 256

#define VCZ_ERR_NO_MEMORY (-100)

/* Built-in-limitations */
#define VCZ_ERR_FIELD_NAME_TOO_LONG (-201)
#define VCZ_ERR_FIELD_UNSUPPORTED_TYPE (-202)

typedef struct {
    char name[VCZ_MAX_FIELD_NAME_LEN];
    size_t name_length;
    int type;
    size_t item_size;
    size_t num_columns;
    const void *data;
} vcz_field_t;

int64_t vcz_field_write(
    const vcz_field_t *self, size_t row, char *buf, size_t buflen, int64_t offset);
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

// clang-format off
int vcz_variant_encoder_init(vcz_variant_encoder_t *self,
    size_t num_variants, size_t num_samples,
    const char *contig_data, size_t contig_item_size,
    const int32_t *position_data,
    const char *id_data, size_t id_item_size, size_t id_num_columns,
    const char *ref_data, size_t ref_item_size,
    const char *alt_data, size_t alt_item_size, size_t alt_num_columns,
    const int32_t *qual_data,
    const char *filter_id_data, size_t filter_id_item_size, size_t filter_id_num_columns,
    const int8_t *filter_data
);
// clang-format on

void vcz_variant_encoder_free(vcz_variant_encoder_t *self);
void vcz_variant_encoder_print_state(const vcz_variant_encoder_t *self, FILE *out);
int vcz_variant_encoder_add_gt_field(vcz_variant_encoder_t *self,
    const void *gt_data, size_t gt_item_size, size_t ploidy,
    const int8_t *gt_phased_data);
int vcz_variant_encoder_add_format_field(vcz_variant_encoder_t *self,
    const char *name, int type,
    size_t item_size, size_t num_columns, const void *data);
int vcz_variant_encoder_add_info_field(vcz_variant_encoder_t *self,
    const char *name, int type,
    size_t item_size, size_t num_columns, const void *data);
int64_t vcz_variant_encoder_write_row(
    const vcz_variant_encoder_t *self, size_t row, char *buf, size_t buflen);

int vcz_itoa(int32_t v, char *out);
