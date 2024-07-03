#include "vcf_encoder.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>

int
vcz_itoa(int32_t value, char *buf)
{
    int p = 0;
    int j, k;

    if (value < 0) {
        buf[p] = '-';
        p++;
        value = -value;
    }
    /*  special case small values */
    if (value < 10) {
        buf[p] = value + '0';
        p++;
    } else {
        k = 0;
        if (value < 100) {
            k = 1;
        } else if (value < 1000) {
            k = 2;
        } else if (value < 10000) {
            k = 3;
        } else if (value < 100000) {
            k = 4;
        } else if (value < 1000000) {
            k = 5;
        } else if (value < 10000000) {
            k = 6;
        } else if (value < 100000000) {
            k = 7;
        } else if (value < 1000000000) {
            k = 8;
        }

        // iterate backwards in buf
        p += k;
        buf[p] = (value % 10) + '0';
        for (j = 0; j < k; j++) {
            p--;
            value = value / 10;
            buf[p] = (value % 10) + '0';
        }
        p += k + 1;
    }
    buf[p] = '\0';
    return p;
}

static int64_t
string_field_write_entry(
    const vcz_field_t *self, const void *data, char *dest, size_t buflen, int64_t offset)
{
    const char *source = (char *) data;
    size_t column, byte;
    const char sep = ',';
    int64_t source_offset = 0;

    for (column = 0; column < self->num_columns; column++) {
        if (column > 0 && source[source_offset] != '\0') {
            dest[offset] = sep;
            offset++;
        }
        for (byte = 0; byte < self->item_size; byte++) {
            if (source[source_offset] != '\0') {
                dest[offset] = source[source_offset];
                offset++;
            }
            source_offset++;
        }
    }
    dest[offset] = '\t';
    offset++;
    dest[offset] = '\0';
    return offset;
}

static int64_t
int32_field_write_entry(
    const vcz_field_t *self, const void *data, char *dest, size_t buflen, int64_t offset)
{
    const int32_t *source = (int32_t *) data;
    int32_t value;
    size_t column;
    /* int written; */
    /* char value_buffer[128]; */

    for (column = 0; column < self->num_columns; column++) {
        value = source[column];
        if (value != VCZ_INT_FILL) {
            if (column > 0) {
                dest[offset] = ',';
                offset++;
            }
            if (value == VCZ_INT_MISSING) {
                dest[offset] = '.';
                offset++;
            } else {
                offset += vcz_itoa(value, dest + offset);
                /* written = snprintf(value_buffer, sizeof(value_buffer), "%d", value);
                 */
                /* memcpy(dest + offset, value_buffer, written); */
                /* offset += written; */
            }
        }
    }
    dest[offset] = '\t';
    offset++;
    dest[offset] = '\0';
    return offset;
}

int64_t
vcz_field_write_entry(
    const vcz_field_t *self, const void *data, char *dest, size_t buflen, int64_t offset)
{

    if (self->type == VCZ_TYPE_INT) {
        if (self->item_size == 4) {
            return int32_field_write_entry(self, data, dest, buflen, offset);
        }
    } else if (self->type == VCZ_TYPE_STRING) {
        return string_field_write_entry(self, data, dest, buflen, offset);
    }
    return VCZ_ERR_FIELD_UNSUPPORTED_TYPE;
}

int64_t
vcz_field_write(
    const vcz_field_t *self, size_t variant, char *dest, size_t buflen, int64_t offset)
{

    size_t row_size = self->num_columns * self->item_size;
    const void *data = self->data + variant * row_size;

    return vcz_field_write_entry(self, data, dest, buflen, offset);
}

void
vcz_field_print_state(const vcz_field_t *self, FILE *out)
{
    fprintf(out, "\t%s\ttype:%d\titem_size=%d\tnum_columns=%d\tdata=%p\n", self->name,
        self->type, (int) self->item_size, (int) self->num_columns, self->data);
}

int64_t
vcz_variant_encoder_write_format_specifiers(
    const vcz_variant_encoder_t *self, char *dest, size_t buflen, int64_t offset)
{
    const int format_len = 7;
    size_t j;

    strcpy(dest + offset, "FORMAT=");
    offset += format_len;
    if (self->gt.data != NULL) {
        strcpy(dest + offset, "GT");
        offset += 2;
    }
    for (j = 0; j < self->num_format_fields; j++) {
        dest[offset] = ':';
        offset++;
        strcpy(dest + offset, self->format_fields[j].name);
        offset += strlen(self->format_fields[j].name);
    }
    dest[offset] = '\t';
    offset++;
    dest[offset] = '\0';
    return offset;
}

int64_t
vcz_variant_encoder_write_sample_gt(const vcz_variant_encoder_t *self, size_t variant,
    size_t sample, char *dest, size_t buflen, int64_t offset)
{
    const size_t ploidy = self->gt.num_columns;
    size_t source_offset = variant * self->num_samples * ploidy + sample * ploidy;
    const int32_t *source = ((int32_t *) self->gt.data) + source_offset;
    const bool phased = self->gt_phased_data[variant * self->num_samples + sample];
    int32_t value;
    size_t ploid;
    char sep = phased ? '|' : '/';

    if (self->gt.item_size != 4) {
        offset = VCZ_ERR_FIELD_UNSUPPORTED_TYPE;
        goto out;
    }

    for (ploid = 0; ploid < ploidy; ploid++) {
        value = source[ploid];
        if (value != VCZ_INT_FILL) {
            if (ploid > 0) {
                dest[offset] = sep;
                offset++;
            }
            if (value == VCZ_INT_MISSING) {
                dest[offset] = '.';
                offset++;
            } else {
                offset += vcz_itoa(value, dest + offset);
                /* written = snprintf(value_buffer, sizeof(value_buffer), "%d", value);
                 */
                /* memcpy(dest + offset, value_buffer, written); */
                /* offset += written; */
            }
        }
    }
    dest[offset] = '\t';
    offset++;
    dest[offset] = '\0';
out:
    return offset;
}

int64_t
vcz_variant_encoder_write_format_fields(const vcz_variant_encoder_t *self,
    size_t variant, size_t sample, char *dest, size_t buflen, int64_t offset)
{
    vcz_field_t field;
    size_t j, row_size;
    const void *data;

    if (self->gt.data != NULL) {
        offset = vcz_variant_encoder_write_sample_gt(
            self, variant, sample, dest, buflen, offset);
        if (offset < 0) {
            goto out;
        }
    }

    for (j = 0; j < self->num_format_fields; j++) {
        field = self->format_fields[j];
        dest[offset - 1] = ':';
        row_size = self->num_samples * field.num_columns * field.item_size;
        data = field.data + variant * row_size
               + sample * field.num_columns * field.item_size;
        offset = vcz_field_write_entry(&field, data, dest, buflen, offset);
        if (offset < 0) {
            goto out;
        }
    }
out:
    return offset;
}

int64_t
vcz_variant_encoder_write_info_fields(const vcz_variant_encoder_t *self, size_t variant,
    char *dest, size_t buflen, int64_t offset)
{
    vcz_field_t field;
    size_t j;

    if (self->num_info_fields == 0) {
        dest[offset] = '.';
        offset++;
        dest[offset] = '\t';
        offset++;
    }
    for (j = 0; j < self->num_info_fields; j++) {
        if (j > 0) {
            dest[offset - 1] = ';';
        }
        field = self->info_fields[j];
        memcpy(dest + offset, field.name, field.name_length);
        offset += field.name_length;
        dest[offset] = '=';
        offset++;
        offset = vcz_field_write(&field, variant, dest, buflen, offset);
        if (offset < 0) {
            goto out;
        }
    }
out:
    return offset;
}

int64_t
vcz_variant_encoder_write_row(
    const vcz_variant_encoder_t *self, size_t row, char *buf, size_t buflen)
{
    int64_t offset = 0;
    size_t j;

    for (j = 0; j < VCZ_NUM_FIXED_FIELDS; j++) {
        offset = vcz_field_write(&self->fixed_fields[j], row, buf, buflen, offset);
        if (offset < 0) {
            goto out;
        }
    }
    offset = vcz_variant_encoder_write_info_fields(self, row, buf, buflen, offset);
    if (offset < 0) {
        goto out;
    }
    if (self->num_samples > 0) {
        offset = vcz_variant_encoder_write_format_specifiers(self, buf, buflen, offset);
        if (offset < 0) {
            goto out;
        }
        for (j = 0; j < self->num_samples; j++) {
            /* printf("Run sample %d\n", (int) j); */
            offset = vcz_variant_encoder_write_format_fields(
                self, row, j, buf, buflen, offset);
            if (offset < 0) {
                goto out;
            }
        }
    }
    offset--;
    buf[offset] = '\0';
out:
    return offset;
}

void
vcz_variant_encoder_print_state(const vcz_variant_encoder_t *self, FILE *out)
{
    size_t j;

    fprintf(out, "vcz_variant_encoder: %p\n", (void *) self);
    fprintf(out, "\tnum_samples: %d\n", (int) self->num_samples);
    fprintf(out, "\tnum_variants: %d\n", (int) self->num_variants);
    for (j = 0; j < VCZ_NUM_FIXED_FIELDS; j++) {
        vcz_field_print_state(&self->fixed_fields[j], out);
    }
    fprintf(out, "\tINFO:\n");
    for (j = 0; j < self->num_info_fields; j++) {
        vcz_field_print_state(&self->info_fields[j], out);
    }
    fprintf(out, "\tFORMAT:\n");
    if (self->gt.data != NULL) {
        vcz_field_print_state(&self->gt, out);
    }
    for (j = 0; j < self->num_format_fields; j++) {
        vcz_field_print_state(&self->format_fields[j], out);
    }
}

static int
vcz_variant_encoder_add_field(vcz_variant_encoder_t *self, vcz_field_t *field,
    const char *name, int type, size_t item_size, size_t num_columns, const void *data)
{
    int ret = 0;

    field->name_length = strlen(name);
    if (field->name_length >= VCZ_MAX_FIELD_NAME_LEN) {
        ret = VCZ_ERR_FIELD_NAME_TOO_LONG;
        goto out;
    }
    strcpy(field->name, name);
    field->type = type;
    field->item_size = item_size;
    field->num_columns = num_columns;
    field->data = data;
out:
    return ret;
}

int
vcz_variant_encoder_add_info_field(vcz_variant_encoder_t *self, const char *name,
    int type, size_t item_size, size_t num_columns, const void *data)
{
    int ret = 0;
    vcz_field_t *tmp, *field;

    if (self->num_info_fields == self->max_info_fields) {
        self->max_info_fields += self->field_array_size_increment;
        tmp = realloc(
            self->info_fields, self->max_info_fields * sizeof(*self->info_fields));
        if (tmp == NULL) {
            ret = VCZ_ERR_NO_MEMORY;
            goto out;
        }
        self->info_fields = tmp;
    }
    field = self->info_fields + self->num_info_fields;
    self->num_info_fields++;
    ret = vcz_variant_encoder_add_field(
        self, field, name, type, item_size, num_columns, data);
out:
    return ret;
}

int
vcz_variant_encoder_add_format_field(vcz_variant_encoder_t *self, const char *name,
    int type, size_t item_size, size_t num_columns, const void *data)
{
    int ret = 0;
    vcz_field_t *tmp, *field;

    if (self->num_format_fields == self->max_format_fields) {
        self->max_format_fields += self->field_array_size_increment;
        /* NOTE: assuming realloc(NULL) is safe and portable. check */
        tmp = realloc(
            self->format_fields, self->max_format_fields * sizeof(*self->format_fields));
        if (tmp == NULL) {
            ret = VCZ_ERR_NO_MEMORY;
            goto out;
        }
        self->format_fields = tmp;
    }
    field = self->format_fields + self->num_format_fields;
    self->num_format_fields++;

    ret = vcz_variant_encoder_add_field(
        self, field, name, type, item_size, num_columns, data);
out:
    return ret;
}

int
vcz_variant_encoder_add_gt_field(vcz_variant_encoder_t *self, const void *gt_data,
    size_t gt_item_size, size_t ploidy, const int8_t *gt_phased_data)
{
    strcpy(self->gt.name, "GT");
    self->gt.item_size = gt_item_size;
    self->gt.data = gt_data;
    self->gt.num_columns = ploidy;
    self->gt_phased_data = gt_phased_data;
    return 0;
}

// clang-format off
int
vcz_variant_encoder_init(vcz_variant_encoder_t *self,
    size_t num_samples, size_t num_variants,
    const char *chrom_data, size_t chrom_item_size,
    const int32_t *pos_data,
    const char *id_data, size_t id_item_size, size_t id_num_columns,
    const char *ref_data, size_t ref_item_size,
    const char *alt_data, size_t alt_item_size, size_t alt_num_columns,
    const int32_t *qual_data,
    const char *filter_data, size_t filter_item_size, size_t filter_num_columns)

{
    vcz_field_t fixed_fields[] = {
        { .name = "CHROM",
            .type = VCZ_TYPE_STRING,
            .item_size = chrom_item_size,
            .num_columns = 1,
            .data = chrom_data },
        { .name = "POS",
            .type = VCZ_TYPE_INT,
            .item_size = 4,
            .num_columns = 1,
            .data = pos_data },
        { .name = "ID",
            .type = VCZ_TYPE_STRING,
            .item_size = id_item_size,
            .num_columns = id_num_columns,
            .data = id_data },
        { .name = "REF",
            .type = VCZ_TYPE_STRING,
            .item_size = ref_item_size,
            .num_columns = 1,
            .data = ref_data },
        { .name = "ALT",
            .type = VCZ_TYPE_STRING,
            .item_size = alt_item_size,
            .num_columns = alt_num_columns,
            .data = alt_data },
        { .name = "QUAL",
            .type = VCZ_TYPE_INT,
            .item_size = 4,
            .num_columns = 1,
            .data = qual_data },
        { .name = "FILTER",
            .type = VCZ_TYPE_STRING,
            .item_size = filter_item_size,
            .num_columns = filter_num_columns,
            .data = filter_data } };

    // clang-format on

    memset(self, 0, sizeof(*self));
    self->num_samples = num_samples;
    self->num_variants = num_variants;
    self->field_array_size_increment = 64;
    memcpy(self->fixed_fields, fixed_fields, sizeof(fixed_fields));

    return 0;
}

void
vcz_variant_encoder_free(vcz_variant_encoder_t *self)
{
    if (self->info_fields != NULL) {
        free(self->info_fields);
        self->info_fields = NULL;
    }
    if (self->format_fields != NULL) {
        free(self->format_fields);
        self->format_fields = NULL;
    }
}
