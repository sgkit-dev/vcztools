#include "vcf_encoder.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>

int
vcz_itoa(char *restrict buf, int64_t value)
{
    int p = 0;
    int j, k;

    if (value < 0) {
        buf[p] = '-';
        p++;
        value = -value;
    }
    /* We only support int32_t values. The +1 here is for supporting the
     * float converter below */
    assert(value <= (1LL + INT32_MAX));
    /*  special case small values */
    if (value < 10) {
        buf[p] = (char) value + '0';
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
        } else if (value < 10000000000) {
            // Largest possible INT32 value
            k = 9;
        }

        // iterate backwards in buf
        p += k;
        buf[p] = (char) (value % 10) + '0';
        for (j = 0; j < k; j++) {
            p--;
            value = value / 10;
            buf[p] = (char) (value % 10) + '0';
        }
        p += k + 1;
    }
    buf[p] = '\0';
    return p;
}

int
vcz_ftoa(char *restrict buf, float value)
{
    int p = 0;
    int64_t i, d1, d2, d3;

    if (!isfinite(value) || fabs(value) > INT32_MAX + 1LL) {
        return sprintf(buf, "%.3f", value);
    }

    if (value < 0) {
        buf[p] = '-';
        p++;
        value = -value;
    }

    /* integer part */
    i = (int64_t) round(((double) value) * 1000);
    p += vcz_itoa(buf + p, i / 1000);

    /* fractional part */
    d3 = i % 10;
    d2 = (i / 10) % 10;
    d1 = (i / 100) % 10;
    if (d1 + d2 + d3 > 0) {
        buf[p] = '.';
        p++;
        buf[p] = (char) d1 + '0';
        p++;
        if (d2 + d3 > 0) {
            buf[p] = (char) d2 + '0';
            p++;
            if (d3 > 0) {
                buf[p] = (char) d3 + '0';
                p++;
            }
        }
    }
    buf[p] = '\0';
    return p;
}

static inline int64_t
append_string(char *restrict buf, const char *restrict str, int64_t len, int64_t offset,
    int64_t buflen)
{
    if (offset + len > buflen) {
        /* printf("offset = %d len=%d buflen = %d\n", (int) offset, (int) len, (int)
         * buflen); */
        return VCZ_ERR_BUFFER_OVERFLOW;
    }
    memcpy(buf + offset, str, (size_t) len);
    return offset + len;
}

static inline int64_t
append_char(char *restrict buf, char c, int64_t offset, int64_t buflen)
{
    if (offset == buflen) {
        return VCZ_ERR_BUFFER_OVERFLOW;
    }
    buf[offset] = c;
    return offset + 1;
}

static inline int64_t
append_int(char *restrict buf, int32_t value, int64_t offset, int64_t buflen)
{
    char tmp[VCZ_INT32_BUF_SIZE];
    int64_t len;

    if (value == VCZ_INT_MISSING) {
        return append_char(buf, '.', offset, buflen);
    }
    /* Note: it would be slightly more efficient to write directly into the output
     * buffer here, but doing it this way makes it easier to test that we're correctly
     * catching all buffer overflows */
    len = vcz_itoa(tmp, value);
    /* printf("%d: %d\n", value, (int) len); */
    return append_string(buf, tmp, len, offset, buflen);
}

static inline int64_t
append_float(
    char *restrict buf, int32_t int32_value, float value, int64_t offset, int64_t buflen)
{
    char tmp[VCZ_FLOAT32_BUF_SIZE];
    int64_t len;

    if (int32_value == VCZ_FLOAT32_MISSING_AS_INT32) {
        return append_char(buf, '.', offset, buflen);
    }
    len = vcz_ftoa(tmp, value);
    /* printf("%f: %d\n", value, (int) len); */
    return append_string(buf, tmp, len, offset, buflen);
}

static bool
bool_all_missing(const int8_t *restrict data, size_t n)
{
    assert(n == 1);
    return !data[0];
}

static bool
int8_all_missing(const int8_t *restrict data, size_t n)
{
    size_t j;

    for (j = 0; j < n; j++) {
        if (data[j] != VCZ_INT_FILL && data[j] != VCZ_INT_MISSING) {
            return false;
        }
    }
    return true;
}

static bool
int16_all_missing(const int16_t *restrict data, size_t n)
{
    size_t j;

    for (j = 0; j < n; j++) {
        if (data[j] != VCZ_INT_FILL && data[j] != VCZ_INT_MISSING) {
            return false;
        }
    }
    return true;
}

static bool
int32_all_missing(const int32_t *restrict data, size_t n)
{
    size_t j;

    for (j = 0; j < n; j++) {
        if (data[j] != VCZ_INT_FILL && data[j] != VCZ_INT_MISSING) {
            return false;
        }
    }
    return true;
}

static bool
float32_all_missing(const float *restrict data, size_t n)
{
    size_t j;
    const int32_t *restrict di32 = (const int32_t *) data;

    for (j = 0; j < n; j++) {
        if (di32[j] != VCZ_FLOAT32_FILL_AS_INT32
            && di32[j] != VCZ_FLOAT32_MISSING_AS_INT32) {
            return false;
        }
    }
    return true;
}

static bool
string_all_missing(const char *restrict data, size_t item_size, size_t n)
{
    size_t j;

    for (j = 0; j < n * item_size; j++) {
        if (data[j] != VCZ_STRING_FILL && data[j] != VCZ_STRING_MISSING) {
            return false;
        }
    }
    return true;
}

static int64_t
string_write_entry(size_t num_columns, size_t item_size, const void *data, char *buf,
    int64_t buflen, int64_t offset)
{
    const char *source = (const char *) data;
    size_t column, byte;
    const char sep = ',';
    int64_t source_offset = 0;

    for (column = 0; column < num_columns; column++) {
        if (column > 0 && source[source_offset] != '\0') {
            offset = append_char(buf, sep, offset, buflen);
            if (offset < 0) {
                goto out;
            }
        }
        for (byte = 0; byte < item_size; byte++) {
            if (source[source_offset] != '\0') {
                offset = append_char(buf, source[source_offset], offset, buflen);
                if (offset < 0) {
                    goto out;
                }
            }
            source_offset++;
        }
    }
out:
    return offset;
}

static int64_t
int8_write_entry(size_t num_columns, const void *restrict data, char *buf,
    int64_t buflen, int64_t offset, char separator)
{
    const int8_t *restrict source = (const int8_t *) data;
    size_t column;

    for (column = 0; column < num_columns; column++) {
        if (source[column] == VCZ_INT_FILL) {
            break;
        }
        if (column > 0) {
            offset = append_char(buf, separator, offset, buflen);
            if (offset < 0) {
                goto out;
            }
        }
        offset = append_int(buf, source[column], offset, buflen);
        if (offset < 0) {
            goto out;
        }
    }
out:
    return offset;
}

static int64_t
int16_write_entry(size_t num_columns, const void *restrict data, char *buf,
    int64_t buflen, int64_t offset, char separator)
{
    const int16_t *restrict source = (const int16_t *) data;
    size_t column;

    for (column = 0; column < num_columns; column++) {
        if (source[column] == VCZ_INT_FILL) {
            break;
        }
        if (column > 0) {
            offset = append_char(buf, separator, offset, buflen);
            if (offset < 0) {
                goto out;
            }
        }
        offset = append_int(buf, source[column], offset, buflen);
        if (offset < 0) {
            goto out;
        }
    }
out:
    return offset;
}

static int64_t
int32_write_entry(size_t num_columns, const void *restrict data, char *buf,
    int64_t buflen, int64_t offset, char separator)
{
    const int32_t *restrict source = (const int32_t *) data;
    size_t column;

    for (column = 0; column < num_columns; column++) {
        if (source[column] == VCZ_INT_FILL) {
            break;
        }
        if (column > 0) {
            offset = append_char(buf, separator, offset, buflen);
            if (offset < 0) {
                goto out;
            }
        }
        offset = append_int(buf, source[column], offset, buflen);
        if (offset < 0) {
            goto out;
        }
    }
out:
    return offset;
}

static int64_t
float32_write_entry(size_t num_columns, const void *restrict data, char *restrict buf,
    int64_t buflen, int64_t offset)
{
    const float *restrict source = (const float *restrict) data;
    const int32_t *restrict int32_source = (const int32_t *restrict) data;
    int32_t int32_value;
    size_t column;

    for (column = 0; column < num_columns; column++) {
        int32_value = int32_source[column];
        if (int32_value == VCZ_FLOAT32_FILL_AS_INT32) {
            break;
        }
        if (column > 0) {
            offset = append_char(buf, ',', offset, buflen);
            if (offset < 0) {
                goto out;
            }
        }
        offset = append_float(buf, int32_value, source[column], offset, buflen);
        if (offset < 0) {
            goto out;
        }
    }
out:
    return offset;
}

static int64_t
write_entry(int type, size_t item_size, size_t num_columns, const void *data, char *buf,
    int64_t buflen, int64_t offset)
{
    if (type == VCZ_TYPE_INT) {
        switch (item_size) {
            case 1:
                return int8_write_entry(num_columns, data, buf, buflen, offset, ',');
            case 2:
                return int16_write_entry(num_columns, data, buf, buflen, offset, ',');
            default:
                assert(item_size == 4);
                return int32_write_entry(num_columns, data, buf, buflen, offset, ',');
        }
    } else if (type == VCZ_TYPE_FLOAT) {
        assert(item_size == 4);
        return float32_write_entry(num_columns, data, buf, buflen, offset);
    } else if (type == VCZ_TYPE_BOOL) {
        assert(item_size == 1);
        assert(num_columns == 1);
        /* Erase the previously inserted "=" we need for other fields.*/
        return offset - 1;
    }
    assert(type == VCZ_TYPE_STRING);
    return string_write_entry(num_columns, item_size, data, buf, buflen, offset);
}

static bool
all_missing(int type, size_t item_size, size_t n, const char *restrict data)
{
    if (type == VCZ_TYPE_INT) {
        switch (item_size) {
            case 1:
                return int8_all_missing((const int8_t *) data, n);
            case 2:
                return int16_all_missing((const int16_t *) data, n);
            default:
                assert(item_size == 4);
                return int32_all_missing((const int32_t *) data, n);
        }
    } else if (type == VCZ_TYPE_FLOAT) {
        assert(item_size == 4);
        return float32_all_missing((const float *) data, n);
    } else if (type == VCZ_TYPE_BOOL) {
        assert(item_size == 1);
        return bool_all_missing((const int8_t *) data, n);
    }
    assert(type == VCZ_TYPE_STRING);
    return string_all_missing(data, item_size, n);
}

int64_t
vcz_field_write_1d(
    const vcz_field_t *self, size_t variant, char *buf, int64_t buflen, int64_t offset)
{
    size_t row_size = self->num_columns * self->item_size;
    const void *data = self->data + variant * row_size;

    return write_entry(
        self->type, self->item_size, self->num_columns, data, buf, buflen, offset);
}

static bool
vcz_field_is_missing_1d(const vcz_field_t *self, size_t variant)
{
    size_t row_size = self->num_columns * self->item_size;
    const void *data = self->data + variant * row_size;

    return all_missing(self->type, self->item_size, self->num_columns, data);
}

static int64_t
vcz_field_write_2d(const vcz_field_t *self, size_t variant, size_t num_samples,
    size_t sample, char *buf, int64_t buflen, int64_t offset)
{
    size_t row_size = self->num_columns * self->item_size * num_samples;
    const void *data
        = self->data + variant * row_size + sample * self->num_columns * self->item_size;
    return write_entry(
        self->type, self->item_size, self->num_columns, data, buf, buflen, offset);
}

static bool
vcz_field_is_missing_2d(const vcz_field_t *self, size_t variant, size_t num_samples)
{
    size_t row_size = self->num_columns * self->item_size * num_samples;
    const void *data = self->data + variant * row_size;

    return all_missing(
        self->type, self->item_size, self->num_columns * num_samples, data);
}

void
vcz_field_print_state(const vcz_field_t *self, FILE *out)
{
    fprintf(out, "\t%s\ttype:%d\titem_size=%d\tnum_columns=%d\tdata=%p\n", self->name,
        self->type, (int) self->item_size, (int) self->num_columns, self->data);
}

int
vcz_field_init(vcz_field_t *self, const char *name, int type, size_t item_size,
    size_t num_columns, const void *data)
{
    int ret = 0;

    self->name_length = strlen(name);
    if (self->name_length > VCZ_MAX_FIELD_NAME_LEN) {
        ret = VCZ_ERR_FIELD_NAME_TOO_LONG;
        goto out;
    }
    if (!(type == VCZ_TYPE_INT || type == VCZ_TYPE_BOOL || type == VCZ_TYPE_STRING
            || type == VCZ_TYPE_FLOAT)) {
        ret = VCZ_ERR_FIELD_UNSUPPORTED_TYPE;
        goto out;
    }
    if (item_size <= 0) {
        ret = VCZ_ERR_FIELD_UNSUPPORTED_ITEM_SIZE;
        goto out;
    }
    if (num_columns <= 0) {
        ret = VCZ_ERR_FIELD_UNSUPPORTED_NUM_COLUMNS;
        goto out;
    }

    if (type == VCZ_TYPE_BOOL) {
        if (item_size != 1) {
            ret = VCZ_ERR_FIELD_UNSUPPORTED_ITEM_SIZE;
            goto out;
        }
        if (num_columns != 1) {
            ret = VCZ_ERR_FIELD_UNSUPPORTED_NUM_COLUMNS;
            goto out;
        }
    } else if (type == VCZ_TYPE_INT) {
        switch (item_size) {
            case 1:
            case 2:
            case 4:
                break;
            default:
                ret = VCZ_ERR_FIELD_UNSUPPORTED_ITEM_SIZE;
                goto out;
        }
    } else if (type == VCZ_TYPE_FLOAT) {
        if (item_size != 4) {
            ret = VCZ_ERR_FIELD_UNSUPPORTED_ITEM_SIZE;
            goto out;
        }
    } else {
        assert(type == VCZ_TYPE_STRING);
    }

    // strcpy includes the terminating NULL byte.
    strcpy(self->name, name);
    self->type = type;
    self->item_size = item_size;
    self->num_columns = num_columns;
    /* NOTE: we don't make any checks on data because NULL is a valid input if the
     * number of variants is zero */
    self->data = data;
out:
    return ret;
}

static int64_t
vcz_variant_encoder_write_sample_gt(const vcz_variant_encoder_t *self, size_t variant,
    size_t sample, char *buf, int64_t buflen, int64_t offset)
{
    const vcz_field_t gt = self->gt;
    size_t row_size = gt.num_columns * gt.item_size * self->num_samples;
    const void *data
        = gt.data + variant * row_size + sample * gt.num_columns * gt.item_size;
    const bool phased = self->gt_phased_data[variant * self->num_samples + sample];
    char sep = phased ? '|' : '/';

    switch (gt.item_size) {
        case 1:
            return int8_write_entry(gt.num_columns, data, buf, buflen, offset, sep);
        case 2:
            return int16_write_entry(gt.num_columns, data, buf, buflen, offset, sep);
        default:
            assert(gt.item_size == 4);
            return int32_write_entry(gt.num_columns, data, buf, buflen, offset, sep);
    }
}

static int64_t
vcz_variant_encoder_write_info_fields(const vcz_variant_encoder_t *self, size_t variant,
    char *buf, int64_t buflen, int64_t offset)
{
    vcz_field_t field;
    size_t j;
    bool *missing = NULL;
    bool all_missing = true;
    bool first_field;

    if (self->num_info_fields > 0) {
        missing = malloc(self->num_info_fields * sizeof(*missing));
        if (missing == NULL) {
            offset = VCZ_ERR_NO_MEMORY; // GCOVR_EXCL_LINE
            goto out;                   // GCOVR_EXCL_LINE
        }
        for (j = 0; j < self->num_info_fields; j++) {
            missing[j] = vcz_field_is_missing_1d(&self->info_fields[j], variant);
            if (!missing[j]) {
                all_missing = false;
            }
        }
    }

    if (all_missing) {
        offset = append_string(buf, ".\t", 2, offset, buflen);
        if (offset < 0) {
            goto out;
        }
    } else {
        first_field = true;
        for (j = 0; j < self->num_info_fields; j++) {
            if (!missing[j]) {
                if (!first_field) {
                    buf[offset - 1] = ';';
                }
                first_field = false;
                field = self->info_fields[j];
                offset = append_string(
                    buf, field.name, (int64_t) field.name_length, offset, buflen);
                if (offset < 0) {
                    goto out;
                }
                offset = append_char(buf, '=', offset, buflen);
                if (offset < 0) {
                    goto out;
                }
                offset = vcz_field_write_1d(&field, variant, buf, buflen, offset);
                if (offset < 0) {
                    goto out;
                }
                offset = append_char(buf, '\t', offset, buflen);
                if (offset < 0) {
                    goto out;
                }
            }
        }
    }
out:
    if (missing != NULL) {
        free(missing);
    }
    return offset;
}

static int64_t
vcz_variant_encoder_write_format_fields(const vcz_variant_encoder_t *self,
    size_t variant, char *buf, int64_t buflen, int64_t offset)
{
    size_t j, sample;
    bool *missing = NULL;
    bool all_missing = true;
    bool has_gt = (self->gt.data != NULL);
    bool gt_missing = true;
    vcz_field_t field;
    const size_t num_samples = self->num_samples;

    if (has_gt) {
        gt_missing = vcz_field_is_missing_2d(&self->gt, variant, num_samples);
    }

    if (self->num_format_fields > 0) {
        missing = malloc(self->num_format_fields * sizeof(*missing));
        if (missing == NULL) {
            offset = VCZ_ERR_NO_MEMORY; // GCOVR_EXCL_LINE
            goto out;                   // GCOVR_EXCL_LINE
        }
        for (j = 0; j < self->num_format_fields; j++) {
            missing[j]
                = vcz_field_is_missing_2d(&self->format_fields[j], variant, num_samples);
            if (!missing[j]) {
                all_missing = false;
            }
        }
    }
    all_missing = all_missing && gt_missing;

    if (all_missing) {
        for (j = 0; j < num_samples + 1; j++) {
            if (num_samples > 0) {
                offset = append_string(buf, ".\t", 2, offset, buflen);
                if (offset < 0) {
                    goto out;
                }
            }
        }
    } else {
        if (!gt_missing) {
            offset = append_string(buf, "GT:", 3, offset, buflen);
            if (offset < 0) {
                goto out;
            }
        }
        for (j = 0; j < self->num_format_fields; j++) {
            if (!missing[j]) {
                field = self->format_fields[j];
                offset = append_string(
                    buf, field.name, (int64_t) field.name_length, offset, buflen);
                if (offset < 0) {
                    goto out;
                }
                offset = append_char(buf, ':', offset, buflen);
                if (offset < 0) {
                    goto out;
                }
            }
        }
        buf[offset - 1] = '\t';

        for (sample = 0; sample < num_samples; sample++) {
            if (!gt_missing) {
                offset = vcz_variant_encoder_write_sample_gt(
                    self, variant, sample, buf, buflen, offset);
                if (offset < 0) {
                    goto out;
                }
                offset = append_char(buf, ':', offset, buflen);
                if (offset < 0) {
                    goto out;
                }
            }
            for (j = 0; j < self->num_format_fields; j++) {
                if (!missing[j]) {
                    field = self->format_fields[j];
                    offset = vcz_field_write_2d(&self->format_fields[j], variant,
                        num_samples, sample, buf, buflen, offset);
                    if (offset < 0) {
                        goto out;
                    }
                    offset = append_char(buf, ':', offset, buflen);
                    if (offset < 0) {
                        goto out;
                    }
                }
            }
            buf[offset - 1] = '\t';
        }
    }
out:
    if (missing != NULL) {
        free(missing);
    }
    return offset;
}

static int64_t
vcz_variant_encoder_write_filter(const vcz_variant_encoder_t *self, size_t variant,
    char *buf, int64_t buflen, int64_t offset)
{
    const vcz_field_t filter_id = self->filter_id;
    bool all_missing = true;
    bool first = true;
    const int8_t *restrict data = self->filter_data + (variant * filter_id.num_columns);
    const char *filter_id_data = (const char *) self->filter_id.data;
    size_t j, k, source_offset;

    for (j = 0; j < filter_id.num_columns; j++) {
        if (data[j]) {
            all_missing = false;
        }
    }
    if (all_missing) {
        offset = append_char(buf, '.', offset, buflen);
        if (offset < 0) {
            goto out;
        }
    } else {
        source_offset = 0;
        for (j = 0; j < filter_id.num_columns; j++) {
            if (data[j]) {
                source_offset = j * filter_id.item_size;
                if (!first) {
                    offset = append_char(buf, ';', offset, buflen);
                    if (offset < 0) {
                        goto out;
                    }
                }
                for (k = 0; k < filter_id.item_size; k++) {
                    if (filter_id_data[source_offset] == VCZ_STRING_FILL) {
                        break;
                    }
                    offset = append_char(
                        buf, filter_id_data[source_offset], offset, buflen);
                    if (offset < 0) {
                        goto out;
                    }
                    source_offset++;
                    first = false;
                }
            }
        }
    }
    offset = append_char(buf, '\t', offset, buflen);
    if (offset < 0) {
        goto out;
    }
out:
    return offset;
}

int64_t
vcz_variant_encoder_encode(
    const vcz_variant_encoder_t *self, size_t variant, char *buf, size_t buflen)
{
    int64_t offset = 0;
    size_t j;

    if (variant >= self->num_variants) {
        offset = VCZ_ERR_VARIANT_OUT_OF_BOUNDS;
        goto out;
    }

    for (j = 0; j < VCZ_NUM_FIXED_FIELDS; j++) {
        if (vcz_field_is_missing_1d(&self->fixed_fields[j], variant)) {
            offset = append_char(buf, '.', offset, (int64_t) buflen);
            if (offset < 0) {
                goto out;
            }
        } else {
            offset = vcz_field_write_1d(
                &self->fixed_fields[j], variant, buf, (int64_t) buflen, offset);
            if (offset < 0) {
                goto out;
            }
        }
        offset = append_char(buf, '\t', offset, (int64_t) buflen);
        if (offset < 0) {
            goto out;
        }
    }
    offset
        = vcz_variant_encoder_write_filter(self, variant, buf, (int64_t) buflen, offset);
    if (offset < 0) {
        goto out;
    }
    offset = vcz_variant_encoder_write_info_fields(
        self, variant, buf, (int64_t) buflen, offset);
    if (offset < 0) {
        goto out;
    }
    offset = vcz_variant_encoder_write_format_fields(
        self, variant, buf, (int64_t) buflen, offset);
    if (offset < 0) {
        goto out;
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

    fprintf(out, "vcz_variant_encoder: %p\n", (const void *) self);
    fprintf(out, "\tnum_samples: %d\n", (int) self->num_samples);
    fprintf(out, "\tnum_variants: %d\n", (int) self->num_variants);
    vcz_field_print_state(&self->filter_id, out);
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

int
vcz_variant_encoder_add_info_field(vcz_variant_encoder_t *self, const char *name,
    int type, size_t item_size, size_t num_columns, const void *data)
{
    int ret = 0;
    vcz_field_t *tmp, *field;

    if (self->num_info_fields == self->max_info_fields) {
        self->max_info_fields += self->field_array_size_increment;
        /* self->info_fields is initially NULL */
        tmp = realloc(
            self->info_fields, self->max_info_fields * sizeof(*self->info_fields));
        if (tmp == NULL) {
            ret = VCZ_ERR_NO_MEMORY; // GCOVR_EXCL_LINE
            goto out;                // GCOVR_EXCL_LINE
        }
        self->info_fields = tmp;
    }
    field = self->info_fields + self->num_info_fields;
    ret = vcz_field_init(field, name, type, item_size, num_columns, data);
    if (ret != 0) {
        goto out;
    }
    self->num_info_fields++;
out:
    return ret;
}

int
vcz_variant_encoder_add_format_field(vcz_variant_encoder_t *self, const char *name,
    int type, size_t item_size, size_t num_columns, const void *data)
{
    int ret = 0;
    vcz_field_t *tmp, *field;

    if (type == VCZ_TYPE_BOOL) {
        ret = VCZ_ERR_FIELD_UNSUPPORTED_TYPE;
        goto out;
    }

    if (self->num_format_fields == self->max_format_fields) {
        self->max_format_fields += self->field_array_size_increment;
        /* self->format_fields is initially NULL */
        tmp = realloc(
            self->format_fields, self->max_format_fields * sizeof(*self->format_fields));
        if (tmp == NULL) {
            ret = VCZ_ERR_NO_MEMORY; // GCOVR_EXCL_LINE
            goto out;                // GCOVR_EXCL_LINE
        }
        self->format_fields = tmp;
    }
    field = self->format_fields + self->num_format_fields;

    ret = vcz_field_init(field, name, type, item_size, num_columns, data);
    if (ret != 0) {
        goto out;
    }
    self->num_format_fields++;
out:
    return ret;
}

int
vcz_variant_encoder_add_gt_field(vcz_variant_encoder_t *self, size_t item_size,
    size_t num_columns, const void *data, const int8_t *phased_data)
{
    self->gt_phased_data = phased_data;
    return vcz_field_init(&self->gt, "GT", VCZ_TYPE_INT, item_size, num_columns, data);
}

#define CHROM_FIELD_INDEX  0
#define POS_FIELD_INDEX    1
#define ID_FIELD_INDEX     2
#define REF_FIELD_INDEX    3
#define ALT_FIELD_INDEX    4
#define QUAL_FIELD_INDEX   5
#define FILTER_FIELD_INDEX 6

int
vcz_variant_encoder_add_chrom_field(
    vcz_variant_encoder_t *self, size_t item_size, const char *data)
{
    return vcz_field_init(self->fixed_fields + CHROM_FIELD_INDEX, "CHROM",
        VCZ_TYPE_STRING, item_size, 1, data);
}

int
vcz_variant_encoder_add_pos_field(vcz_variant_encoder_t *self, const int32_t *data)
{
    return vcz_field_init(
        self->fixed_fields + POS_FIELD_INDEX, "POS", VCZ_TYPE_INT, 4, 1, data);
}

int
vcz_variant_encoder_add_qual_field(vcz_variant_encoder_t *self, const float *data)
{
    return vcz_field_init(
        self->fixed_fields + QUAL_FIELD_INDEX, "QUAL", VCZ_TYPE_FLOAT, 4, 1, data);
}

int
vcz_variant_encoder_add_ref_field(
    vcz_variant_encoder_t *self, size_t item_size, const char *data)
{
    return vcz_field_init(self->fixed_fields + REF_FIELD_INDEX, "REF", VCZ_TYPE_STRING,
        item_size, 1, data);
}

int
vcz_variant_encoder_add_id_field(
    vcz_variant_encoder_t *self, size_t item_size, size_t num_columns, const char *data)
{
    return vcz_field_init(self->fixed_fields + ID_FIELD_INDEX, "ID", VCZ_TYPE_STRING,
        item_size, num_columns, data);
}

int
vcz_variant_encoder_add_alt_field(
    vcz_variant_encoder_t *self, size_t item_size, size_t num_columns, const char *data)
{
    return vcz_field_init(self->fixed_fields + ALT_FIELD_INDEX, "ALT", VCZ_TYPE_STRING,
        item_size, num_columns, data);
}

int
vcz_variant_encoder_add_filter_field(vcz_variant_encoder_t *self, size_t id_item_size,
    size_t id_num_columns, const char *id_data, const int8_t *filter_data)
{
    self->filter_data = filter_data;
    return vcz_field_init(&self->filter_id, "IDS/FILTERS", VCZ_TYPE_STRING, id_item_size,
        id_num_columns, id_data);
}

int
vcz_variant_encoder_init(
    vcz_variant_encoder_t *self, size_t num_variants, size_t num_samples)
{
    memset(self, 0, sizeof(*self));
    self->num_samples = num_samples;
    self->num_variants = num_variants;
    self->field_array_size_increment = 64; // arbitrary
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

/* A1 = ALT (allele index 1), A2 = REF (allele index 0): plink 2's
 * --vcf X --make-bed convention. Diploid sums map to:
 *   0 = REF/REF -> HOM_A2; 1 = REF/ALT or ALT/REF -> HET; 2 = ALT/ALT -> HOM_A1
 * Anything else (negative sentinels, multi-allelic indices > 1) -> MISSING. */
static const uint8_t plink_diploid_codes[3]
    = { VCZ_PLINK_HOM_A2, VCZ_PLINK_HET, VCZ_PLINK_HOM_A1 };

static inline uint8_t
encode_diploid_fixed(int8_t a, int8_t b)
{
    if (b == -2) {
        /* Haploid call: plink encodes as homozygous for the called allele. */
        if (a == 0) {
            return VCZ_PLINK_HOM_A2;
        }
        if (a == 1) {
            return VCZ_PLINK_HOM_A1;
        }
        return VCZ_PLINK_MISSING;
    }
    /* Cast to unsigned to fold negatives (very large unsigned) and any
     * value >= 2 into the same out-of-range branch. */
    if ((unsigned) a > 1 || (unsigned) b > 1) {
        return VCZ_PLINK_MISSING;
    }
    return plink_diploid_codes[a + b];
}

int
vcz_encode_plink(
    size_t num_variants, size_t num_samples, const int8_t *genotypes, char *buf)
{
    const size_t full_bytes = num_samples / 4;
    const size_t tail = num_samples % 4;
    uint8_t c0, c1, c2, c3, byte;
    const int8_t *gt;
    char *out;
    size_t i, j, b;

    for (j = 0; j < num_variants; j++) {
        gt = genotypes + j * num_samples * 2;
        out = buf + j * ((num_samples + 3) / 4);

        for (b = 0; b < full_bytes; b++) {
            c0 = encode_diploid_fixed(gt[0], gt[1]);
            c1 = encode_diploid_fixed(gt[2], gt[3]);
            c2 = encode_diploid_fixed(gt[4], gt[5]);
            c3 = encode_diploid_fixed(gt[6], gt[7]);
            out[b] = (char) (c0 | (c1 << 2) | (c2 << 4) | (c3 << 6));
            gt += 8;
        }

        if (tail > 0) {
            byte = 0;
            for (i = 0; i < tail; i++) {
                byte |= (uint8_t) (encode_diploid_fixed(gt[0], gt[1]) << (i * 2));
                gt += 2;
            }
            out[full_bytes] = (char) byte;
        }
    }
    return 0;
}

size_t
vcz_bgen_geno_block_row_max_size(size_t num_samples)
{
    /* Worst-case (all diploid): 8 header + S ploidy + 2 flags + 2*S probs. */
    return 10 + 3 * num_samples;
}

/* BGEN Layout 2 / 8-bit / biallelic genotype block builder with
 * per-sample ploidy. See vcztools/bgen.py and the BGEN spec for the
 * byte layout. Each row is variable-length: header (8B) + ploidy bytes
 * (one per sample) + phased flag (1B) + bits-per-prob (1B) + per-sample
 * probability bytes (K_s bytes for ploidy K_s; K_s in {1, 2}). Per-
 * variant header reports the variant's actual Pmin/Pmax.
 *
 * Per-sample interpretation of (a, b) = (gt[2s], gt[2s+1]):
 *   a in {0, 1}, b in {0, 1}      -> diploid call. 0x02. 2 prob bytes.
 *   a in {0, 1}, b == -2          -> haploid call. 0x01. 1 prob byte.
 *   a == -1, b == -1              -> missing diploid. 0x82. 2 zero bytes.
 *   a == -1, b == -2              -> missing haploid. 0x81. 1 zero byte.
 *   half-missing diploid           -> 0x82. 2 zero bytes.
 *   a == -2 (any b)               -> zero-ploidy. Returns
 *                                    VCZ_ERR_BGEN_INVALID_PLOIDY.
 *   a or b outside {-2, -1, 0, 1} -> Returns VCZ_ERR_BGEN_INVALID_ALLELE.
 *
 * buf must be at least num_variants * row_stride bytes; row_stride is
 * typically vcz_bgen_geno_block_row_max_size(num_samples). out_lens[v]
 * receives the actual byte count written for variant v. */
int
vcz_encode_bgen_geno_blocks(size_t num_variants, size_t num_samples,
    const int8_t *genotypes, const uint8_t *phased, uint8_t *buf, size_t row_stride,
    uint32_t *out_lens)
{
    int ret = 0;
    uint8_t *row;
    uint8_t *ploidy_out;
    uint8_t *prob_out;
    const int8_t *gt;
    int8_t a, b;
    uint8_t variant_phased, ploidy_byte, pmin, pmax;
    size_t v, s, prob_offset;

    for (v = 0; v < num_variants; v++) {
        row = buf + v * row_stride;
        gt = genotypes + v * num_samples * 2;
        ploidy_out = row + 8;
        prob_out = row + 8 + num_samples + 2;
        variant_phased = phased[v] ? 1 : 0;
        pmin = 2;
        pmax = 1;
        prob_offset = 0;

        for (s = 0; s < num_samples; s++) {
            a = gt[2 * s];
            b = gt[2 * s + 1];

            /* The BGEN encoder is biallelic: only {-2, -1, 0, 1} are
             * accepted; any other value is a data-quality error. */
            if (a < -2 || a > 1 || b < -2 || b > 1) {
                ret = VCZ_ERR_BGEN_INVALID_ALLELE;
                goto out;
            }
            if (a == VCZ_INT_FILL) {
                ret = VCZ_ERR_BGEN_INVALID_PLOIDY;
                goto out;
            }
            if (b == VCZ_INT_FILL) {
                /* Haploid call: ploidy=1. */
                if (a == VCZ_INT_MISSING) {
                    ploidy_byte = VCZ_BGEN_PLOIDY_MISSING_HAPLOID;
                    prob_out[prob_offset] = 0x00;
                } else {
                    ploidy_byte = VCZ_BGEN_PLOIDY_HAPLOID;
                    prob_out[prob_offset] = (a == 0) ? 0xFF : 0x00;
                }
                prob_offset += 1;
                if (pmin > 1) {
                    pmin = 1;
                }
            } else if (a < 0 || b < 0) {
                ploidy_byte = VCZ_BGEN_PLOIDY_MISSING_DIPLOID;
                prob_out[prob_offset] = 0x00;
                prob_out[prob_offset + 1] = 0x00;
                prob_offset += 2;
                pmax = 2;
            } else {
                ploidy_byte = VCZ_BGEN_PLOIDY_DIPLOID;
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
            ploidy_out[s] = ploidy_byte;
        }

        if (num_samples == 0) {
            /* Pmin/Pmax are undefined without samples; spec says report 2/2
             * (the diploid default). Matches the previous diploid-only
             * behaviour for empty stores. */
            pmin = 2;
            pmax = 2;
        }

        /* 8-byte header: N (uint32 LE), K (uint16 LE), Pmin, Pmax. */
        row[0] = (uint8_t) (num_samples & 0xFF);
        row[1] = (uint8_t) ((num_samples >> 8) & 0xFF);
        row[2] = (uint8_t) ((num_samples >> 16) & 0xFF);
        row[3] = (uint8_t) ((num_samples >> 24) & 0xFF);
        row[4] = 2; /* K (n_alleles) low byte */
        row[5] = 0; /* K high byte */
        row[6] = pmin;
        row[7] = pmax;
        row[8 + num_samples] = variant_phased;
        row[8 + num_samples + 1] = VCZ_BGEN_BITS_PER_PROB;

        out_lens[v] = (uint32_t) (8 + num_samples + 2 + prob_offset);
    }
out:
    return ret;
}

/* Adler-32 over `buf` (RFC 1950). NMAX=5552 is the largest run that
 * keeps s1, s2 within 32 bits before the mod 65521 fold. */
static uint32_t
adler32_update(uint32_t adler, const uint8_t *buf, size_t len)
{
    static const uint32_t MOD = 65521;
    static const size_t NMAX = 5552;
    uint32_t s1 = adler & 0xFFFF;
    uint32_t s2 = (adler >> 16) & 0xFFFF;
    size_t n;
    size_t i;

    while (len > 0) {
        n = len < NMAX ? len : NMAX;
        len -= n;
        for (i = 0; i < n; i++) {
            s1 += buf[i];
            s2 += s1;
        }
        s1 %= MOD;
        s2 %= MOD;
        buf += n;
    }
    return (s2 << 16) | s1;
}

/* Wire size of zlib.compress(_, 0) on a payload of D bytes:
 *   2-byte zlib header
 * + ceil(D / 65535) stored DEFLATE blocks (1 even when D == 0), each
 *   5 bytes of framing
 * + D bytes of payload
 * + 4-byte big-endian adler32
 */
static size_t
bgen_zlib_stored_size(size_t D)
{
    size_t num_blocks;
    if (D == 0) {
        num_blocks = 1;
    } else {
        num_blocks = (D + 65534) / 65535;
    }
    return 2 + 5 * num_blocks + D + 4;
}

/* Emit a zlib stream containing `D` bytes from `in` as stored DEFLATE
 * blocks. Returns the number of bytes written to `out` (equal to
 * bgen_zlib_stored_size(D)). */
static size_t
bgen_emit_zlib_stored(uint8_t *out, const uint8_t *in, size_t D)
{
    uint8_t *p = out;
    const uint8_t *payload = in;
    size_t remaining = D;
    uint32_t adler;
    uint16_t block_len;
    uint16_t nlen;

    /* zlib header: CMF=0x78 (deflate, 32K window), FLG=0x01 (no dict,
     * FLEVEL=0, FCHECK chosen so (CMF*256+FLG) % 31 == 0). Matches
     * Python's zlib.compress(_, 0). */
    *p++ = 0x78;
    *p++ = 0x01;

    if (D == 0) {
        /* Single empty stored block with BFINAL=1. */
        *p++ = 0x01;
        *p++ = 0x00;
        *p++ = 0x00;
        *p++ = 0xFF;
        *p++ = 0xFF;
    } else {
        while (remaining > 0) {
            if (remaining > 65535) {
                block_len = 65535;
                *p++ = 0x00; /* BFINAL=0, BTYPE=00 (stored) */
            } else {
                block_len = (uint16_t) remaining;
                *p++ = 0x01; /* BFINAL=1, BTYPE=00 */
            }
            nlen = (uint16_t) ~block_len;
            *p++ = (uint8_t) (block_len & 0xFF);
            *p++ = (uint8_t) ((block_len >> 8) & 0xFF);
            *p++ = (uint8_t) (nlen & 0xFF);
            *p++ = (uint8_t) ((nlen >> 8) & 0xFF);
            memcpy(p, in, block_len);
            p += block_len;
            in += block_len;
            remaining -= block_len;
        }
    }

    /* Big-endian adler32 over the uncompressed payload. */
    adler = adler32_update(1, payload, D);
    *p++ = (uint8_t) ((adler >> 24) & 0xFF);
    *p++ = (uint8_t) ((adler >> 16) & 0xFF);
    *p++ = (uint8_t) ((adler >> 8) & 0xFF);
    *p++ = (uint8_t) (adler & 0xFF);

    return (size_t) (p - out);
}

/* Fixed-size BGEN variant block encoder: writes
 *   num_variants * bytes_per_variant
 * bytes into `out_buf`, where bytes_per_variant = 28 + varid_max +
 * rsid_max + chrom_max + 2*allele_max + bgen_zlib_stored_size(geno_size).
 *
 * Every string field arrives pre-padded to its `*_max` width as a
 * `(num_variants, *_max) uint8` row-major block. `position` is one
 * int32 per variant. `geno_blocks` rows are `geno_row_stride` bytes
 * wide; only the first `geno_size` bytes are read (the trailing tail
 * is the worst-case diploid stride). Mirrors the byte-at-a-time
 * little-endian writes elsewhere in this module. */
int
vcz_encode_bgen_chunk_slice_level0(size_t num_variants, const uint8_t *varid,
    size_t varid_max, const uint8_t *rsid, size_t rsid_max, const uint8_t *chrom,
    size_t chrom_max, const uint8_t *allele1, const uint8_t *allele2, size_t allele_max,
    const int32_t *position, const uint8_t *geno_blocks, size_t geno_row_stride,
    size_t geno_size, uint8_t *out_buf)
{
    size_t v;
    size_t payload_size;
    size_t bpv;
    uint8_t *out;
    uint32_t pos;
    uint32_t C;

    payload_size = bgen_zlib_stored_size(geno_size);
    bpv = 28 + varid_max + rsid_max + chrom_max + 2 * allele_max + payload_size;

    for (v = 0; v < num_variants; v++) {
        out = out_buf + v * bpv;

        /* varid: uint16 LE length + bytes */
        *out++ = (uint8_t) (varid_max & 0xFF);
        *out++ = (uint8_t) ((varid_max >> 8) & 0xFF);
        memcpy(out, varid + v * varid_max, varid_max);
        out += varid_max;

        /* rsid */
        *out++ = (uint8_t) (rsid_max & 0xFF);
        *out++ = (uint8_t) ((rsid_max >> 8) & 0xFF);
        memcpy(out, rsid + v * rsid_max, rsid_max);
        out += rsid_max;

        /* chrom */
        *out++ = (uint8_t) (chrom_max & 0xFF);
        *out++ = (uint8_t) ((chrom_max >> 8) & 0xFF);
        memcpy(out, chrom + v * chrom_max, chrom_max);
        out += chrom_max;

        /* position: uint32 LE */
        pos = (uint32_t) position[v];
        *out++ = (uint8_t) (pos & 0xFF);
        *out++ = (uint8_t) ((pos >> 8) & 0xFF);
        *out++ = (uint8_t) ((pos >> 16) & 0xFF);
        *out++ = (uint8_t) ((pos >> 24) & 0xFF);

        /* K = 2 alleles */
        *out++ = 0x02;
        *out++ = 0x00;

        /* allele1: uint32 LE length + bytes */
        *out++ = (uint8_t) (allele_max & 0xFF);
        *out++ = (uint8_t) ((allele_max >> 8) & 0xFF);
        *out++ = (uint8_t) ((allele_max >> 16) & 0xFF);
        *out++ = (uint8_t) ((allele_max >> 24) & 0xFF);
        memcpy(out, allele1 + v * allele_max, allele_max);
        out += allele_max;

        /* allele2 */
        *out++ = (uint8_t) (allele_max & 0xFF);
        *out++ = (uint8_t) ((allele_max >> 8) & 0xFF);
        *out++ = (uint8_t) ((allele_max >> 16) & 0xFF);
        *out++ = (uint8_t) ((allele_max >> 24) & 0xFF);
        memcpy(out, allele2 + v * allele_max, allele_max);
        out += allele_max;

        /* C = 4 + compressed payload size, then D = uncompressed size. */
        C = (uint32_t) (4 + payload_size);
        *out++ = (uint8_t) (C & 0xFF);
        *out++ = (uint8_t) ((C >> 8) & 0xFF);
        *out++ = (uint8_t) ((C >> 16) & 0xFF);
        *out++ = (uint8_t) ((C >> 24) & 0xFF);
        *out++ = (uint8_t) (geno_size & 0xFF);
        *out++ = (uint8_t) ((geno_size >> 8) & 0xFF);
        *out++ = (uint8_t) ((geno_size >> 16) & 0xFF);
        *out++ = (uint8_t) ((geno_size >> 24) & 0xFF);

        /* Stored zlib payload. */
        bgen_emit_zlib_stored(out, geno_blocks + v * geno_row_stride, geno_size);
    }
    return 0;
}
