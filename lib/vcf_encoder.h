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

#define VCZ_INT_MISSING              -1
#define VCZ_INT_FILL                 -2
#define VCZ_STRING_MISSING           '.'
#define VCZ_STRING_FILL              '\0'
#define VCZ_FLOAT32_MISSING_AS_INT32 0x7F800001
#define VCZ_FLOAT32_FILL_AS_INT32    0x7F800002

#define VCZ_NUM_FIXED_FIELDS 6

#define VCZ_TYPE_INT    1
#define VCZ_TYPE_FLOAT  2
#define VCZ_TYPE_STRING 3
#define VCZ_TYPE_BOOL   4

// arbitrary - we can increase if needs be
#define VCZ_MAX_FIELD_NAME_LEN 255
#define VCZ_INT32_BUF_SIZE     12 // 10 digits, leading '-' and terminating NULL
// Safe limit, no point in trying to make it too tight as it's easy to represent
// certain very large numbers of floating point.
#define VCZ_FLOAT32_BUF_SIZE 256

#define VCZ_ERR_NO_MEMORY             (-100)
#define VCZ_ERR_BUFFER_OVERFLOW       (-101)
#define VCZ_ERR_VARIANT_OUT_OF_BOUNDS (-102)
#define VCZ_ERR_INVALID_GENOTYPE      (-103)

/* Built-in-limitations */
#define VCZ_ERR_FIELD_NAME_TOO_LONG           (-201)
#define VCZ_ERR_FIELD_UNSUPPORTED_TYPE        (-202)
#define VCZ_ERR_FIELD_UNSUPPORTED_ITEM_SIZE   (-203)
#define VCZ_ERR_FIELD_UNSUPPORTED_NUM_COLUMNS (-204)

/* BGEN encoder errors */
#define VCZ_ERR_BGEN_INVALID_PLOIDY         (-301)
#define VCZ_ERR_BGEN_INVALID_ALLELE         (-302)
#define VCZ_ERR_BGEN_MIXED_PLOIDY           (-303)
#define VCZ_ERR_BGEN_STRING_LENGTH_MISMATCH (-304)

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

#define VCZ_PLINK_HOM_A1  0x0 /* 00 */
#define VCZ_PLINK_HOM_A2  0x3 /* 11 */
#define VCZ_PLINK_HET     0x2 /* 10 */
#define VCZ_PLINK_MISSING 0x1 /* 01 */
int vcz_encode_plink(
    size_t num_variants, size_t num_samples, const int8_t *genotypes, char *buf);

/* Per-variant AC (allele count) and AN (total called alleles) from a
 * 3-D genotype buffer of shape (num_variants, num_samples, ploidy).
 * num_alleles[j] is the total allele count (REF + ALT) at variant j;
 * the accepted genotype range is -2 <= v < num_alleles[j]. Slots
 * holding VCZ_INT_MISSING (-1) or VCZ_INT_FILL (-2) are excluded from
 * both AC and AN. Values v with 0 < v < num_alleles[j] increment
 * ac_out[j*max_num_alt + (v-1)]. Any value outside the accepted range
 * causes VCZ_ERR_INVALID_GENOTYPE to be returned. AC cells beyond
 * num_alleles[j]-1 within each row are filled with VCZ_INT_FILL.
 * max_num_alt is the trailing dimension of ac_out; pass 0 to compute
 * AN only (ac_out is then untouched). */
int vcz_compute_ac_an(size_t num_variants, size_t num_samples, size_t ploidy,
    size_t max_num_alt, const int32_t *num_alleles, const int8_t *genotypes,
    int32_t *ac_out, int32_t *an_out);

#define VCZ_BGEN_PLOIDY_HAPLOID         0x01
#define VCZ_BGEN_PLOIDY_DIPLOID         0x02
#define VCZ_BGEN_PLOIDY_MISSING_HAPLOID 0x81
#define VCZ_BGEN_PLOIDY_MISSING_DIPLOID 0x82
#define VCZ_BGEN_BITS_PER_PROB          8
int vcz_encode_bgen_geno_blocks(size_t num_variants, size_t num_samples,
    const int8_t *genotypes, const uint8_t *phased, uint8_t *buf, size_t row_stride,
    uint32_t *out_lens);
size_t vcz_bgen_geno_block_size(size_t num_samples, size_t uniform_ploidy);
size_t vcz_bgen_variant_block_size(
    size_t num_samples, size_t uniform_ploidy, size_t total_string_length);
int vcz_encode_bgen_chunk_slice_level0(size_t num_variants, size_t num_samples,
    size_t uniform_ploidy, size_t total_string_length, const uint8_t *varid,
    size_t varid_stride, const uint8_t *rsid, size_t rsid_stride, const uint8_t *chrom,
    size_t chrom_stride, const uint8_t *allele1, size_t allele1_stride,
    const uint8_t *allele2, size_t allele2_stride, const int32_t *position,
    const int8_t *genotypes, const uint8_t *phased, uint8_t *out_buf);

/* Drop-in replacements for the zlib symbols referenced by the BGEN encoder.
 * Same name (with vcz_ prefix), same parameter order and semantics as zlib,
 * stdint types instead of zlib typedefs. Specialised to level-0/stored
 * DEFLATE so we don't have to link libz in production; the C test suite
 * cross-checks them against libz. */
#define VCZ_Z_OK           0
#define VCZ_Z_STREAM_ERROR (-401)
#define VCZ_Z_BUF_ERROR    (-402)
uint32_t vcz_adler32(uint32_t adler, const uint8_t *buf, size_t len);
size_t vcz_compress_bound(size_t source_len);
int vcz_compress2(uint8_t *dest, size_t *dest_len, const uint8_t *source,
    size_t source_len, int level);
