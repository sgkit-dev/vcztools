#include "vcztools.h"

int
vcz_compute_ac_an(size_t num_variants, size_t num_samples, size_t ploidy,
    size_t max_num_alt, const int32_t *num_alleles, const int8_t *genotypes,
    int32_t *ac_out, int32_t *an_out)
{
    int ret = 0;
    size_t j, k, slots_per_variant, num_alt_j;
    int32_t an, na_j;
    int32_t *ac_row;
    const int8_t *gt_row;
    int8_t v;

    slots_per_variant = num_samples * ploidy;
    for (j = 0; j < num_variants; j++) {
        na_j = num_alleles[j];
        if (na_j < 1) {
            ret = VCZ_ERR_INVALID_NUM_ALLELES;
            goto out;
        }
        num_alt_j = (size_t) na_j - 1;
        gt_row = genotypes + j * slots_per_variant;
        ac_row = ac_out + j * max_num_alt;
        for (k = 0; k < num_alt_j && k < max_num_alt; k++) {
            ac_row[k] = 0;
        }
        for (k = num_alt_j; k < max_num_alt; k++) {
            ac_row[k] = VCZ_INT_FILL;
        }
        an = 0;
        for (k = 0; k < slots_per_variant; k++) {
            v = gt_row[k];
            if (v == VCZ_INT_FILL || v == VCZ_INT_MISSING) {
                continue;
            }
            if (v < 0 || v >= na_j) {
                ret = VCZ_ERR_INVALID_GENOTYPE;
                goto out;
            }
            an++;
            if (v > 0 && max_num_alt > 0) {
                ac_row[v - 1]++;
            }
        }
        an_out[j] = an;
    }
out:
    return ret;
}
