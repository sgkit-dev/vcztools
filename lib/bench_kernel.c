/* Standalone microbenchmark driver for vcz_encode_bgen_chunk_slice_level0.
 *
 * Builds representative wide-bench input (1000 variants × 100k samples,
 * uniform diploid, mixed genotype values), runs the kernel `nreps` times,
 * reports wall time. Designed for perf record / perf stat: no Python
 * overhead, no benchmark harness, just the C kernel hit in a loop. */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "vcf_encoder.h"

#define NUM_VARIANTS   1000
#define NUM_SAMPLES    100000
#define UNIFORM_PLOIDY 2
#define VARID_MAX      64
#define RSID_MAX       64
#define CHROM_MAX      4
#define ALLELE_MAX     1

static double
now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec / 1e9;
}

static void
populate(uint8_t *varid, uint8_t *rsid, uint8_t *chrom, uint8_t *allele1,
    uint8_t *allele2, int32_t *position, int8_t *genotypes, uint8_t *phased)
{
    /* String fields are NUL-padded fixed-width. Fill with a representative
     * pattern so memcpy in the kernel has real bytes to move. */
    size_t v, s;
    for (v = 0; v < NUM_VARIANTS; v++) {
        char buf[VARID_MAX + 1];
        int n = snprintf(buf, sizeof(buf), "rs%07zu", v);
        memset(varid + v * VARID_MAX, 0, VARID_MAX);
        memcpy(varid + v * VARID_MAX, buf, (size_t) n);
        memset(rsid + v * RSID_MAX, 0, RSID_MAX);
        memcpy(rsid + v * RSID_MAX, buf, (size_t) n);
        memcpy(chrom + v * CHROM_MAX, "chr1", 4);
        allele1[v * ALLELE_MAX] = 'A';
        allele2[v * ALLELE_MAX] = 'T';
        position[v] = (int32_t) (1000 + v);
        phased[v] = 0;
    }

    /* Genotypes: mix of hom-ref, het, hom-alt, missing, cycling deterministically
     * per (v, s) so each variant has a different distribution. */
    static const int8_t pattern[8][2] = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 },
        { 0, 0 }, { -1, -1 }, { 0, 1 }, { 1, 0 } };
    for (v = 0; v < NUM_VARIANTS; v++) {
        for (s = 0; s < NUM_SAMPLES; s++) {
            size_t idx = (v + s) & 7;
            genotypes[(v * NUM_SAMPLES + s) * 2 + 0] = pattern[idx][0];
            genotypes[(v * NUM_SAMPLES + s) * 2 + 1] = pattern[idx][1];
        }
    }
}

int
main(int argc, char **argv)
{
    int nreps = 5;
    if (argc >= 2) {
        nreps = atoi(argv[1]);
    }

    size_t geno_size = 10 + (UNIFORM_PLOIDY + 1) * NUM_SAMPLES;
    size_t payload_size = vcz_compress_bound(geno_size);
    size_t bpv = 28 + VARID_MAX + RSID_MAX + CHROM_MAX + 2 * ALLELE_MAX + payload_size;
    size_t out_size = NUM_VARIANTS * bpv;

    uint8_t *varid = malloc(NUM_VARIANTS * VARID_MAX);
    uint8_t *rsid = malloc(NUM_VARIANTS * RSID_MAX);
    uint8_t *chrom = malloc(NUM_VARIANTS * CHROM_MAX);
    uint8_t *allele1 = malloc(NUM_VARIANTS * ALLELE_MAX);
    uint8_t *allele2 = malloc(NUM_VARIANTS * ALLELE_MAX);
    int32_t *position = malloc(NUM_VARIANTS * sizeof(int32_t));
    int8_t *genotypes = malloc((size_t) NUM_VARIANTS * NUM_SAMPLES * 2);
    uint8_t *phased = malloc(NUM_VARIANTS);
    uint8_t *out_buf = malloc(out_size);

    if (!varid || !rsid || !chrom || !allele1 || !allele2 || !position || !genotypes
        || !phased || !out_buf) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    populate(varid, rsid, chrom, allele1, allele2, position, genotypes, phased);

    /* Warmup */
    int rc = vcz_encode_bgen_chunk_slice_level0(NUM_VARIANTS, NUM_SAMPLES,
        UNIFORM_PLOIDY, varid, VARID_MAX, rsid, RSID_MAX, chrom, CHROM_MAX, allele1,
        allele2, ALLELE_MAX, position, genotypes, phased, out_buf);
    if (rc != 0) {
        fprintf(stderr, "kernel returned %d\n", rc);
        return 1;
    }

    fprintf(stderr,
        "config: %d variants x %d samples, geno_size=%zu, bpv=%zu, "
        "out_size=%zu MiB, nreps=%d\n",
        NUM_VARIANTS, NUM_SAMPLES, geno_size, bpv, out_size / (1024 * 1024), nreps);

    double t0 = now_seconds();
    for (int i = 0; i < nreps; i++) {
        rc = vcz_encode_bgen_chunk_slice_level0(NUM_VARIANTS, NUM_SAMPLES,
            UNIFORM_PLOIDY, varid, VARID_MAX, rsid, RSID_MAX, chrom, CHROM_MAX, allele1,
            allele2, ALLELE_MAX, position, genotypes, phased, out_buf);
        if (rc != 0) {
            fprintf(stderr, "rep %d: kernel returned %d\n", i, rc);
            return 1;
        }
    }
    double elapsed = now_seconds() - t0;
    double per_call = elapsed / nreps;
    double mib_out = (double) out_size / (1024.0 * 1024.0);
    fprintf(stderr, "elapsed: %.3f s (%d reps); %.1f ms/call; %.0f MiB/s out\n", elapsed,
        nreps, per_call * 1000.0, mib_out / per_call);

    free(varid);
    free(rsid);
    free(chrom);
    free(allele1);
    free(allele2);
    free(position);
    free(genotypes);
    free(phased);
    free(out_buf);
    return 0;
}
