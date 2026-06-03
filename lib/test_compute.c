#include <CUnit/Basic.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vcztools.h>

static void
test_compute_ac_an_ploidy2(void)
{
    /* Diploid storage with mixed ploidy realised via VCZ_INT_FILL in the
     * trailing slot of haploid samples. The per-variant num_alleles
     * array gives each row a different REF+ALT count; AC cells beyond
     * num_alleles[j]-1 must come back as VCZ_INT_FILL. */
    const size_t num_variants = 5;
    const size_t num_samples = 3;
    const size_t ploidy = 2;
    const size_t max_num_alt = 3;
    /* clang-format off */
    int32_t num_alleles[] = { 2, 3, 4, 2, 3 };
    int8_t genotypes[] = {
        /* var 0 (na=2): 0/0 0/1 1/1               -> AN=6, AC=[3, F, F] */
         0,  0,  0,  1,  1,  1,
        /* var 1 (na=3): 0/0 0/2 1/2               -> AN=6, AC=[1, 2, F] */
         0,  0,  0,  2,  1,  2,
        /* var 2 (na=4): 0|1 1/2 3                 -> AN=5, AC=[2, 1, 1]
         *   sample 2 is haploid with allele 3, encoded as (3, -2) */
         0,  1,  1,  2,  3, -2,
        /* var 3 (na=2): ./.  ./fill 1              -> AN=1, AC=[1, F, F]
         *   sample 0: missing/missing; sample 1: missing/fill; sample 2:
         *   haploid (1, -2). Realistic mixed-ploidy missing pattern. */
        -1, -1, -1, -2,  1, -2,
        /* var 4 (na=3): 0|2 ./fill 1               -> AN=3, AC=[1, 1, F]
         *   sample 1 is missing-haploid encoded (-1, -2). */
         0,  2, -1, -2,  1, -2,
    };
    int32_t expected_ac[] = {
        3, VCZ_INT_FILL, VCZ_INT_FILL,
        1, 2,            VCZ_INT_FILL,
        2, 1,            1,
        1, VCZ_INT_FILL, VCZ_INT_FILL,
        1, 1,            VCZ_INT_FILL,
    };
    int32_t expected_an[] = { 6, 6, 5, 1, 3 };
    /* clang-format on */
    int32_t ac_buf[15];
    int32_t an_buf[5];
    size_t i;
    int ret;

    /* Pre-fill the output buffers with sentinel values to verify the
     * kernel writes every AC cell (counts and FILL padding alike). */
    for (i = 0; i < 15; i++) {
        ac_buf[i] = 999;
    }
    for (i = 0; i < 5; i++) {
        an_buf[i] = 999;
    }

    ret = vcz_compute_ac_an(num_variants, num_samples, ploidy, max_num_alt, num_alleles,
        genotypes, ac_buf, an_buf);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    for (i = 0; i < 15; i++) {
        CU_ASSERT_EQUAL_FATAL(ac_buf[i], expected_ac[i]);
    }
    for (i = 0; i < 5; i++) {
        CU_ASSERT_EQUAL_FATAL(an_buf[i], expected_an[i]);
    }
}

static void
test_compute_ac_an_an_only(void)
{
    /* AN-only mode: max_num_alt=0, ac_out is untouched. Same realistic
     * mixed-ploidy genotypes as test_compute_ac_an_ploidy2. */
    const size_t num_variants = 5;
    const size_t num_samples = 3;
    const size_t ploidy = 2;
    /* clang-format off */
    int32_t num_alleles[] = { 2, 3, 4, 2, 3 };
    int8_t genotypes[] = {
         0,  0,  0,  1,  1,  1,
         0,  0,  0,  2,  1,  2,
         0,  1,  1,  2,  3, -2,
        -1, -1, -1, -2,  1, -2,
         0,  2, -1, -2,  1, -2,
    };
    /* clang-format on */
    int32_t expected_an[] = { 6, 6, 5, 1, 3 };
    int32_t an_buf[5];
    int32_t *ac_buf = NULL;
    size_t i;
    int ret;

    ret = vcz_compute_ac_an(
        num_variants, num_samples, ploidy, 0, num_alleles, genotypes, ac_buf, an_buf);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    for (i = 0; i < 5; i++) {
        CU_ASSERT_EQUAL_FATAL(an_buf[i], expected_an[i]);
    }
}

static void
test_compute_ac_an_ploidy1(void)
{
    /* Haploid genotypes only. The kernel sees ploidy=1 with no FILL
     * sentinels; only VCZ_INT_MISSING marks an uncalled allele. */
    const size_t num_variants = 3;
    const size_t num_samples = 4;
    const size_t ploidy = 1;
    const size_t max_num_alt = 2;
    /* clang-format off */
    int32_t num_alleles[] = { 2, 3, 2 };
    int8_t genotypes[] = {
        /* var 0 (na=2): 0 1 0 1                  -> AN=4, AC=[2, F] */
         0,  1,  0,  1,
        /* var 1 (na=3): 2 .  0 1                 -> AN=3, AC=[1, 1] */
         2, -1,  0,  1,
        /* var 2 (na=2): .  .  .  .               -> AN=0, AC=[0, F] */
        -1, -1, -1, -1,
    };
    int32_t expected_ac[] = {
        2, VCZ_INT_FILL,
        1, 1,
        0, VCZ_INT_FILL,
    };
    int32_t expected_an[] = { 4, 3, 0 };
    /* clang-format on */
    int32_t ac_buf[6];
    int32_t an_buf[3];
    size_t i;
    int ret;

    ret = vcz_compute_ac_an(num_variants, num_samples, ploidy, max_num_alt, num_alleles,
        genotypes, ac_buf, an_buf);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    for (i = 0; i < 6; i++) {
        CU_ASSERT_EQUAL_FATAL(ac_buf[i], expected_ac[i]);
    }
    for (i = 0; i < 3; i++) {
        CU_ASSERT_EQUAL_FATAL(an_buf[i], expected_an[i]);
    }
}

static void
test_compute_ac_an_ploidy3(void)
{
    /* Triploid genotypes, including a row that mixes the three alleles
     * (0, 1, 2) in a single call. */
    const size_t num_variants = 3;
    const size_t num_samples = 2;
    const size_t ploidy = 3;
    const size_t max_num_alt = 2;
    /* clang-format off */
    int32_t num_alleles[] = { 3, 3, 3 };
    int8_t genotypes[] = {
        /* var 0 (na=3): (0,1,2) (1,1,2)          -> AN=6, AC=[3, 2] */
         0,  1,  2,  1,  1,  2,
        /* var 1 (na=3): (0,0,0) (.,.,.)          -> AN=3, AC=[0, 0] */
         0,  0,  0, -1, -1, -1,
        /* var 2 (na=3): (2,2,.) (1,2,0)          -> AN=5, AC=[1, 3] */
         2,  2, -1,  1,  2,  0,
    };
    int32_t expected_ac[] = {
        3, 2,
        0, 0,
        1, 3,
    };
    int32_t expected_an[] = { 6, 3, 5 };
    /* clang-format on */
    int32_t ac_buf[6];
    int32_t an_buf[3];
    size_t i;
    int ret;

    ret = vcz_compute_ac_an(num_variants, num_samples, ploidy, max_num_alt, num_alleles,
        genotypes, ac_buf, an_buf);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    for (i = 0; i < 6; i++) {
        CU_ASSERT_EQUAL_FATAL(ac_buf[i], expected_ac[i]);
    }
    for (i = 0; i < 3; i++) {
        CU_ASSERT_EQUAL_FATAL(an_buf[i], expected_an[i]);
    }
}

static void
test_compute_ac_an_invalid_genotype(void)
{
    /* Every accepted genotype lies in [-2, num_alleles[j]); anything
     * else makes the kernel return VCZ_ERR_INVALID_GENOTYPE. */
    const size_t num_samples = 1;
    const size_t ploidy = 2;
    const size_t max_num_alt = 2;
    int32_t num_alleles_2[] = { 2 };
    int32_t num_alleles_1[] = { 1 };
    int8_t gt_below[] = { 0, -3 };
    int8_t gt_above[] = { 0, 2 };
    int8_t gt_ref_only[] = { 0, 1 };
    int32_t ac_buf[2];
    int32_t an_buf[1];
    int ret;

    /* Case 1: v == -3 is below the missing/fill range. */
    ret = vcz_compute_ac_an(
        1, num_samples, ploidy, max_num_alt, num_alleles_2, gt_below, ac_buf, an_buf);
    CU_ASSERT_EQUAL_FATAL(ret, VCZ_ERR_INVALID_GENOTYPE);

    /* Case 2: v == num_alleles[j] is one past the upper bound. */
    ret = vcz_compute_ac_an(
        1, num_samples, ploidy, max_num_alt, num_alleles_2, gt_above, ac_buf, an_buf);
    CU_ASSERT_EQUAL_FATAL(ret, VCZ_ERR_INVALID_GENOTYPE);

    /* Case 3: num_alleles=1 (REF only) rejects any non-zero ALT. */
    ret = vcz_compute_ac_an(
        1, num_samples, ploidy, max_num_alt, num_alleles_1, gt_ref_only, ac_buf, an_buf);
    CU_ASSERT_EQUAL_FATAL(ret, VCZ_ERR_INVALID_GENOTYPE);
}

static void
test_compute_ac_an_invalid_num_alleles(void)
{
    /* num_alleles[j] < 1 is rejected with VCZ_ERR_INVALID_NUM_ALLELES,
     * distinct from the out-of-range genotype error. */
    const size_t num_samples = 1;
    const size_t ploidy = 2;
    const size_t max_num_alt = 2;
    int32_t num_alleles_zero[] = { 0 };
    int32_t num_alleles_negative[] = { -1 };
    int8_t genotypes[] = { 0, 0 };
    int32_t ac_buf[2];
    int32_t an_buf[1];
    int ret;

    ret = vcz_compute_ac_an(1, num_samples, ploidy, max_num_alt, num_alleles_zero,
        genotypes, ac_buf, an_buf);
    CU_ASSERT_EQUAL_FATAL(ret, VCZ_ERR_INVALID_NUM_ALLELES);

    ret = vcz_compute_ac_an(1, num_samples, ploidy, max_num_alt, num_alleles_negative,
        genotypes, ac_buf, an_buf);
    CU_ASSERT_EQUAL_FATAL(ret, VCZ_ERR_INVALID_NUM_ALLELES);
}

static void
test_compute_ac_an_zero_variants(void)
{
    /* No variants -> kernel must not write anything; pre-filled
     * sentinels in unrelated buffers stay untouched. */
    int8_t genotypes[1] = { 0 };
    int32_t num_alleles[1] = { 2 };
    int32_t ac_buf[4] = { 11, 22, 33, 44 };
    int32_t an_buf[2] = { 55, 66 };
    int ret;

    ret = vcz_compute_ac_an(0, 2, 2, 2, num_alleles, genotypes, ac_buf, an_buf);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(ac_buf[0], 11);
    CU_ASSERT_EQUAL_FATAL(ac_buf[1], 22);
    CU_ASSERT_EQUAL_FATAL(ac_buf[2], 33);
    CU_ASSERT_EQUAL_FATAL(ac_buf[3], 44);
    CU_ASSERT_EQUAL_FATAL(an_buf[0], 55);
    CU_ASSERT_EQUAL_FATAL(an_buf[1], 66);
}

static int
vcz_suite_init(void)
{
    return CUE_SUCCESS;
}

static int
vcz_suite_cleanup(void)
{
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
            .pName = "vcz_compute",
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
        suite = CU_get_suite_by_name("vcz_compute", CU_get_registry());
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
        { "test_compute_ac_an_ploidy2", test_compute_ac_an_ploidy2 },
        { "test_compute_ac_an_an_only", test_compute_ac_an_an_only },
        { "test_compute_ac_an_ploidy1", test_compute_ac_an_ploidy1 },
        { "test_compute_ac_an_ploidy3", test_compute_ac_an_ploidy3 },
        { "test_compute_ac_an_invalid_genotype", test_compute_ac_an_invalid_genotype },
        { "test_compute_ac_an_invalid_num_alleles",
            test_compute_ac_an_invalid_num_alleles },
        { "test_compute_ac_an_zero_variants", test_compute_ac_an_zero_variants },
        { NULL, NULL },
    };
    return test_main(tests, argc, argv);
}
