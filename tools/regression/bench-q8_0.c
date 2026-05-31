// Self-contained Q8_0 vec_dot benchmark for AVX-512 VNNI.
// Compile with AOCC: clang -O3 -mavx512f -mavx512vnni -mavx512bf16 -mfma -o bench-q8_0 tools/regression/bench-q8_0.c
// Compile with AVX2:  clang -O3 -mavx2 -mfma -o bench-q8_0_avx2 tools/regression/bench-q8_0.c
// Compile with GCC:   gcc -O3 -mavx512f -mavx512vnni -o bench-q8_0 tools/regression/bench-q8_0.c

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#define QK8_0 32

typedef struct {
    uint16_t d;
    int8_t  qs[QK8_0];
} block_q8_0;

// ---------------------------------------------------------------
// Reference: scalar dot product
// ---------------------------------------------------------------
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t v;
    if (exp == 0) {
        v = (sign << 31) | (118 - 15) << 23 | mant << 13;
    } else if (exp == 0x1f) {
        v = (sign << 31) | 0x7f800000 | mant << 13;
    } else {
        v = (sign << 31) | (exp - 15 + 0x7f) << 23 | mant << 13;
    }
    float f;
    memcpy(&f, &v, sizeof(f));
    return f;
}

static float vec_dot_scalar(const block_q8_0 *x, const block_q8_0 *y, int nb) {
    double total = 0.0;
    for (int i = 0; i < nb; i++) {
        int sumi = 0;
        for (int j = 0; j < QK8_0; j++) {
            sumi += (int)x[i].qs[j] * (int)y[i].qs[j];
        }
        total += (double)sumi * (double)(fp16_to_fp32(x[i].d) * fp16_to_fp32(y[i].d));
    }
    return (float)total;
}

// ---------------------------------------------------------------
// AVX2: sign_epi8 trick + maddubs (matches ggml exactly)
// ---------------------------------------------------------------
#if defined(__AVX2__)
static inline __m256 sum_i16_pairs_float(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
    return _mm256_cvtepi32_ps(summed_pairs);
}

static inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return sum_i16_pairs_float(dot);
}

static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return mul_sum_us8_pairs_float(ax, sy);
}

__attribute__((noinline))
static float vec_dot_avx2(const block_q8_0 *x, const block_q8_0 *y, int nb) {
    __m256 acc = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const __m256 d = _mm256_set1_ps(
            fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d));
        __m256i qx = _mm256_loadu_si256((const __m256i *)x[ib].qs);
        __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);
        const __m256 q = mul_sum_i8_pairs_float(qx, qy);
        acc = _mm256_fmadd_ps(d, q, acc);
    }
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float result;
    _mm_store_ss(&result, s);
    return result;
}
#endif

// ---------------------------------------------------------------
// AVX-512 VNNI: XOR-0x80 trick, corrected bias
// ---------------------------------------------------------------
#if defined(__AVX512VNNI__)
static int hsum_i32_8(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

__attribute__((noinline))
static float vec_dot_avx512_vnni(const block_q8_0 *x, const block_q8_0 *y, int nb) {
    int ib = 0;
    float total = 0.0f;

    for (; ib + 1 < nb; ib += 2) {
        __m256i sx_lo = _mm256_loadu_si256((const __m256i *)x[ib].qs);
        __m256i sx_hi = _mm256_loadu_si256((const __m256i *)x[ib + 1].qs);
        __m512i sx = _mm512_inserti64x4(_mm512_castsi256_si512(sx_lo), sx_hi, 1);

        __m256i sy_lo = _mm256_loadu_si256((const __m256i *)y[ib].qs);
        __m256i sy_hi = _mm256_loadu_si256((const __m256i *)y[ib + 1].qs);
        __m512i sy = _mm512_inserti64x4(_mm512_castsi256_si512(sy_lo), sy_hi, 1);

        __m512i xor_mask = _mm512_set1_epi8((int8_t)0x80);
        __m512i ux = _mm512_xor_si512(sx, xor_mask);
        __m512i dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ux, sy);

        // sy_gsum = Σ(y[i] ^ 0x80) per dword via maddubs+madd
        // Since (y[i] ^ 0x80) as unsigned = (y[i] + 128) as integer,
        // sy_gsum = Σ(y[i]) + 512 per dword (4 elements)
        // correction = 128 * sy_gsum = 128*Σy + 65536
        // dot - correction = (Σx*y + 128*Σy) - (128*Σy + 65536) = Σx*y - 65536
        __m512i sy_u = _mm512_xor_si512(sy, xor_mask);
        __m512i sy_psum = _mm512_maddubs_epi16(sy_u, _mm512_set1_epi8(1));
        __m512i sy_gsum = _mm512_madd_epi16(sy_psum, _mm512_set1_epi16(1));

        // correction and bias fix:
        __m512i corr = _mm512_mullo_epi32(sy_gsum, _mm512_set1_epi32(128));
        dot = _mm512_sub_epi32(dot, corr);
        dot = _mm512_add_epi32(dot, _mm512_set1_epi32(65536));

        int sum0 = hsum_i32_8(_mm512_castsi512_si256(dot));
        int sum1 = hsum_i32_8(_mm512_extracti64x4_epi64(dot, 1));

        float scale0 = fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d);
        float scale1 = fp16_to_fp32(x[ib + 1].d) * fp16_to_fp32(y[ib + 1].d);
        total += scale0 * (float)sum0 + scale1 * (float)sum1;
    }

    for (; ib < nb; ++ib) {
        int sumi = 0;
        for (int j = 0; j < QK8_0; ++j) {
            sumi += x[ib].qs[j] * y[ib].qs[j];
        }
        total += (float)sumi * (fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d));
    }
    return total;
}
#endif

// ---------------------------------------------------------------
// Self-test with known values
// ---------------------------------------------------------------
static int self_test(void) {
    block_q8_0 xb, yb;
    xb.d = 0x3c00; // 1.0
    yb.d = 0x3c00; // 1.0
    // fill with simple pattern: i*(i+3)
    for (int j = 0; j < QK8_0; j++) {
        xb.qs[j] = (int8_t)(j * 3 + 1);
        yb.qs[j] = (int8_t)(j * 2 + 5);
    }
    float expected = vec_dot_scalar(&xb, &yb, 1);
    float avx2_r = 0, vnni_r = 0;
#if defined(__AVX2__)
    avx2_r = vec_dot_avx2(&xb, &yb, 1);
    double e_avx2 = fabsf(expected) > 1e-10f
        ? fabsf(avx2_r - expected) / fabsf(expected) : fabsf(avx2_r - expected);
    printf("Self-test (1 block, known values):\n");
    printf("  scalar ref: %f\n", expected);
    printf("  AVX2:       %f  (rel_err=%g)  %s\n", avx2_r, e_avx2, e_avx2 < 1e-3f ? "PASS" : "FAIL");
#endif
#if defined(__AVX512VNNI__)
    vnni_r = vec_dot_avx512_vnni(&xb, &yb, 1);
    double e_vnni = fabsf(expected) > 1e-10f
        ? fabsf(vnni_r - expected) / fabsf(expected) : fabsf(vnni_r - expected);
    printf("  AVX-512 VNNI: %f  (rel_err=%g)  %s\n", vnni_r, e_vnni, e_vnni < 1e-3f ? "PASS" : "FAIL");
#endif
    printf("\n");
    return (fabsf(expected - avx2_r) < 1e-3f * fmaxf(1.0f, fabsf(expected)) &&
            fabsf(expected - vnni_r) < 1e-3f * fmaxf(1.0f, fabsf(expected))) ? 0 : 1;
}

// ---------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_blocks(block_q8_0 *blocks, int nb, unsigned seed) {
    srand(seed);
    for (int i = 0; i < nb; i++) {
        uint16_t exp = 0x3c00 + (rand() & 0x7);
        uint16_t mant = rand() & 0x3ff;
        blocks[i].d = exp | mant;
        for (int j = 0; j < QK8_0; j++) {
            blocks[i].qs[j] = (int8_t)(rand() & 0xff);
        }
    }
}

typedef float (*vec_dot_fn)(const block_q8_0 *, const block_q8_0 *, int);

static void bench(const char *name, vec_dot_fn fn,
                  const block_q8_0 *x, const block_q8_0 *y, int nb,
                  float expected, int *all_pass) {
    // Validate
    float result = fn(x, y, nb);
    float rel_err = fabsf(expected) > 1e-10f
        ? fabsf(result - expected) / fabsf(expected)
        : fabsf(result - expected);
    int correct = rel_err < 1e-3f;
    if (!correct) *all_pass = 0;

    // Warmup
    volatile float sink = 0.0f;
    for (int i = 0; i < 10; i++) sink += fn(x, y, nb);
    (void)sink;

    // Benchmark: call fn on entire dataset repeatedly
    // Use volatile sink to force evaluation and prevent CSE
    double best_us = 1e100;
    for (int round = 0; round < 5; round++) {
        volatile float sink2 = 0.0f;
        double start = now_sec();
        int64_t count = 0;
        double elapsed;
        do {
            // Call fn on fresh data slices so each call is unique
            // and the compiler cannot CSE across calls
            int step = 64;
            for (int off = 0; off + step <= nb; off += step) {
                sink2 += fn(x + off, y + off, step);
            }
            count += nb / step;
            elapsed = now_sec() - start;
        } while (elapsed < 0.5);
        double us = elapsed / (double)count * 1e6;
        if (us < best_us) best_us = us;
        (void)sink2;
    }
    (void)sink;

    // ops per call = 2 * step * QK8_0  (step = 64 blocks, or nb if nb < 64)
    int step = nb < 64 ? nb : 64;
    double ops = 2.0 * step * QK8_0;
    double gflops = ops / (best_us * 1e3);

    printf("  %-20s %10s  %8.2f us/call (%4d blks)  %9.2f GFLOPS\n",
           name, correct ? "PASS" : "FAIL", best_us, step, gflops);
}

int main(void) {
    printf("Q8_0 vec_dot benchmark\n");
    printf("Compiler: %s\n", __VERSION__);
#if defined(__AVX512VNNI__)
    printf("AVX-512 VNNI:  YES\n");
#else
    printf("AVX-512 VNNI:  NO\n");
#endif
#if defined(__AVX2__)
    printf("AVX2:          YES\n");
#endif
    printf("----------------------------------------\n");

    if (self_test() != 0) {
        printf("Self-test FAILED\n");
        return 1;
    }

    const int test_nb[] = {8, 64, 256, 1024, 4096, 8192};
    const int num_sizes = sizeof(test_nb) / sizeof(test_nb[0]);

    int all_pass = 1;

    for (int si = 0; si < num_sizes; si++) {
        int nb = test_nb[si];
        int n  = nb * QK8_0;

        block_q8_0 *x = malloc(sizeof(block_q8_0) * nb);
        block_q8_0 *y = malloc(sizeof(block_q8_0) * nb);
        if (!x || !y) { fprintf(stderr, "OOM\n"); return 1; }

        fill_blocks(x, nb, 42);
        fill_blocks(y, nb, 12345);

        float expected = vec_dot_scalar(x, y, nb);

        printf("\nn=%5d (%4d blocks):\n", n, nb);

#if defined(__AVX2__)
        bench("AVX2 x86", vec_dot_avx2, x, y, nb, expected, &all_pass);
#endif
#if defined(__AVX512VNNI__)
        bench("AVX-512 VNNI", vec_dot_avx512_vnni, x, y, nb, expected, &all_pass);
#endif

        free(x);
        free(y);
    }

    printf("\n----------------------------------------\n");
    printf("All tests: %s\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 1;
}
