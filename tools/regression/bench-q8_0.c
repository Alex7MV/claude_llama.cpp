// Self-contained Q8_0 vec_dot benchmark for AVX-512 VNNI.
// Compile with AOCC: clang -O3 -mavx512f -mavx512vnni -mavx512bf16 -mfma -o bench-q8_0 tools/regression/bench-q8_0.c
// Compile with AVX2:  clang -O3 -mavx2 -mfma -o bench-q8_0_avx2 tools/regression/bench-q8_0.c

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
// Alternative sign_epi8: hand-rolled from cmpgt + blendv to avoid
// potential AOCC intrinsic bug
// ---------------------------------------------------------------
#if defined(__AVX2__)
static inline __m256i my_sign_epi8(__m256i a, __m256i b) {
    __m256i zero = _mm256_setzero_si256();
    __m256i neg_a = _mm256_sub_epi8(zero, a);
    __m256i mask_neg = _mm256_cmpgt_epi8(zero, b);
    __m256i mask_pos = _mm256_cmpgt_epi8(b, zero);
    __m256i r = _mm256_blendv_epi8(zero, a, mask_pos);
    r = _mm256_blendv_epi8(r, neg_a, mask_neg);
    return r;
}

static inline __m256 my_sum_i16_pairs_float(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
    return _mm256_cvtepi32_ps(summed_pairs);
}

static inline __m256 my_mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return my_sum_i16_pairs_float(dot);
}

__attribute__((noinline))
static float vec_dot_avx2_fallback(const block_q8_0 *x, const block_q8_0 *y, int nb) {
    __m256 acc = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const __m256 d = _mm256_set1_ps(
            fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d));
        __m256i qx = _mm256_loadu_si256((const __m256i *)x[ib].qs);
        __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);
        __m256i ax = my_sign_epi8(qx, qx);
        __m256i sy = my_sign_epi8(qy, qx);
        const __m256 q = my_mul_sum_us8_pairs_float(ax, sy);
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

// -----------------------------------------------------------------
// Corrected AVX2: sign-extend to 16-bit, avoiding sign_epi8 -128 overflow
// -----------------------------------------------------------------
// sign_epi8 fails when y[i] = -128 and x[i] < 0 because -(-128) = 128
// overflows int8 and wraps back to -128. Sign-extension avoids this.
__attribute__((noinline))
static float vec_dot_avx2_signext(const block_q8_0 *x, const block_q8_0 *y, int nb) {
    __m256 acc = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const __m256 d = _mm256_set1_ps(
            fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d));
        __m256i qx = _mm256_loadu_si256((const __m256i *)x[ib].qs);
        __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);

        // Sign-extend bytes to 16-bit (two 128-bit halves per 256-bit reg)
        __m256i x_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(qx));
        __m256i x_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(qx, 1));
        __m256i y_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(qy));
        __m256i y_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(qy, 1));

        // Sum pairs of 16-bit products to 32-bit (8 per half)
        __m256i s_lo = _mm256_madd_epi16(x_lo, y_lo);
        __m256i s_hi = _mm256_madd_epi16(x_hi, y_hi);

        // Combine adjacent 2-element sums into 4-element sums
        // hadd(s_lo, s_hi) gives: [0-3, 4-7, 16-19, 20-23, 8-11, 12-15, 24-27, 28-31]
        // Need:              [0-3, 4-7, 8-11, 12-15, 16-19, 20-23, 24-27, 28-31]
        __m256i p = _mm256_hadd_epi32(s_lo, s_hi);
        const __m256i perm_idx = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
        p = _mm256_permutevar8x32_epi32(p, perm_idx);

        __m256 q = _mm256_cvtepi32_ps(p);
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

// Keep original ggml-style version for A/B comparison
static inline __m256 ggml_mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return my_sum_i16_pairs_float(dot);
}

__attribute__((noinline))
static float vec_dot_avx2_ggml(const block_q8_0 *x, const block_q8_0 *y, int nb) {
    __m256 acc = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const __m256 d = _mm256_set1_ps(
            fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d));
        __m256i qx = _mm256_loadu_si256((const __m256i *)x[ib].qs);
        __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);
        const __m256 q = ggml_mul_sum_i8_pairs_float(qx, qy);
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
// AVX-512 VNNI: dpbusd + XOR-0x80 + direct Σ y correction
// (no XOR of y, no 65536 bias; AOCC 5.2 lacks dpbssd intrinsic)
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

        // XOR-0x80: signed x -> unsigned (adds 128 bias per byte)
        __m512i ux = _mm512_xor_si512(sx, _mm512_set1_epi8((int8_t)0x80));
        __m512i dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ux, sy);

        // Σ y per dword: maddubs sums adjacent signed bytes -> int16,
        // then madd sums int16 pairs -> int32 per dword
        __m512i sy_psum = _mm512_maddubs_epi16(_mm512_set1_epi8(1), sy);
        __m512i sy_gsum = _mm512_madd_epi16(sy_psum, _mm512_set1_epi16(1));
        dot = _mm512_sub_epi32(dot, _mm512_mullo_epi32(sy_gsum, _mm512_set1_epi32(128)));

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
// Debug: verbose per-block comparison on failing case
// ---------------------------------------------------------------
typedef float (*vec_dot_fn)(const block_q8_0 *, const block_q8_0 *, int);
static int debug_on = 0;

static void debug_check(const char *name, vec_dot_fn fn,
                         const block_q8_0 *x, const block_q8_0 *y, int nb,
                         float expected) {
    float result = fn(x, y, nb);
    float rel_err = fabsf(expected) > 1e-10f
        ? fabsf(result - expected) / fabsf(expected)
        : fabsf(result - expected);
    printf("  %-20s full(%3d blks) = %12.1f  (expected=%12.1f  rel_err=%9.2e)  %s\n",
           name, nb, result, expected, rel_err, rel_err < 1e-3f ? "PASS" : "FAIL");

    if (debug_on) {
        int bad = -1;
        for (int step = 1; step <= nb; step++) {
            float sub = fn(x, y, step);
            float sub_exp = vec_dot_scalar(x, y, step);
            float sub_err = fabsf(sub_exp) > 1e-10f
                ? fabsf(sub - sub_exp) / fabsf(sub_exp) : fabsf(sub - sub_exp);
            if (sub_err >= 1e-3f) { bad = step; break; }
        }
        if (bad > 0) {
            printf("    Divergence at block %d.\n", bad - 1);
            printf("    Cumulative sums per block:\n");
            int start = (bad - 3) > 0 ? (bad - 3) : 0;
            int end = (bad + 5) < nb ? (bad + 5) : nb;
            for (int b = start; b < end; b++) {
                float scalar_cum = vec_dot_scalar(x, y, b + 1);
                float fn_cum = fn(x, y, b + 1);
                printf("      blk %3d: scalar_cum=%12.1f  fn_cum=%12.1f  diff=%g\n",
                       b, scalar_cum, fn_cum, fn_cum - scalar_cum);
                fflush(stdout);
            }
            int bb = bad - 1;
            printf("    block %d: x.d=0x%04x y.d=0x%04x scale_x=%f scale_y=%f\n",
                   bb, x[bb].d, y[bb].d,
                   fp16_to_fp32(x[bb].d), fp16_to_fp32(y[bb].d));
            printf("      x.qs[0..31]=");
            for (int j = 0; j < QK8_0; j++)
                printf("%d ", x[bb].qs[j]);
            printf("\n      y.qs[0..31]=");
            for (int j = 0; j < QK8_0; j++)
                printf("%d ", y[bb].qs[j]);
            printf("\n");
            // Also compute exact scalar dot product for this block
            int qs_dot = 0;
            for (int j = 0; j < QK8_0; j++)
                qs_dot += (int)x[bb].qs[j] * (int)y[bb].qs[j];
            float scale = fp16_to_fp32(x[bb].d) * fp16_to_fp32(y[bb].d);
            printf("      qs_dot=%d  scale=%f  exact_contrib=%f\n",
                   qs_dot, scale, (float)((double)qs_dot * (double)scale));
            // Check for sign_epi8 overflow (y=-128, x<0)
            int overflow_count = 0;
            for (int j = 0; j < QK8_0; j++)
                if (x[bb].qs[j] < 0 && y[bb].qs[j] == -128) overflow_count++;
            if (overflow_count > 0)
                printf("      *** ROOT CAUSE: %d byte(s) with y=-128 AND x<0 -> "
                       "sign_epi8 overflow (-(-128)=128 wraps to -128 in int8)\n",
                       overflow_count);
        }
        fflush(stdout);
    }
}

// ---------------------------------------------------------------
// Self-test with known values
// ---------------------------------------------------------------
static int self_test(void) {
    block_q8_0 xb, yb;
    xb.d = 0x3c00;
    yb.d = 0x3c00;
    for (int j = 0; j < QK8_0; j++) {
        xb.qs[j] = (int8_t)(j * 3 + 1);
        yb.qs[j] = (int8_t)(j * 2 + 5);
    }
    float expected = vec_dot_scalar(&xb, &yb, 1);
    printf("Self-test (1 block, scale=1.0, qs pattern):\n");
    printf("  scalar ref:            %f\n", expected);

    int ok = 1;
#if defined(__AVX2__)
    {
    float r = vec_dot_avx2_ggml(&xb, &yb, 1);
    float d = fabsf(r - expected);
    printf("  AVX2 (ggml sign):      %f  (diff=%g)  %s\n", r, d, d < 1e-3f ? "PASS" : "FAIL");
    ok = ok && (d < 1e-3f);
    }
    {
    float r = vec_dot_avx2_fallback(&xb, &yb, 1);
    float d = fabsf(r - expected);
    printf("  AVX2 (fallback sign):  %f  (diff=%g)  %s\n", r, d, d < 1e-3f ? "PASS" : "FAIL");
    ok = ok && (d < 1e-3f);
    }
    {
    float r = vec_dot_avx2_signext(&xb, &yb, 1);
    float d = fabsf(r - expected);
    printf("  AVX2 (sign-extension): %f  (diff=%g)  %s\n", r, d, d < 1e-3f ? "PASS" : "FAIL");
    ok = ok && (d < 1e-3f);
    }
#endif
#if defined(__AVX512VNNI__)
    {
    float r = vec_dot_avx512_vnni(&xb, &yb, 1);
    float d = fabsf(r - expected);
    printf("  AVX-512 VNNI:          %f  (diff=%g)  %s\n", r, d, d < 1e-3f ? "PASS" : "FAIL");
    ok = ok && (d < 1e-3f);
    }
#endif

    // Stress test with y=-128 to expose sign_epi8 overflow bug
    printf("\n  Stress test (y=-128, x<0):\n");
    xb.d = 0x3c00; yb.d = 0x3c00;
    memset(xb.qs, -105, QK8_0);
    memset(yb.qs, -128, QK8_0);
    float str_expected = vec_dot_scalar(&xb, &yb, 1);
    printf("    scalar ref:           %13.1f\n", str_expected);
    int str_ok = 1;
#if defined(__AVX2__)
    // ggml-sign is known-buggy for y=-128, shown for reference only
    {
    float r = vec_dot_avx2_ggml(&xb, &yb, 1);
    float d = fabsf(r - str_expected);
    printf("    AVX2 (ggml sign):      %13.1f  (diff=%g)  %s  %s\n",
           r, d, d < 1e-3f ? "PASS" : "FAIL",
           d > 1e-3f ? "(known bug: sign_epi8 -128 overflow)" : "");
    }
    {
    float r = vec_dot_avx2_signext(&xb, &yb, 1);
    float d = fabsf(r - str_expected);
    printf("    AVX2 (sign-extension): %13.1f  (diff=%g)  %s\n", r, d, d < 1e-3f ? "PASS" : "FAIL");
    str_ok = str_ok && (d < 1e-3f);
    }
#endif
#if defined(__AVX512VNNI__)
    {
    float r = vec_dot_avx512_vnni(&xb, &yb, 1);
    float d = fabsf(r - str_expected);
    printf("    AVX-512 VNNI:          %13.1f  (diff=%g)  %s\n", r, d, d < 1e-3f ? "PASS" : "FAIL");
    str_ok = str_ok && (d < 1e-3f);
    }
#endif
    printf("    Stress test: %s\n", str_ok ? "PASS" : "FAIL");
    printf("\n");
    return (ok && str_ok) ? 0 : 1;
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

static void bench(const char *name, vec_dot_fn fn,
                  const block_q8_0 *x, const block_q8_0 *y, int nb,
                  float expected, int *all_pass) {
    // Validate
    float result = fn(x, y, nb);
    float rel_err = fabsf(expected) > 1e-10f
        ? fabsf(result - expected) / fabsf(expected)
        : fabsf(result - expected);
    int correct = rel_err < 1e-3f;
    if (!correct && all_pass) {
        *all_pass = 0;
        // Debug: find where it diverges
        debug_check(name, fn, x, y, nb, expected);
    }

    // Warmup
    volatile float sink = 0.0f;
    for (int i = 0; i < 10; i++) sink += fn(x, y, nb);
    (void)sink;

    // Benchmark: call on full dataset repeatedly (larger datasets for
    // throughput); or on sliding windows for CSE resistance.
    double best_us = 1e100;
    int step = nb < 64 ? nb : 64;
    for (int round = 0; round < 5; round++) {
        volatile float sink2 = 0.0f;
        double start = now_sec();
        int64_t count = 0;
        double elapsed;
        do {
            if (nb <= 64) {
                // Small dataset: call full dataset, each call unique via sink
                sink2 += fn(x, y, nb);
                count++;
            } else {
                // Sliding window over the data
                for (int off = 0; off + step <= nb; off += step) {
                    sink2 += fn(x + off, y + off, step);
                }
                count += nb / step;
            }
            elapsed = now_sec() - start;
        } while (elapsed < 0.5);
        double us = count > 0 ? elapsed / (double)count * 1e6 : 0.0;
        if ((us < best_us) && (us > 0.0)) best_us = us;
        (void)sink2;
    }
    (void)sink;

    double ops = 2.0 * step * QK8_0;
    double gflops = best_us < 1e99 ? ops / (best_us * 1e3) : 0.0;

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

    // Enable debug for the failing case
    debug_on = 1;

    // Test sizes: small (L1) to large (L3/DRAM)
    const int test_nb[] = {8, 64, 256, 1024, 4096, 16384, 65536};
    const int num_sizes = sizeof(test_nb) / sizeof(test_nb[0]);

    // all_pass only tracks CORRECT implementations (sign-ext, VNNI).
    // ggml-sign and fallback are reference-only (known bug with y=-128).
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
        bench("AVX2 ggml-sign", vec_dot_avx2_ggml, x, y, nb, expected, NULL);
        bench("AVX2 fallback",  vec_dot_avx2_fallback, x, y, nb, expected, NULL);
        bench("AVX2 sign-ext",  vec_dot_avx2_signext, x, y, nb, expected, &all_pass);
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
