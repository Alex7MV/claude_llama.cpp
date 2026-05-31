#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "simd-mappings.h"

#include "../../quants.h"
#include "../../ggml-cpu-impl.h"

#include <immintrin.h>
#include <assert.h>

#define UNUSED GGML_UNUSED

// ---------------------------------------------------------------
// AVX-512 VNNI vec_dot for Q8_0 (signed INT8 × signed INT8)
// ---------------------------------------------------------------
// Q8_0 block: {ggml_half d; int8_t qs[32]} = 34 bytes
// Processes 2 blocks per iteration (64 INT8 pairs).
//
// Uses _mm512_dpbusd_epi32 (unsigned × signed) with XOR-0x80 to
// convert signed x to unsigned. Computes Σ y per dword directly
// (without XORing y) for the -128 correction:
//
//   ux[i] = sx[i] ^ 0x80  => unsigned: sx[i] + 128 bias
//   dot = dpbusd(ux, sy)   = Σ((x[i]+128) × y[i]) per dword
//                           = Σ(x[i]×y[i]) + 128×Σ(y[i])
//   correction = 128 × Σ(y[i]) per dword
//   result = dot - correction  =  Σ(x[i]×y[i])
// ---------------------------------------------------------------

#if defined(__AVX512VNNI__)

static inline int hsum_i32_8(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

void ggml_vec_dot_q8_0_q8_0_avx512_vnni(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx,
        const void * GGML_RESTRICT vy, size_t by, int nrc) {

    assert(n % QK8_0 == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q8_0 * GGML_RESTRICT x = (const block_q8_0 *) vx;
    const block_q8_0 * GGML_RESTRICT y = (const block_q8_0 *) vy;

    const int nb = n / QK8_0;
    int ib = 0;

    float total = 0.0f;

    for (; ib + 1 < nb; ib += 2) {
        __m256i sx_lo = _mm256_loadu_si256((const __m256i *)x[ib].qs);
        __m256i sx_hi = _mm256_loadu_si256((const __m256i *)x[ib + 1].qs);
        __m512i sx = _mm512_inserti64x4(_mm512_castsi256_si512(sx_lo), sx_hi, 1);

        __m256i sy_lo = _mm256_loadu_si256((const __m256i *)y[ib].qs);
        __m256i sy_hi = _mm256_loadu_si256((const __m256i *)y[ib + 1].qs);
        __m512i sy = _mm512_inserti64x4(_mm512_castsi256_si512(sy_lo), sy_hi, 1);

        __builtin_prefetch(&x[ib + 3], 0, 3);
        __builtin_prefetch(&y[ib + 3], 0, 3);
        __builtin_prefetch(&x[ib + 5], 0, 3);
        __builtin_prefetch(&y[ib + 5], 0, 3);

        // XOR-0x80: signed x -> unsigned (adds 128 bias per byte)
        __m512i ux = _mm512_xor_si512(sx, _mm512_set1_epi8((int8_t)0x80));
        __m512i dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ux, sy);

        // Σ y per dword: maddubs sums adjacent signed bytes -> int16,
        // then madd sums int16 pairs -> int32 per dword
        __m512i sy_psum = _mm512_maddubs_epi16(_mm512_set1_epi8(1), sy);
        __m512i sy_gsum = _mm512_madd_epi16(sy_psum, _mm512_set1_epi16(1));
        dot = _mm512_sub_epi32(dot, _mm512_mullo_epi32(sy_gsum, _mm512_set1_epi32(128)));

        __m256i blk0 = _mm512_castsi512_si256(dot);
        __m256i blk1 = _mm512_extracti64x4_epi64(dot, 1);

        int sum0 = hsum_i32_8(blk0);
        int sum1 = hsum_i32_8(blk1);

        float d0 = GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d);
        float d1 = GGML_CPU_FP16_TO_FP32(x[ib + 1].d) * GGML_CPU_FP16_TO_FP32(y[ib + 1].d);

        total += d0 * (float)sum0 + d1 * (float)sum1;
    }

    for (; ib < nb; ++ib) {
        int sumi = 0;
        for (int j = 0; j < QK8_0; ++j) {
            sumi += x[ib].qs[j] * y[ib].qs[j];
        }
        total += (float)sumi * (GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d));
    }

    *s = total;
}

#endif /* __AVX512VNNI__ */
