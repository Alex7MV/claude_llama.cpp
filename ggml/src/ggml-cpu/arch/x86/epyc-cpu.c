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
// Processes 2 blocks per iteration using independent YMM dpbusd
// chains (avoids inserti64x4/extracti64x4 data motion overhead)
// with ZMM float accumulation (avoids hsum in hot loop).
//
// Math:
//   ux[i] = sx[i] ^ 0x80   => unsigned: sx[i] + 128 bias
//   dpbusd(ux, sy) = Σ((x[i]+128) × y[i]) = Σ(x×y) + 128×Σ(y)
//   correction = 128 × Σ(y)  per dword
//   Σ y per dword: maddubs sums adjacent signed bytes -> int16,
//                  then madd sums int16 pairs -> int32 per dword
//   result = dot - correction = Σ(x[i] × y[i])
// ---------------------------------------------------------------

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)

// Float reduction of an __m256 to scalar
static inline float hsum_ps_256(__m256 v) {
    __m128 bot = _mm256_castps256_ps128(v);
    __m128 top = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_hadd_ps(_mm_add_ps(bot, top), _mm_add_ps(bot, top));
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
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
    __m512 acc = _mm512_setzero_ps();

    for (; ib + 1 < nb; ib += 2) {
        __m256i sx0 = _mm256_loadu_si256((const __m256i *)x[ib].qs);
        __m256i sy0 = _mm256_loadu_si256((const __m256i *)y[ib].qs);
        __m256i ux0 = _mm256_xor_si256(sx0, _mm256_set1_epi8((int8_t)0x80));
        __m256i d0 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ux0, sy0);
        __m256i p0 = _mm256_maddubs_epi16(_mm256_set1_epi8(1), sy0);
        __m256i g0 = _mm256_madd_epi16(p0, _mm256_set1_epi16(1));
        d0 = _mm256_sub_epi32(d0, _mm256_mullo_epi32(g0, _mm256_set1_epi32(128)));

        __m256i sx1 = _mm256_loadu_si256((const __m256i *)x[ib + 1].qs);
        __m256i sy1 = _mm256_loadu_si256((const __m256i *)y[ib + 1].qs);
        __m256i ux1 = _mm256_xor_si256(sx1, _mm256_set1_epi8((int8_t)0x80));
        __m256i d1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ux1, sy1);
        __m256i p1 = _mm256_maddubs_epi16(_mm256_set1_epi8(1), sy1);
        __m256i g1 = _mm256_madd_epi16(p1, _mm256_set1_epi16(1));
        d1 = _mm256_sub_epi32(d1, _mm256_mullo_epi32(g1, _mm256_set1_epi32(128)));

        __m256 f0 = _mm256_cvtepi32_ps(d0);
        __m256 f1 = _mm256_cvtepi32_ps(d1);
        __m512i fdot_i = _mm512_inserti64x4(
            _mm512_castsi256_si512(_mm256_castps_si256(f0)),
            _mm256_castps_si256(f1), 1);
        __m512 fdot = _mm512_castsi512_ps(fdot_i);

        float s0 = GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d);
        float s1 = GGML_CPU_FP16_TO_FP32(x[ib + 1].d) * GGML_CPU_FP16_TO_FP32(y[ib + 1].d);
        __m256 slo = _mm256_set1_ps(s0);
        __m256 shi = _mm256_set1_ps(s1);
        __m512i scale_i = _mm512_inserti64x4(
            _mm512_castsi256_si512(_mm256_castps_si256(slo)),
            _mm256_castps_si256(shi), 1);
        __m512 scale = _mm512_castsi512_ps(scale_i);

        acc = _mm512_fmadd_ps(scale, fdot, acc);
    }

    for (; ib < nb; ++ib) {
        int sumi = 0;
        for (int j = 0; j < QK8_0; ++j) {
            sumi += x[ib].qs[j] * y[ib].qs[j];
        }
        total += (float)sumi * (GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d));
    }

    __m256 hi = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(acc), 1));
    __m256 lo = _mm512_castps512_ps256(acc);
    total += hsum_ps_256(_mm256_add_ps(lo, hi));
    *s = total;
}

#elif defined(__AVX512VNNI__)

// Fallback: ZMM dpbusd (no VL required)
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

        __m512i ux = _mm512_xor_si512(sx, _mm512_set1_epi8((int8_t)0x80));
        __m512i dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ux, sy);

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

// ---------------------------------------------------------------
// Q4_0 × Q8_0 VNNI vec_dot
// ---------------------------------------------------------------
// Q4_0 block: {ggml_half d; uint8_t qs[16]} = 18 bytes, 32 packed
// 4-bit nibbles (0..15). Dequantized to (-8..+7) by subtracting 8.
//
// Uses _mm256_dpbusd_epi32 (unsigned × signed):
//   dot  = dpbusd(nibbles_u8, q8)           = Σ (nibble × q8)
//   corr = dpbusd({8}_u8, q8)               = 8 × Σ (q8)
//   result = dot - corr                      = Σ ((nibble-8) × q8)
// ---------------------------------------------------------------
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)

void ggml_vec_dot_q4_0_q8_0_avx512_vnni(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx,
        const void * GGML_RESTRICT vy, size_t by, int nrc) {

    assert(n % QK4_0 == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_0 * GGML_RESTRICT x = (const block_q4_0 *) vx;
    const block_q8_0 * GGML_RESTRICT y = (const block_q8_0 *) vy;

    const int nb = n / QK4_0;

    __m256 acc = _mm256_setzero_ps();
    const __m256i zero = _mm256_setzero_si256();
    const __m256i c8   = _mm256_set1_epi8(8);  // 8 as unsigned byte

    for (int ib = 0; ib < nb; ib++) {
        // Expand 16 packed nibbles -> 32 uint8 values (0..15)
        __m128i nib = _mm_loadu_si128((const __m128i *)x[ib].qs);
        __m256i qx = _mm256_inserti128_si256(
            _mm256_castsi128_si256(nib),
            _mm_srli_epi16(nib, 4), 1);
        qx = _mm256_and_si256(qx, _mm256_set1_epi8(0x0F));

        __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);

        __m256i dot  = _mm256_dpbusd_epi32(zero, qx, qy);
        __m256i corr = _mm256_dpbusd_epi32(zero, c8,  qy);
        dot = _mm256_sub_epi32(dot, corr);

        __m256 f = _mm256_cvtepi32_ps(dot);
        float scale = GGML_CPU_FP16_TO_FP32(x[ib].d)
                    * GGML_CPU_FP16_TO_FP32(y[ib].d);
        acc = _mm256_fmadd_ps(f, _mm256_set1_ps(scale), acc);
    }

    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s128 = _mm_hadd_ps(_mm_add_ps(lo, hi), _mm_add_ps(lo, hi));
    s128 = _mm_hadd_ps(s128, s128);
    *s = _mm_cvtss_f32(s128);
}

#endif /* __AVX512VNNI__ && __AVX512VL__ */
#endif /* __AVX512VNNI__ */
