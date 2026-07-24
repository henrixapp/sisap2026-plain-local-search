#pragma once

#include <immintrin.h>
#include <stdint.h>

uint16_t *matrix_load_h5_data(const char *Path, const char *Dataset, int *m, int *n);

int *matrix_load_h5_indices(const char *Path, const char *Dataset, int *m, int *n);

static inline float dot_fp16_avx2(const uint16_t *vecA, const uint16_t *vecB, int n)
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    int i = 0;
    for (; i <= n - 16; i += 16)
    {
        __m256 a0 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(vecA + i)));
        __m256 b0 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(vecB + i)));
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);

        __m256 a1 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(vecA + i + 8)));
        __m256 b1 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(vecB + i + 8)));
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
    }

    acc0 = _mm256_add_ps(acc0, acc1);

    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    float total_sum = _mm_cvtss_f32(lo);

    for (; i < n; i++)
        total_sum += _cvtsh_ss(vecA[i]) * _cvtsh_ss(vecB[i]);

    return total_sum;
}

__attribute__((target("avx512f"))) static inline float
dot_fp16_avx512(const uint16_t *vecA, const uint16_t *vecB, int n)
{
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();

    int i = 0;
    for (; i <= n - 32; i += 32)
    {
        acc0 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)(vecA + i))),
                               _mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)(vecB + i))), acc0);
        acc1 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)(vecA + i + 16))),
                               _mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)(vecB + i + 16))), acc1);
    }
    for (; i <= n - 16; i += 16)
        acc0 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)(vecA + i))),
                               _mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)(vecB + i))), acc0);

    float total_sum = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));

    for (; i < n; i++)
        total_sum += _cvtsh_ss(vecA[i]) * _cvtsh_ss(vecB[i]);

    return total_sum;
}

// Symmetric int8 codes. Reranking is entirely memory bound, so halving the
// bytes per vector very nearly halves the pass. Precision is far more than the
// candidate ordering needs: the per component error is under 0.4% and it
// averages down by sqrt(n) over the 1024 term sum.
static inline int dot_i8_avx2(const int8_t *a, const int8_t *b, int n)
{
    __m256i acc = _mm256_setzero_si256();

    int i = 0;
    for (; i <= n - 16; i += 16)
    {
        __m256i a16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(a + i)));
        __m256i b16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(b + i)));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a16, b16));
    }

    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    int total = _mm_cvtsi128_si32(s);

    for (; i < n; i++)
        total += (int)a[i] * (int)b[i];

    return total;
}

__attribute__((target("avx512f,avx512bw"))) static inline int
dot_i8_avx512(const int8_t *a, const int8_t *b, int n)
{
    __m512i acc = _mm512_setzero_si512();

    int i = 0;
    for (; i <= n - 32; i += 32)
    {
        __m512i a16 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(a + i)));
        __m512i b16 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(b + i)));
        acc = _mm512_add_epi32(acc, _mm512_madd_epi16(a16, b16));
    }

    int total = _mm512_reduce_add_epi32(acc);

    for (; i < n; i++)
        total += (int)a[i] * (int)b[i];

    return total;
}

void matrix_store_h5_results(const char *Path, int m, int n, int *N, float *D, float t_build, float t_query, const char *Dataset, int p_qb, int p_qn, int p_k, int p_ln);
