#pragma once

#include <immintrin.h>
#include <stdint.h>

uint16_t *matrix_load_h5_data(const char *Path, const char *Dataset, int *m, int *n);

int *matrix_load_h5_indices(const char *Path, const char *Dataset, int *m, int *n);

static inline float compute_dot_product(const uint16_t *Matrix, int m, int n, int rowA, int rowB)
{
    if (rowA >= m || rowB >= m)
        return 1.0e8f;

    const uint16_t *vecA = Matrix + ((size_t)rowA * n);
    const uint16_t *vecB = Matrix + ((size_t)rowB * n);

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    size_t i = 0;
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

    float res[8] __attribute__((__aligned__(32)));
    _mm256_storeu_ps(res, acc0);
    float total_sum = 0.0;
    for (int j = 0; j < 8; j++)
    {
        total_sum += res[j];
    }

    // Clean up remaining elements (if n is not a multiple of 16)
    for (; i < n; i++)
    {
        float fa = _cvtsh_ss(vecA[i]);
        float fb = _cvtsh_ss(vecB[i]);
        total_sum += fa * fb;
    }

    return total_sum;
}

void matrix_store_h5_results(const char *Path, int m, int n, int *N, float *D, float t_build, float t_query, const char *Dataset, int p_qb, int p_qn, int p_k, int p_ln);
