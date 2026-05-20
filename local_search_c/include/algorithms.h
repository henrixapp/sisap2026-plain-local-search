#pragma once
#include <immintrin.h>
#include <stdint.h>

static inline int hamming_portable(const uint64_t *restrict a, const uint64_t *restrict b,
                                   int n)
{
    int c = 0;
    for (int i = 0; i < 16; i++)
        c += __builtin_popcountll(a[i] ^ b[i]);
    return c;
}

static inline void quantize_1024_fp16_avx2(const uint16_t *restrict input,
                                           uint8_t *output, int comp)
{
    __m256 zero = _mm256_setzero_ps();

    // 1024 floats / 8 floats per SIMD register = 128 iterations
    for (int i = 0; i < 128; i++)
    {
        // Load 8 half-precision floats into a 128-bit register
        __m128i half_v = _mm_loadu_si128((__m128i *)&input[i * 8]);

        // Convert those 8 half-floats to 8 single-precision (32-bit) floats
        __m256 v = _mm256_cvtph_ps(half_v);

        // Compare and get mask as before
        __m256 mask = _mm256_cmp_ps(v, zero, comp);
        output[i] = (uint8_t)_mm256_movemask_ps(mask);
    }
}
static inline void minimize_to_8(uint8_t *input, uint16_t *output)
{
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            output[i] |= (output[i] << 2) + input[i * 8 + j] & 0xF ^ (input[i * 8 + j] >> 4);
        }
    }
}