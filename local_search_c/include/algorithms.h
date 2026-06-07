#pragma once
#include "constants.h"
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

static inline __m256i popcnt(const __m256i v)
{
    const __m256i m1 = _mm256_set1_epi8(0x55);
    const __m256i m2 = _mm256_set1_epi8(0x33);
    const __m256i m4 = _mm256_set1_epi8(0x0F);

    const __m256i t1 = _mm256_sub_epi8(v, (_mm256_srli_epi16(v, 1) & m1));
    const __m256i t2 = _mm256_add_epi8(t1 & m2, (_mm256_srli_epi16(t1, 2) & m2));
    const __m256i t3 = _mm256_add_epi8(t2, _mm256_srli_epi16(t2, 4)) & m4;
    return _mm256_sad_epu8(t3, _mm256_setzero_si256());
}

static inline int hamming_portable(const uint64_t *restrict a, const uint64_t *restrict b, int bn)
{
#ifdef LS_AVX_512_POP_COUNT
    // Cast pointers to 512-bit vector types
    const __m512i *va = (const __m512i *)a;
    const __m512i *vb = (const __m512i *)b;

    // Load the 1024 bits (Two 512-bit vectors per word)
    __m512i a0 = _mm512_load_si512(&va[0]);
    __m512i b0 = _mm512_load_si512(&vb[0]);
    __m512i a1 = _mm512_load_si512(&va[1]);
    __m512i b1 = _mm512_load_si512(&vb[1]);

    // XOR the vectors together
    __m512i xor0 = _mm512_xor_si512(a0, b0);
    __m512i xor1 = _mm512_xor_si512(a1, b1);

    // Hardware Popcount on 512-bit registers (Requires AVX512_VPOPCNTDQ)
    __m512i pop0 = _mm512_popcnt_epi64(xor0);
    __m512i pop1 = _mm512_popcnt_epi64(xor1);

    // Sum the popcounts vertically
    __m512i total_pop = _mm512_add_epi64(pop0, pop1);

    // Reduce the 512-bit vector into a single integer scalar
    return _mm512_reduce_add_epi64(total_pop);
#else
    // const __m256i *p1 = (const __m256i *)a;
    // const __m256i *p2 = (const __m256i *)b;

    // uint64_t final_distance = 0;
    // // Load and XOR to isolate differing bits
    // for (int j = 0; j < 4; j++)
    // {
    //     __m256i total = popcnt(_mm256_xor_si256(
    //         _mm256_loadu_si256(p1 + j),
    //         _mm256_loadu_si256(p2 + j)));
    //     final_distance += (_mm256_extract_epi64(total, 0)) + (_mm256_extract_epi64(total, 1)) + (_mm256_extract_epi64(total, 2)) + (_mm256_extract_epi64(total, 3));
    // }
    // return final_distance;
    int c = 0;
    __builtin_prefetch(((const char *)a) + 64, 0, 3);
    __builtin_prefetch(((const char *)b) + 64, 0, 3);

    for (int i = 0; i < bn; i++)
        c += __builtin_popcountll(a[i] ^ b[i]);
    return c;
#endif
}

static inline void quantize_1024_fp16_avx2(const uint16_t *input,
                                           uint8_t *output)
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
        __m256 mask = _mm256_cmp_ps(v, zero, _CMP_GT_OQ);
        output[i] = (uint8_t)_mm256_movemask_ps(mask);
    }
}
static void prefetch_1024_bits(uint64_t *Matrix, size_t v)
{
    const void *address = Matrix + v * LOCAL_SEARCH_DIMS_IN_64BIT;
    // Cast to char* so pointer arithmetic operates in bytes
    const char *ptr = (const char *)address;

    // Prefetch the first 64 bytes (512 bits)
    __builtin_prefetch(ptr, 0, 2);

    // Prefetch the second 64 bytes (the remaining 512 bits)
    __builtin_prefetch(ptr + 64, 0, 2);
}