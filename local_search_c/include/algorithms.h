#pragma once

#include "constants.h"
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
static __thread uint32_t thread_seed;

// Simple, fast 32-bit PRNG (Xorshift)
static inline uint32_t fast_rand()
{
    uint32_t x = thread_seed;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return thread_seed = x;
}

static inline void init_thread_seed(uint32_t global_seed, int thread_id)
{
    // Mix global seed with thread_id so threads don't produce the same sequence
    thread_seed = global_seed + thread_id + 1;
    if (thread_seed == 0)
        thread_seed = 1; // Xorshift cannot have a 0 seed
}

static inline void prepare_random()
{
#pragma omp parallel
    {
        // Each thread gets a unique starting point
        int tid = omp_get_thread_num();
        init_thread_seed(42, tid); // 42 is your global seed
    }
}

static inline int hamming_portable(const uint64_t *a, const uint64_t *b, int n)
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
    int c = 0;
    for (int i = 0; i < n; i++)
        c += __builtin_popcountll(a[i] ^ b[i]);
    return c;
#endif
}

static inline void quantize_8_fp16_avx2(const uint16_t *input, uint8_t *output, int n)
{
    __m256 zero = _mm256_setzero_ps();

    for (int i = 0; i < n; i += 8)
    {
        // Load 8 half-precision floats into a 128-bit register
        __m128i half_v = _mm_loadu_si128((__m128i *)&input[i]);

        // Convert those 8 half-floats to 8 single-precision (32-bit) floats
        __m256 v = _mm256_cvtph_ps(half_v);

        // Compare and get mask as before
        __m256 mask = _mm256_cmp_ps(v, zero, _CMP_GT_OQ);
        output[i / 8] = (uint8_t)_mm256_movemask_ps(mask);
    }
}
