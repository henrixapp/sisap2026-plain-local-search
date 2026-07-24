#pragma once

#include <immintrin.h>
#include <omp.h>
#include <stdint.h>

// Must have external linkage: a static __thread in a header would give every
// translation unit its own zero initialised copy, and a zero seeded xorshift
// never leaves zero.
extern __thread uint32_t thread_seed;

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
    int c = 0;
    for (int i = 0; i < n; i++)
        c += __builtin_popcountll(a[i] ^ b[i]);
    return c;
}

static inline int hamming_avx2_lut(const uint64_t *a, const uint64_t *b, int n)
{
    // 4-bit look-up table for popcount values mapped across bytes
    __m256i low_mask = _mm256_set1_epi8(0x0F);
    __m256i lut = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    __m256i total_cnt = _mm256_setzero_si256();

    for (int i = 0; i < n; i += 4)
    {
        __m256i va = _mm256_load_si256((const __m256i *)&a[i]);
        __m256i vb = _mm256_load_si256((const __m256i *)&b[i]);
        __m256i vxor = _mm256_xor_si256(va, vb);

        // Split bytes into low and high 4-bit nibbles
        __m256i low_bits = _mm256_and_si256(vxor, low_mask);
        __m256i high_bits = _mm256_and_si256(_mm256_srli_epi32(vxor, 4), low_mask);

        // Lookup bit counts via register shuffle mapping
        __m256i cnt1 = _mm256_shuffle_epi8(lut, low_bits);
        __m256i cnt2 = _mm256_shuffle_epi8(lut, high_bits);

        // Accumulate byte counts into the running total
        total_cnt = _mm256_add_epi8(total_cnt, _mm256_add_epi8(cnt1, cnt2));
    }

    // Horizontal sum of the accumulated bytes
    __m256i sad = _mm256_sad_epu8(total_cnt, _mm256_setzero_si256());
    uint64_t buffer[4] __attribute__((__aligned__(32)));
    _mm256_store_si256((__m256i *)buffer, sad);

    return (int)(buffer[0] + buffer[1] + buffer[2] + buffer[3]);
}

// Hardware popcount version. Roughly 1.7x the throughput of the LUT variant,
// only reachable on machines that report avx512vpopcntdq at runtime.
__attribute__((target("avx512f,avx512vpopcntdq"))) static inline int
hamming_avx512(const uint64_t *a, const uint64_t *b, int n)
{
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();

    int i = 0;
    for (; i + 16 <= n; i += 16)
    {
        __m512i x0 = _mm512_xor_si512(_mm512_load_si512((const void *)&a[i]),
                                      _mm512_load_si512((const void *)&b[i]));
        __m512i x1 = _mm512_xor_si512(_mm512_load_si512((const void *)&a[i + 8]),
                                      _mm512_load_si512((const void *)&b[i + 8]));
        acc0 = _mm512_add_epi64(acc0, _mm512_popcnt_epi64(x0));
        acc1 = _mm512_add_epi64(acc1, _mm512_popcnt_epi64(x1));
    }
    for (; i + 8 <= n; i += 8)
    {
        __m512i x = _mm512_xor_si512(_mm512_load_si512((const void *)&a[i]),
                                     _mm512_load_si512((const void *)&b[i]));
        acc0 = _mm512_add_epi64(acc0, _mm512_popcnt_epi64(x));
    }

    return (int)_mm512_reduce_add_epi64(_mm512_add_epi64(acc0, acc1));
}

static inline void quantize_8_fp16_avx2(const uint16_t *input, uint8_t *output, int n)
{
    __m256 zero = _mm256_setzero_ps();

    int i = 0;
    for (; i <= n - 8; i += 8)
    {
        // Load 8 half-precision floats into a 128-bit register
        __m128i half_v = _mm_loadu_si128((__m128i *)&input[i]);

        // Convert those 8 half-floats to 8 single-precision (32-bit) floats
        __m256 v = _mm256_cvtph_ps(half_v);

        // Compare and get mask as before
        __m256 mask = _mm256_cmp_ps(v, zero, _CMP_GT_OQ);
        output[i / 8] = (uint8_t)_mm256_movemask_ps(mask);
    }

    for (; i < n; i++)
    {
        float v = _cvtsh_ss(input[i]);
        if (v > 0)
            output[i / 8] |= (1 << (i % 8));
    }
}

// Runtime CPU feature probe, resolved once in main.
static inline int cpu_has_avx512_popcnt(void)
{
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vpopcntdq");
}
