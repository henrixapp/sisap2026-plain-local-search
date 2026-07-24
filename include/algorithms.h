#pragma once

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
    int c = 0;
    for (int i = 0; i < n; i++)
        c += __builtin_popcountll(a[i] ^ b[i]);
    return c;
}

#define POPCNT_LUT                             \
    _mm256_setr_epi8(                          \
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, \
        3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, \
        2, 3, 2, 3, 3, 4)

// Byte-wise popcount of a 256-bit word using the 4-bit nibble LUT
static inline __m256i popcnt_bytes(__m256i v, __m256i lut, __m256i low_mask)
{
    __m256i lo = _mm256_and_si256(v, low_mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi32(v, 4), low_mask);
    return _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo), _mm256_shuffle_epi8(lut, hi));
}

// Horizontal sum of byte lanes, kept in vector registers (no store/reload).
static inline int hsum_bytes(__m256i acc)
{
    __m256i sad = _mm256_sad_epu8(acc, _mm256_setzero_si256());
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(sad), _mm256_extracti128_si256(sad, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si32(s);
}

static inline int hamming_avx2_lut(const uint64_t *a, const uint64_t *b, int n)
{
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    const __m256i lut = POPCNT_LUT;
    __m256i acc = _mm256_setzero_si256();

    for (int i = 0; i < n; i += 4)
    {
        __m256i x = _mm256_xor_si256(_mm256_load_si256((const __m256i *)&a[i]),
                                     _mm256_load_si256((const __m256i *)&b[i]));
        acc = _mm256_add_epi8(acc, popcnt_bytes(x, lut, low_mask));
    }

    return hsum_bytes(acc);
}

/*
    Two distances against a shared left operand. The `a` loads are shared and the
    two dependency chains interleave, which keeps the shuffle ports busy.
*/
static inline void hamming_avx2_lut_x2(const uint64_t *a, const uint64_t *b0, const uint64_t *b1,
                                       int n, int *d0, int *d1)
{
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    const __m256i lut = POPCNT_LUT;
    __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256();

    for (int i = 0; i < n; i += 4)
    {
        __m256i va = _mm256_load_si256((const __m256i *)&a[i]);
        __m256i x0 = _mm256_xor_si256(va, _mm256_load_si256((const __m256i *)&b0[i]));
        __m256i x1 = _mm256_xor_si256(va, _mm256_load_si256((const __m256i *)&b1[i]));
        acc0 = _mm256_add_epi8(acc0, popcnt_bytes(x0, lut, low_mask));
        acc1 = _mm256_add_epi8(acc1, popcnt_bytes(x1, lut, low_mask));
    }

    *d0 = hsum_bytes(acc0);
    *d1 = hsum_bytes(acc1);
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

/*
    Reranking uses int8 codes instead of the raw fp16 rows. The candidate scan is
    bound by DRAM traffic, so halving the bytes per vector is what matters; the
    per-vector scale keeps the dot products faithful enough to order candidates
    that the exact fp16 ranking would separate.
*/
static inline float quantize_i8_fp16_avx2(const uint16_t *input, int8_t *output, int n)
{
    __m256 amax = _mm256_setzero_ps();
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    for (int i = 0; i < n; i += 8)
    {
        __m256 v = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)&input[i]));
        amax = _mm256_max_ps(amax, _mm256_and_ps(v, sign_mask));
    }

    float buf[8] __attribute__((__aligned__(32)));
    _mm256_store_ps(buf, amax);
    float mx = buf[0];
    for (int j = 1; j < 8; j++)
        if (buf[j] > mx)
            mx = buf[j];

    if (mx <= 0.0f)
    {
        for (int i = 0; i < n; i++)
            output[i] = 0;
        return 0.0f;
    }

    const float scale = 127.0f / mx;
    const __m256 vscale = _mm256_set1_ps(scale);

    for (int i = 0; i < n; i += 16)
    {
        __m256 v0 = _mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)&input[i])), vscale);
        __m256 v1 = _mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)&input[i + 8])), vscale);

        __m256i i0 = _mm256_cvtps_epi32(v0);
        __m256i i1 = _mm256_cvtps_epi32(v1);

        // 16 x int32 -> 16 x int8, saturating. The lane order of the AVX2 pack
        // instructions is fixed up with a final permute.
        __m256i p16 = _mm256_packs_epi32(i0, i1);
        __m256i p8 = _mm256_packs_epi16(p16, p16);
        p8 = _mm256_permutevar8x32_epi32(p8, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm_storeu_si128((__m128i *)&output[i], _mm256_castsi256_si128(p8));
    }

    return mx / 127.0f;
}

static inline float dot_f16_avx2(const uint16_t *a, const uint16_t *b, int n)
{
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();

    for (int i = 0; i < n; i += 16)
    {
        acc0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_load_si128((const __m128i *)(a + i))),
                               _mm256_cvtph_ps(_mm_load_si128((const __m128i *)(b + i))), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_load_si128((const __m128i *)(a + i + 8))),
                               _mm256_cvtph_ps(_mm_load_si128((const __m128i *)(b + i + 8))), acc1);
    }

    __m256 s = _mm256_add_ps(acc0, acc1);
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
    return _mm_cvtss_f32(h);
}

// Exact int32 dot product of two int8 vectors (n a multiple of 32).
static inline int dot_i8_avx2(const int8_t *a, const int8_t *b, int n)
{
    __m256i acc = _mm256_setzero_si256();

    for (int i = 0; i < n; i += 32)
    {
        __m256i va = _mm256_load_si256((const __m256i *)&a[i]);
        __m256i vb = _mm256_load_si256((const __m256i *)&b[i]);

        __m256i alo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
        __m256i ahi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
        __m256i blo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
        __m256i bhi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));

        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(alo, blo));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(ahi, bhi));
    }

    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}
