#pragma once

#include "algorithms.h"
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <string.h>

/*
    A plain test-and-set spinlock rather than omp_lock_t: these are held for a few
    dozen cycles and are almost never contended (8 threads over 8192 locks), so the
    out-of-line call into libgomp on both acquire and release was the bulk of the
    cost.
*/
typedef struct
{
    int flag;
    char padding[60];
} PaddedLock;

static inline void knn_lock(PaddedLock *l)
{
    while (__atomic_exchange_n(&l->flag, 1, __ATOMIC_ACQUIRE))
        while (__atomic_load_n(&l->flag, __ATOMIC_RELAXED))
            _mm_pause();
}

static inline void knn_unlock(PaddedLock *l)
{
    __atomic_store_n(&l->flag, 0, __ATOMIC_RELEASE);
}

typedef struct
{
    int v;
    int d;
} VDEntry;

typedef struct KNN_struct
{
    int n;              // Number of vertices
    int k;              // Size of neighborhoods
    int *V;             // List of neighbor IDs (SOA)
    int16_t *D;         // List of neighbor distances (SOA)
    int *Dmax;          // Worst distance per vertex, kept separately
    PaddedLock L[8192]; // Locks for mutual exclusion
} KNN;

/*
    Dmax duplicates the worst distance of every neighbourhood in a compact array.
    Rejecting a candidate is by far the common case, and testing it against Dmax
    touches 4 bytes out of a 25 MB array instead of pulling a cache line out of
    the 1.6 GB distance table.
*/

KNN *KNN_init(int n, int k, float ref);

void KNN_free(KNN *g);

int KNN_validate(KNN *g);

void KNN_sort_lists(KNN *g);

/*
    Scratch table for candidate deduplication, one per thread.

    A point's neighbour-of-neighbour set contains the same w many times over,
    since nearby points share neighbours. Re-deriving a distance costs a couple of
    random cache lines; a probe costs a few cycles. Re-offering a candidate is a
    no-op anyway (Dmax only ever falls, and inserts are idempotent), so dropping
    the repeats changes the resulting graph in no way.
*/
#define KNN_SEEN_BITS 11
#define KNN_SEEN_SIZE (1 << KNN_SEEN_BITS)

static inline int knn_seen_probe(uint64_t *seen, int u, int w)
{
    uint64_t key = ((uint64_t)(u + 1) << 32) | (uint32_t)w;
    uint32_t h = ((uint32_t)w * 2654435761u) >> (32 - KNN_SEEN_BITS);
    if (seen[h] == key)
        return 1;
    seen[h] = key;
    return 0;
}

void KNN_greedy_dist1_update_simple(KNN *g, int u, const uint64_t *Matrix, int m, int n, uint64_t *seen);

void KNN_greedy_dist1_update_simple_round2(KNN *g, int u, const uint64_t *Matrix, int m, int n, uint64_t *seen);

float KNN_compute_recall(KNN *g, const int *KNNs, int m, int n, int k);

float KNN_compute_recall_standalone(int *ANNs, const int *KNNs, int m, int n, int k);

/*
    Insert used while the neighbourhoods are still being filled (indexing phase).

    The list is kept unsorted with the current maximum pinned to slot 0. Landing a
    candidate is then a slot-0 overwrite plus a rescan for the new maximum, which
    avoids the memmove of a sorted insert. During indexing most offers succeed, so
    that memmove is the dominant cost; the ordering is restored once, afterwards.
*/
#define KNN_TAG_BITS 6
#define KNN_TAG_MASK ((1 << KNN_TAG_BITS) - 1)

static inline void KNN_offer_max0(KNN *g, int u, int v, int d)
{
    const int k = g->k;
    int *V = g->V + (size_t)u * k;
    uint16_t *D = (uint16_t *)(g->D + (size_t)u * k);
    const uint16_t dp = (uint16_t)((d > 1023 ? 1023 : d) << KNN_TAG_BITS);

    if (dp >= D[0])
        return;

    // Duplicate check, one pass with the comparisons OR-ed together
    __m256i target = _mm256_set1_epi32(v);
    __m256i hits = _mm256_setzero_si256();
    for (int i = 0; i < k; i += 8)
        hits = _mm256_or_si256(hits, _mm256_cmpeq_epi32(_mm256_load_si256((const __m256i *)&V[i]), target));
    if (!_mm256_testz_si256(hits, hits))
        return;

    /*
        Locate the new maximum. Through the indexing phase D holds each distance
        already shifted left with its own slot index in the low bits, so one plain
        max-reduction yields both the largest distance and where it lives -- no
        second, badly predicted scan to find the argmax, and no per-element
        tagging work inside the loop.

        Slot 0 is blended in registers rather than stored then reloaded (avoiding
        a store-forwarding stall on every successful offer).
    */
    __m128i d0_low = _mm_blend_epi16(_mm_load_si128((const __m128i *)&D[0]), _mm_set1_epi16(dp), 0x01);
    __m128i d0_high = _mm_load_si128((const __m128i *)&D[8]);
    __m256i mx = _mm256_set_m128i(d0_high, d0_low);

    for (int i = 16; i < k; i += 16)
        mx = _mm256_max_epu16(mx, _mm256_load_si256((const __m256i *)&D[i]));

    mx = _mm256_max_epu16(mx, _mm256_permute2x128_si256(mx, mx, 0x01));
    __m128i m128 = _mm256_castsi256_si128(mx);
    m128 = _mm_max_epu16(m128, _mm_shuffle_epi32(m128, _MM_SHUFFLE(1, 0, 3, 2)));
    m128 = _mm_max_epu16(m128, _mm_shuffle_epi32(m128, _MM_SHUFFLE(2, 3, 0, 1)));
    m128 = _mm_max_epu16(m128, _mm_srli_si128(m128, 2));

    uint16_t packed = (uint16_t)_mm_cvtsi128_si32(m128);
    int p = packed & KNN_TAG_MASK;

    if (p == 0)
    {
        V[0] = v;
        D[0] = dp;
    }
    else
    {
        V[0] = V[p];
        D[0] = D[p] & ~KNN_TAG_MASK; // slot 0 always carries tag 0
        V[p] = v;
        D[p] = dp | p;
    }

    g->Dmax[u] = D[0] >> KNN_TAG_BITS;
}

/*
    Insert used during refinement, where the list must stay sorted by distance so
    that the neighbour-of-neighbour scan can walk the closest entries first.
*/
static inline void KNN_insert_unsafe(KNN *g, int u, int v, int d)
{
    int *V = g->V + ((size_t)u * g->k);
    int16_t *D = g->D + ((size_t)u * g->k);

    // Stop if d is worse than the worst neighbor
    if (D[g->k - 1] <= d)
        return;

    // Duplicate check and find insertion position using AVX2
    __m256i target_v = _mm256_set1_epi32(v);
    __m128i target_d = _mm_set1_epi16((int16_t)d);

    int pos = g->k;

    for (int i = 0; i < g->k; i += 8)
    {
        __m256i chunk_v = _mm256_load_si256((const __m256i *)&V[i]);
        __m128i chunk_d = _mm_load_si128((const __m128i *)&D[i]);

        __m256i cmp_v = _mm256_cmpeq_epi32(chunk_v, target_v);
        if (_mm256_movemask_epi8(cmp_v) != 0)
            return;

        // Find first element > d. Since D is sorted ascending, we look for first D[i] > d.
        __m128i cmp_d = _mm_cmpgt_epi16(chunk_d, target_d);
        int mask = _mm_movemask_epi8(cmp_d);
        if (mask != 0)
        {
            pos = i + (__builtin_ctz(mask) >> 1);
            break;
        }
    }

    if (pos < g->k)
    {
        // Every slot before pos was covered by the duplicate scan above, and a
        // repeat of v would sit there (same pair always yields the same distance).

        /*
            Open a hole at pos. memmove with a runtime length compiles to a libc
            call, which dominates a shift this short; do it inline in 8-element blocks
            from the top. Each block's source is disjoint from the previous block's
            destination, and the block at pos is loaded up front because the loop
            overwrites it (with the same values, so the final store is consistent).
        */
        if (pos <= g->k - 9)
        {
            __m256i sv = _mm256_loadu_si256((const __m256i *)&V[pos]);
            __m128i sd = _mm_loadu_si128((const __m128i *)&D[pos]);

            for (int i = g->k - 9; i > pos; i -= 8)
            {
                _mm256_storeu_si256((__m256i *)&V[i + 1], _mm256_loadu_si256((const __m256i *)&V[i]));
                _mm_storeu_si128((__m128i *)&D[i + 1], _mm_loadu_si128((const __m128i *)&D[i]));
            }

            _mm256_storeu_si256((__m256i *)&V[pos + 1], sv);
            _mm_storeu_si128((__m128i *)&D[pos + 1], sd);
        }
        else
        {
            for (int i = g->k - 1; i > pos; i--)
            {
                V[i] = V[i - 1];
                D[i] = D[i - 1];
            }
        }

        // Insert
        V[pos] = v;
        D[pos] = (int16_t)d;

        g->Dmax[u] = D[g->k - 1];
    }
}

static inline void KNN_insert_safe(KNN *g, int u, int v, int d)
{
    knn_lock(&g->L[u & 8191]);
    KNN_insert_unsafe(g, u, v, d);
    knn_unlock(&g->L[u & 8191]);
}
