#pragma once

#include "algorithms.h"
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <string.h>

// Number of shards used for fine grained mutual exclusion. Must be a power of
// two. Each lock sits alone in a cache line: an atomic RMW takes the line
// exclusively, so packing several locks per line would make unrelated vertices
// ping-pong lines between cores.
#define KNN_NLOCKS 8192
#define KNN_LOCK_STRIDE 16

typedef struct
{
    int v;
    int d;
} VDEntry;

/*
    Neighborhood store.

    V holds the neighbor ids and D the matching (Hamming) distances, both as
    n * k row major blocks kept sorted by increasing distance.

    W is a compact copy of the worst distance of every row, i.e. W[u] equals
    D[u * k + k - 1]. Every candidate that the search produces is first tested
    against this bound and the overwhelming majority is rejected right there.
    Reading the bound from a dedicated n element array instead of from the far
    larger D block keeps that test inside L2 and is what makes the rejection
    path cheap.
*/
typedef struct KNN_struct
{
    int n;                        // Number of vertices
    int k;                        // Size of neighborhoods (multiple of 16)
    int *V;      // Neighbor ids
    uint16_t *D; // Neighbor distances
    uint16_t *W; // Per vertex worst distance (bound cache)
    volatile int L[KNN_NLOCKS * KNN_LOCK_STRIDE]; // Spin locks, one per line
} KNN;

KNN *KNN_init(int n, int k, int ref);

void KNN_free(KNN *g);

int KNN_validate(KNN *g);

float KNN_compute_recall(KNN *g, const int *KNNs, int m, int n, int k);

float KNN_compute_recall_standalone(int *ANNs, const int *KNNs, int m, int n, int k);

// Lightweight test-and-set lock. omp_set_lock is an out of line call into
// libgomp, which is far too expensive for the millions of short critical
// sections the build performs.
static inline void KNN_lock(KNN *g, int u)
{
    volatile int *l = &g->L[(u & (KNN_NLOCKS - 1)) * KNN_LOCK_STRIDE];
    while (__atomic_exchange_n(l, 1, __ATOMIC_ACQUIRE))
        while (__atomic_load_n(l, __ATOMIC_RELAXED))
            _mm_pause();
}

static inline void KNN_unlock(KNN *g, int u)
{
    __atomic_store_n(&g->L[(u & (KNN_NLOCKS - 1)) * KNN_LOCK_STRIDE], 0, __ATOMIC_RELEASE);
}

static inline void KNN_insert_unsafe(KNN *g, int u, int v, int d)
{
    const int k = g->k;
    int *V = g->V + ((size_t)u * k);
    uint16_t *D = g->D + ((size_t)u * k);

    // Stop if d is worse than the worst neighbor
    if (D[k - 1] <= d)
        return;

    __m256i target_v = _mm256_set1_epi32(v);
    __m256i target_d = _mm256_set1_epi16((short)d);

    int pos = k;

    // Walk the row in blocks of 16. D is sorted ascending, so the first entry
    // with D > d is the insertion point. A duplicate of v necessarily carries
    // the exact same distance and therefore sits at or before that point,
    // which is why testing for it block by block is sufficient.
    for (int i = 0; i < k; i += 16)
    {
        __m256i cv0 = _mm256_load_si256((const __m256i *)&V[i]);
        __m256i cv1 = _mm256_load_si256((const __m256i *)&V[i + 8]);
        __m256i eq = _mm256_or_si256(_mm256_cmpeq_epi32(cv0, target_v),
                                     _mm256_cmpeq_epi32(cv1, target_v));
        if (_mm256_movemask_epi8(eq) != 0)
            return;

        __m256i cd = _mm256_load_si256((const __m256i *)&D[i]);
        int mask = _mm256_movemask_epi8(_mm256_cmpgt_epi16(cd, target_d));
        if (mask != 0)
        {
            pos = i + (__builtin_ctz(mask) >> 1);
            break;
        }
    }

    if (pos < k)
    {
        memmove(&V[pos + 1], &V[pos], (k - pos - 1) * sizeof(int));
        memmove(&D[pos + 1], &D[pos], (k - pos - 1) * sizeof(uint16_t));

        V[pos] = v;
        D[pos] = (uint16_t)d;

        g->W[u] = D[k - 1];
    }
}

static inline void KNN_insert_safe(KNN *g, int u, int v, int d)
{
    KNN_lock(g, u);
    KNN_insert_unsafe(g, u, v, d);
    KNN_unlock(g, u);
}
