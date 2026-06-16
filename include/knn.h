#pragma once

#include "algorithms.h"
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <string.h>

typedef struct
{
    omp_lock_t lock;
    char padding[64 - sizeof(omp_lock_t)];
} PaddedLock;

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
    int *D;             // List of neighbor distances (SOA)
    PaddedLock L[8192]; // Locks for mutual exclusion
} KNN;

KNN *KNN_init(int n, int k, float ref);

void KNN_free(KNN *g);

int KNN_validate(KNN *g);

void KNN_greedy_dist1_update_simple(KNN *g, int u, const uint64_t *Matrix, int m, int n);

void KNN_greedy_dist1_update_simple_round2(KNN *g, int u, const uint64_t *Matrix, int m, int n);

float KNN_compute_recall(KNN *g, const int *KNNs, int m, int n, int k);

float KNN_compute_recall_standalone(int *ANNs, const int *KNNs, int m, int n, int k);

static inline void KNN_insert_unsafe(KNN *g, int u, int v, int d)
{
    int *V = g->V + (u * g->k);
    int *D = g->D + (u * g->k);

    // Stop if d is worse than the worst neighbor
    if (D[g->k - 1] <= d)
        return;

    // Duplicate check using AVX2
    __m256i target_v = _mm256_set1_epi32(v);
    for (int i = 0; i < g->k; i += 8)
    {
        __m256i chunk_v = _mm256_load_si256((const __m256i *)&V[i]);
        __m256i cmp = _mm256_cmpeq_epi32(chunk_v, target_v);
        if (_mm256_movemask_epi8(cmp) != 0)
            return;
    }

    // Find insertion position using AVX2
    int pos = g->k;
    __m256i target_d = _mm256_set1_epi32(d);
    for (int i = 0; i < g->k; i += 8)
    {
        __m256i chunk_d = _mm256_load_si256((const __m256i *)&D[i]);
        // Find first element > d. Since D is sorted ascending, we look for first D[i] > d.
        __m256i cmp = _mm256_cmpgt_epi32(chunk_d, target_d);
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
        if (mask != 0)
        {
            pos = i + __builtin_ctz(mask);
            break;
        }
    }

    if (pos < g->k)
    {
        // Shift elements

        if (g->k - pos - 1 > 32)
        {
            memmove(&V[pos + 1], &V[pos], (g->k - pos - 1) * sizeof(int));
            memmove(&D[pos + 1], &D[pos], (g->k - pos - 1) * sizeof(int));
        }
        else
        {
            for (int i = pos + 1; i < g->k; i++)
                V[pos] = V[pos - 1];

            for (int i = pos + 1; i < g->k; i++)
                D[pos] = D[pos - 1];
        }

        // Insert
        V[pos] = v;
        D[pos] = d;
    }
}

static inline void KNN_insert_safe(KNN *g, int u, int v, int d)
{
    omp_set_lock(&g->L[u & 8191].lock);
    KNN_insert_unsafe(g, u, v, d);
    omp_unset_lock(&g->L[u & 8191].lock);
}