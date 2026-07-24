#include "knn.h"

#include "algorithms.h"
#include "matrix.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

void *knn_alloc_huge(size_t bytes)
{
    // Round up so the whole range can be backed by 2 MB pages
    size_t rounded = (bytes + (2u << 20) - 1) & ~(size_t)((2u << 20) - 1);
    void *p = aligned_alloc(2u << 20, rounded);
    if (p == NULL)
    {
        fprintf(stderr, "Failed to allocate %zu bytes\n", bytes);
        exit(1);
    }
    // Everything past the code matrix is touched at random, and 4 KB pages make
    // that a TLB walk per access. THP is on "madvise" here, so ask for it.
    madvise(p, rounded, MADV_HUGEPAGE);
    return p;
}

KNN *KNN_init(int n, int k, float ref)
{
    KNN *g = malloc(sizeof(KNN));

    // Pad k to multiple of 8 for AVX2 alignment
    g->k = (k + 7) & ~7;
    g->n = n;

    // KNN_offer_max0 packs a slot index into KNN_TAG_BITS low bits of a distance
    if (g->k > (1 << KNN_TAG_BITS))
    {
        fprintf(stderr, "k=%d exceeds the %d slots addressable by KNN_TAG_BITS\n", g->k, 1 << KNN_TAG_BITS);
        exit(1);
    }
    g->V = knn_alloc_huge(sizeof(int) * (size_t)n * g->k);
    g->D = knn_alloc_huge(sizeof(int) * (size_t)n * g->k);
    g->Dmax = knn_alloc_huge(sizeof(int) * (size_t)n);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
    {
        // D starts in the tagged form used by KNN_offer_max0 during indexing:
        // distance in the high bits, own slot index in the low bits.
        for (int j = 0; j < g->k; j++)
        {
            g->V[(size_t)i * g->k + j] = n;
            g->D[(size_t)i * g->k + j] = ((int)ref << KNN_TAG_BITS) | j;
        }
        g->D[(size_t)i * g->k] = (int)ref << KNN_TAG_BITS;
        g->Dmax[i] = (int)ref;
    }

    for (int i = 0; i < 8192; i++)
        g->L[i].flag = 0;

    return g;
}

void KNN_free(KNN *g)
{
    free(g->V);
    free(g->D);
    free(g->Dmax);
    free(g);
}

int KNN_validate(KNN *g)
{
    int error = 0;

#pragma omp parallel for reduction(+ : error)
    for (int i = 0; i < g->n; i++)
    {
        for (int j = 0; j < g->k; j++)
        {
            int u = g->V[(size_t)i * g->k + j];
            if (u == g->n)
                continue;

            for (int k_idx = j + 1; k_idx < g->k; k_idx++)
            {
                int v = g->V[(size_t)i * g->k + k_idx];
                if (u == v)
                    error = 1;
            }
        }
    }
    return error == 0;
}

/*
    Indexing leaves the neighbourhoods unsorted (maximum pinned to slot 0).
    Refinement wants them ordered by distance, so restore that once here.
*/
void KNN_sort_lists(KNN *g)
{
    const int k = g->k;

#pragma omp parallel for schedule(static)
    for (int u = 0; u < g->n; u++)
    {
        int *V = g->V + (size_t)u * k;
        int *D = g->D + (size_t)u * k;

        // Drop the slot tags; from here on D holds plain distances.
        for (int i = 0; i < k; i++)
            D[i] >>= KNN_TAG_BITS;

        for (int i = 1; i < k; i++)
        {
            int dv = D[i], vv = V[i];
            int j = i - 1;
            while (j >= 0 && D[j] > dv)
            {
                D[j + 1] = D[j];
                V[j + 1] = V[j];
                j--;
            }
            D[j + 1] = dv;
            V[j + 1] = vv;
        }

        g->Dmax[u] = D[k - 1];
    }
}

#define PF_DIST 8

/*
    Neighbour-of-neighbour scan. Every candidate w costs two random lines out of
    the 813 MB code matrix plus one out of Dmax, so the loop is latency bound
    rather than ALU bound: the ids are all known up front, and running the loads
    PF_DIST iterations ahead is what keeps the memory pipeline full.
*/
void KNN_greedy_dist1_update_simple(KNN *g, int u, const uint64_t *Matrix, int m, int n, uint64_t *seen)
{
    const int k = g->k;
    const uint64_t *curr = Matrix + (size_t)u * n;
    const int *Vu = g->V + (size_t)u * k;
    int *Dmax = g->Dmax;
    int D = k / 2;

    for (int i = D; i >= 0; i--)
    {
        int v = Vu[i];
        const int *Vv = g->V + (size_t)v * k;
        const int target = D - i + 4;

        // The next v's neighbour list is itself a cold random read, and nothing
        // in this iteration can be prefetched until it lands. Start it early.
        if (i > 0)
        {
            const char *nx = (const char *)(g->V + (size_t)Vu[i - 1] * k);
            _mm_prefetch(nx, _MM_HINT_T0);
            _mm_prefetch(nx + 64, _MM_HINT_T0);
        }

        for (int p = 0; p < PF_DIST && p < target; p++)
        {
            int w = Vv[p];
            _mm_prefetch((const char *)(Matrix + (size_t)w * n), _MM_HINT_T0);
            _mm_prefetch((const char *)(Matrix + (size_t)w * n + 8), _MM_HINT_T0);
            _mm_prefetch((const char *)(Dmax + w), _MM_HINT_T0);
        }

        for (int j = 0; j < target; j++)
        {
            int w = Vv[j];

            if (j + PF_DIST < target)
            {
                int wp = Vv[j + PF_DIST];
                _mm_prefetch((const char *)(Matrix + (size_t)wp * n), _MM_HINT_T0);
                _mm_prefetch((const char *)(Matrix + (size_t)wp * n + 8), _MM_HINT_T0);
                _mm_prefetch((const char *)(Dmax + wp), _MM_HINT_T0);
            }

            if (w == u || w >= g->n || knn_seen_probe(seen, u, w))
                continue;

            int d = hamming_avx2_lut(curr, Matrix + (size_t)w * n, n);

            if (d < g->Dmax[u])
                KNN_insert_safe(g, u, w, d);
            if (d < Dmax[w])
                KNN_insert_safe(g, w, u, d);
        }
        while (i <= D && v != Vu[i])
            i++;
    }
}

void KNN_greedy_dist1_update_simple_round2(KNN *g, int u, const uint64_t *Matrix, int m, int n, uint64_t *seen)
{
    const int k = g->k;
    const uint64_t *curr = Matrix + (size_t)u * n;
    const int *Vu = g->V + (size_t)u * k;
    int D = k / 2;

    for (int i = D; i >= 0; i--)
    {
        int v = Vu[i];
        const int *Vv = g->V + (size_t)v * k;
        const int target = D - i;

        for (int p = 0; p <= PF_DIST && p <= target; p++)
        {
            int w = Vv[target - p];
            _mm_prefetch((const char *)(Matrix + (size_t)w * n), _MM_HINT_T0);
            _mm_prefetch((const char *)(Matrix + (size_t)w * n + 8), _MM_HINT_T0);
        }

        for (int j = target; j >= 0; j--)
        {
            int w = Vv[j];

            if (j - PF_DIST >= 0)
            {
                int wp = Vv[j - PF_DIST];
                _mm_prefetch((const char *)(Matrix + (size_t)wp * n), _MM_HINT_T0);
                _mm_prefetch((const char *)(Matrix + (size_t)wp * n + 8), _MM_HINT_T0);
            }

            if (w == u || w >= g->n || knn_seen_probe(seen, u, w))
                continue;

            int d = hamming_avx2_lut(curr, Matrix + (size_t)w * n, n);

            if (d < g->Dmax[u])
                KNN_insert_unsafe(g, u, w, d);
        }
        while (i <= D && v != Vu[i])
            i++;
    }
}

float KNN_compute_recall(KNN *g, const int *KNNs, int m, int n, int k)
{
    int tp = 0;

    assert(g->n == m);

#pragma omp parallel for reduction(+ : tp)
    for (int i = 0; i < g->n; i++)
    {
        for (int j = 0; j < g->k; j++)
        {
            int u = g->V[(size_t)i * g->k + j];
            if (u == i)
                continue;

            for (int l = 0; l < k + 1; l++) // Using k + 1 since result contain self loops
            {
                int v = KNNs[(size_t)i * n + l] - 1;
                if (u == v)
                {
                    tp++;
                    break;
                }
            }
        }
    }
    return (double)tp / (double)(m * k);
}

float KNN_compute_recall_standalone(int *ANNs, const int *KNNs, int m, int n, int k)
{
    int tp = 0;

#pragma omp parallel for reduction(+ : tp)
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            int u = ANNs[(size_t)i * k + j] - 1;
            if (u == i)
                continue;

            for (int l = 0; l < k + 1; l++) // Using k + 1 since result contain self loops
            {
                int v = KNNs[(size_t)i * n + l] - 1;
                if (u == v)
                {
                    tp++;
                    break;
                }
            }
        }
    }
    return (double)tp / (double)(m * k);
}
