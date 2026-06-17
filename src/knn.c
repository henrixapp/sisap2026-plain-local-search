#include "knn.h"

#include "algorithms.h"
#include "matrix.h"

#include <assert.h>
#include <stdlib.h>

KNN *KNN_init(int n, int k, float ref)
{
    KNN *g = malloc(sizeof(KNN));

    // Pad k to multiple of 8 for AVX2 alignment
    g->k = (k + 7) & ~7;
    g->n = n;
    g->V = aligned_alloc(32, sizeof(int) * (size_t)n * g->k);
    g->D = aligned_alloc(32, sizeof(int) * (size_t)n * g->k);

    for (int i = 0; i < n * g->k; i++)
    {
        g->V[i] = n;
        g->D[i] = (int)ref;
    }

    for (int i = 0; i < 8192; i++)
    {
        omp_init_lock(&g->L[i].lock);
    }

    return g;
}

void KNN_free(KNN *g)
{
    free(g->V);
    free(g->D);

    for (int i = 0; i < 8192; i++)
    {
        omp_destroy_lock(&g->L[i].lock);
    }

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
            int u = g->V[i * g->k + j];
            if (u == g->n)
                continue;

            for (int k_idx = j + 1; k_idx < g->k; k_idx++)
            {
                int v = g->V[i * g->k + k_idx];
                if (u == v)
                    error = 1;
            }
        }
    }
    return error == 0;
}

void KNN_greedy_dist1_update_simple(KNN *g, int u, const uint64_t *Matrix, int m, int n)
{
    const uint64_t *curr = Matrix + u * n;
    int D = g->k / 2;
    for (int i = D; i >= 0; i--)
    {
        int v = g->V[u * g->k + i];
        for (int j = 0; j < D - i; j++)
        {
            int w = g->V[v * g->k + j];
            if (w == u || w >= g->n)
                continue;

            int d = hamming_avx2_lut(curr, Matrix + w * n, n);

            if (d < g->D[(u * g->k) + (g->k - 1)])
                KNN_insert_safe(g, u, w, d);
            if (d < g->D[(w * g->k) + (g->k - 1)])
                KNN_insert_safe(g, w, u, d);
        }
        while (i <= D && v != g->V[u * g->k + i])
            i++;
    }
}
void KNN_greedy_dist1_update_simple_round2(KNN *g, int u, const uint64_t *Matrix, int m, int n)
{
    const uint64_t *curr = Matrix + u * n;
    int D = g->k / 2;
    for (int i = D; i >= 0; i--)
    {
        int v = g->V[u * g->k + i];
        for (int j = 0; j < D - i; j++)
        {
            int w = g->V[v * g->k + j];
            if (w == u)
                continue;

            int d = hamming_avx2_lut(curr, Matrix + w * n, n);

            if (d < g->D[(u * g->k) + (g->k - 1)])
                KNN_insert_unsafe(g, u, w, d);
        }
        while (i <= D && v != g->V[u * g->k + i])
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
            int u = g->V[i * g->k + j];
            if (u == i)
                continue;

            for (int l = 0; l < k + 1; l++) // Using k + 1 since result contain self loops
            {
                int v = KNNs[i * n + l] - 1;
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
            int u = ANNs[i * k + j] - 1;
            if (u == i)
                continue;

            for (int l = 0; l < k + 1; l++) // Using k + 1 since result contain self loops
            {
                int v = KNNs[i * n + l] - 1;
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
