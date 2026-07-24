#include "knn.h"

#include "algorithms.h"
#include "alloc.h"
#include "matrix.h"

#include <assert.h>
#include <stdlib.h>

__thread uint32_t thread_seed;

KNN *KNN_init(int n, int k, int ref)
{
    KNN *g = malloc(sizeof(KNN));

    // Pad k to a multiple of 16 so the insert scan can walk V (8 lanes) and
    // D (16 lanes) in lockstep without a tail.
    g->k = (k + 15) & ~15;
    g->n = n;
    g->V = alloc64(sizeof(int) * (size_t)n * g->k);
    g->D = alloc64(sizeof(uint16_t) * (size_t)n * g->k);
    g->W = alloc64(sizeof(uint16_t) * (size_t)n);

    if (ref > UINT16_MAX)
        ref = UINT16_MAX;

#pragma omp parallel for
    for (size_t i = 0; i < (size_t)n * g->k; i++)
    {
        g->V[i] = n;
        g->D[i] = (uint16_t)ref;
    }

#pragma omp parallel for
    for (int i = 0; i < n; i++)
        g->W[i] = (uint16_t)ref;

    for (int i = 0; i < KNN_NLOCKS * KNN_LOCK_STRIDE; i++)
        g->L[i] = 0;

    return g;
}

void KNN_free(KNN *g)
{
    free(g->V);
    free(g->D);
    free(g->W);
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

        if (g->W[i] != g->D[(size_t)i * g->k + g->k - 1])
            error = 1;
    }
    return error == 0;
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
    return (double)tp / (double)((size_t)m * k);
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
    return (double)tp / (double)((size_t)m * k);
}
