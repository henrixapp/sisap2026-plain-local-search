#include "knn.h"

#include "algorithms.h"
#include "matrix.h"

#include <assert.h>
#include <stdlib.h>

KNN *KNN_init(int n, int k, float ref)
{
    KNN *g = malloc(sizeof(KNN));

    g->k = k;
    g->n = n;
    g->V = malloc(sizeof(int) * n * k);
    g->D = malloc(sizeof(float) * n * k);
    g->L = malloc(sizeof(omp_lock_t) * n);

    for (int i = 0; i < n * k; i++)
    {
        g->V[i] = n;
        g->D[i] = ref;
    }

    for (int i = 0; i < n; i++)
    {
        omp_init_lock(&g->L[i]);
    }

    return g;
}

void KNN_free(KNN *g)
{
    free(g->V);
    free(g->D);

    for (int i = 0; i < g->n; i++)
    {
        omp_destroy_lock(&g->L[i]);
    }

    free(g->L);

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

            for (int k = j + 1; k < g->k; k++)
            {
                int v = g->V[i * g->k + k];
                if (u == v)
                    error = 1;
            }
        }
    }
    return error == 0;
}

void KNN_greedy_dist1_update_simple(KNN *g, int u, const uint64_t *Matrix, int m, int n)
{
    for (int i = g->k / 4; i >= 0; i--)
    {
        int v = g->V[u * g->k + i];
        for (int j = g->k - 1; j >= 0; j--)
        {
            int w = g->V[v * g->k + j];

            int d = hamming_portable(Matrix + u * n, Matrix + w * n, n);
            KNN_insert_safe(g, u, w, d);
            KNN_insert_safe(g, w, u, d);
        }
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
