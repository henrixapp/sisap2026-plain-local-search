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
    g->VD = malloc(sizeof(VDEntry) * n * k);

    for (int i = 0; i < n * k; i++)
    {
        g->VD[i].v = n;
        g->VD[i].d = ref;
    }

    for (int i = 0; i < 8192; i++)
    {
        omp_init_lock(&g->L[i].lock);
    }

    return g;
}

void KNN_free(KNN *g)
{
    free(g->VD);

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
            int u = g->VD[i * g->k + j].v;
            if (u == g->n)
                continue;

            for (int k = j + 1; k < g->k; k++)
            {
                int v = g->VD[i * g->k + k].v;
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
    for (int i = g->k / 4; i >= 0; i--)
    {
        int v = g->VD[u * g->k + i].v;
        for (int j = g->k - 1; j >= 0; j--)
        {
            int w = g->VD[v * g->k + j].v;
            if (w == u || w >= g->n)
                continue;

            int d = hamming_avx2_lut(curr, Matrix + w * n, n);

            if (d < g->VD[(u * g->k) + (g->k - 1)].d)
                KNN_insert_safe(g, u, w, d);
            if (d < g->VD[(w * g->k) + (g->k - 1)].d)
                KNN_insert_safe(g, w, u, d);
        }
    }
}
void KNN_greedy_dist1_update_simple_round2(KNN *g, int u, const uint64_t *Matrix, int m, int n)
{
    const uint64_t *curr = Matrix + u * n;
    for (int i = g->k / 4; i >= 0; i--)
    {
        int v = g->VD[u * g->k + i].v;
        for (int j = g->k - 1; j >= 0; j--)
        {
            int w = g->VD[v * g->k + j].v;
            if (w == u)
                continue;

            int d = hamming_avx2_lut(curr, Matrix + w * n, n);

            if (d < g->VD[(u * g->k) + (g->k - 1)].d)
                KNN_insert_safe(g, u, w, d);
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
            int u = g->VD[i * g->k + j].v;
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
