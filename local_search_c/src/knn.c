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
    g->D = malloc(sizeof(int16_t) * n * k);
    g->L = malloc(sizeof(omp_lock_t) * n);
    for (int i = 0; i < n * k; i++)
    {
        g->V[i] = 2 * n;
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
                {
                    error = 1;
                    printf("%d %d %d %d\n", i, u, v, k);
                }
            }
        }
    }
    return error == 0;
}

void KNN_greedy_dist1_update(KNN *g, int u, const uint64_t *Matrix, int m, int n)
{
    int jk = 0;
    int b_change = 1;
    while (b_change > 0)
    {
        b_change = 0;
        for (int i = 1; i <= g->k; i++)
        {

            int v = g->V[(u + 1) * g->k - i];
            if (v >= g->n)
            {
                //             omp_set_lock(&g->L[u]);
                //             if( v ==g->V[(u+1) * g->k - i]) {
                //                 g->V[(u+1) * g->k - i] -=g->n;
                //             }
                //   omp_unset_lock(&g->L[u]);
                continue;
            }
            const int v1 = (v)*g->k;
            int brought_change = 0;
            for (int j = 0; j < g->k; j++)
            {
                int w = g->V[v1 + j];
                if (u == w || w >= 2 * g->n)
                {
                    continue;
                }
                // if (KNN_contains_neighbor(g, u, w))
                //     continue;
                if (w >= g->n)
                {
                    w -= g->n;
                }
                int d = hamming_portable(Matrix + u * n, Matrix + w * n, n);
                // if (d < g->D[(u * g->k) + (g->k - 1)])
                brought_change += KNN_insert_safe_min(g, u, w, d);
                // if (d < g->D[(w * g->k) + (g->k - 1)])
                brought_change += KNN_insert_safe_min(g, w, u, d);
            }
            b_change += brought_change;
            if (brought_change == 0)
            {
                omp_set_lock(&g->L[u]);
                if (v == g->V[(u + 1) * g->k - i])
                {
                    g->V[(u + 1) * g->k - i] += g->n;
                }
                omp_unset_lock(&g->L[u]);
            }
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