#include "knn.h"

#include "algorithms.h"
#include "matrix.h"

#include <assert.h>
#include <stdlib.h>

KNN *KNN_init(int n, float ref)
{
    KNN *g = malloc(sizeof(KNN));

    g->n = n;
    g->VD = malloc(sizeof(VDentry) * n * LOCAL_SEARCH_SIZE_OF_NEIGHBORS);
    // g->D = malloc(sizeof(int16_t) * n * LOCAL_SEARCH_SIZE_OF_NEIGHBORS);
    g->L = malloc(sizeof(omp_lock_t) * n);
    for (int i = 0; i < n * LOCAL_SEARCH_SIZE_OF_NEIGHBORS; i++)
    {
        g->VD[i].v = 2 * n;
        g->VD[i].d = 1024;
    }

    for (int i = 0; i < n; i++)
    {
        omp_init_lock(&g->L[i]);
    }

    return g;
}

void KNN_free(KNN *g)
{
    free(g->VD);
    // free(g->D);

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
        for (int j = 0; j < LOCAL_SEARCH_SIZE_OF_NEIGHBORS; j++)
        {
            int u = g->VD[i * LOCAL_SEARCH_SIZE_OF_NEIGHBORS + j].v;
            if (u == g->n)
                continue;

            for (int k = j + 1; k < LOCAL_SEARCH_SIZE_OF_NEIGHBORS; k++)
            {
                int v = g->VD[i * LOCAL_SEARCH_SIZE_OF_NEIGHBORS + k].v;
                int d = g->VD[i * LOCAL_SEARCH_SIZE_OF_NEIGHBORS + k].d;
                if (u == v)
                {
                    error = 1;
                    printf("%d %d %d %d %d\n", i, u, v, k, d);
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
        for (int i = 1; i <= LOCAL_SEARCH_SIZE_OF_NEIGHBORS; i++)
        {

            int v = g->VD[(u + 1) * LOCAL_SEARCH_SIZE_OF_NEIGHBORS - i].v;
            if (v >= g->n)
            {
                //             omp_set_lock(&g->L[u]);
                //             if( v ==g->V[(u+1) * LOCAL_SEARCH_SIZE_OF_NEIGHBORS - i]) {
                //                 g->V[(u+1) * LOCAL_SEARCH_SIZE_OF_NEIGHBORS - i] -=g->n;
                //             }
                //   omp_unset_lock(&g->L[u]);
                continue;
            }
            const int v1 = (v)*LOCAL_SEARCH_SIZE_OF_NEIGHBORS;
            int brought_change = 0;
            for (int j = 0; j < LOCAL_SEARCH_SIZE_OF_NEIGHBORS; j++)
            {
                int w = g->VD[v1 + j].v;
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
                // if (d < g->D[(u * LOCAL_SEARCH_SIZE_OF_NEIGHBORS) + (LOCAL_SEARCH_SIZE_OF_NEIGHBORS - 1)])
                brought_change += KNN_insert_safe_min(g, u, w, d);
                // if (d < g->D[(w * LOCAL_SEARCH_SIZE_OF_NEIGHBORS) + (LOCAL_SEARCH_SIZE_OF_NEIGHBORS - 1)])
                brought_change += KNN_insert_safe_min(g, w, u, d);
            }
            b_change += brought_change;
            if (brought_change == 0)
            {
                omp_set_lock(&g->L[u]);
                if (v == g->VD[(u + 1) * LOCAL_SEARCH_SIZE_OF_NEIGHBORS - i].v)
                {
                    g->VD[(u + 1) * LOCAL_SEARCH_SIZE_OF_NEIGHBORS - i].v += g->n;
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
        for (int j = 0; j < LOCAL_SEARCH_SIZE_OF_NEIGHBORS; j++)
        {
            int u = g->VD[i * LOCAL_SEARCH_SIZE_OF_NEIGHBORS + j].v;

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