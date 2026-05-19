#include "knn.h"

#include "algorithms.h"
#include "matrix.h"

#include <assert.h>

#include <stdint.h>
#include <stdlib.h>
KNN *KNN_init(int n, int k, float ref)
{
    KNN *g = malloc(sizeof(KNN));

    g->k = k;
    g->n = n;
    g->V = malloc(sizeof(int) * n * k);
    g->gen = malloc(sizeof(int) * n);
    g->D = malloc(sizeof(int16_t) * n * k);
    // g->Dmin = malloc(sizeof(int16_t) * n);

    g->L = malloc(sizeof(omp_lock_t) * LOCKS_COUNT);
    for (int i = 0; i < n * k; i++)
    {
        g->V[i] = 2 * n;
        g->D[i] = ref;
        g->gen[i] = 0;
    }

    for (int i = 0; i < LOCKS_COUNT; i++)
    {
        // g->Dmin[i] = INT16_MAX;
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
    // free(g->Dmin);
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

void KNN_greedy_dist1_update(KNN *g, int u, const uint64_t *restrict Matrix, int m, int n)
{
    int jk = 0;
    int b_change = 1;
    int my_gen = 1;
    const uint64_t *restrict curr = Matrix + n * u;
    while (b_change > 0)
    {
        my_gen++;
        my_gen %= 40;
        b_change = 0;
        for (int i = 1; i <= g->k; i++)
        {

            int v = g->V[(u + 1) * g->k - i];
            if (v >= g->n) //&& my_gen < g->gen[v - g->n]
            {
                // omp_set_lock(&g->L[u]);
                // if (v == g->V[(u + 1) * g->k - i])
                // {
                //     g->V[(u + 1) * g->k - i] -= g->n;
                // }
                // omp_unset_lock(&g->L[u]);
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
                int d = hamming_portable(curr, Matrix + w * n, n);
                // if (d < g->D[(u * g->k) + (g->k - 1)])
                brought_change += KNN_insert_safe_min(g, u, w, d);
                // if (d < g->D[(w * g->k) + (g->k - 1)])
                brought_change += KNN_insert_safe_min(g, w, u, d);
            }
            b_change += brought_change;
            if (brought_change == 0)
            {
                omp_set_lock(&g->L[u % LOCKS_COUNT]);
                if (v == g->V[(u + 1) * g->k - i])
                {
                    g->V[(u + 1) * g->k - i] += g->n;
                }
                omp_unset_lock(&g->L[u % LOCKS_COUNT]);
            }
        }
    }
}
void KNN_greedy_dist1_update_fast(KNN *g, int u, const uint64_t *restrict Matrix, int m, int n, int smaller_k)
{
    int jk = 0;
    int b_change = 1;
    const uint64_t *curr = Matrix + n * u;
    while (b_change > 0)
    {

        b_change = 0;
        for (int i = 1; i <= smaller_k; i++)
        {

            int v = g->V[(u + 1) * g->k - i];
            if (v >= g->n) //&& my_gen < g->gen[v - g->n]
            {
                // omp_set_lock(&g->L[u]);
                // if (v == g->V[(u + 1) * g->k - i])
                // {
                //     g->V[(u + 1) * g->k - i] -= g->n;
                // }
                // omp_unset_lock(&g->L[u]);
                continue;
            }
            const int v1 = (v + 1) * g->k;
            int brought_change = 0;
            for (int j = 1; j <= smaller_k; j++)
            {
                int w = g->V[v1 - j];
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
                int d = hamming_portable(curr, Matrix + w * n, n);
                // if (d < g->D[(u * g->k) + (g->k - 1)])
                brought_change += KNN_insert_safe_min(g, u, w, d);
                // if (d < g->D[(w * g->k) + (g->k - 1)])
                brought_change += KNN_insert_safe_min(g, w, u, d);
            }
            b_change += brought_change;
            if (brought_change == 0)
            {
                omp_set_lock(&g->L[u % LOCKS_COUNT]);
                if (v == g->V[(u + 1) * g->k - i])
                {
                    g->V[(u + 1) * g->k - i] += g->n;
                }
                omp_unset_lock(&g->L[u % LOCKS_COUNT]);
            }
        }
    }
}
int KNN_random_insert(KNN *g, int u2, const uint64_t *restrict Matrix, int m, int n)
{
    int b_change = 0;

    for (int z = 0; z < g->k * g->k; z++)
    {
        int w = rand() % g->n;
        int u = rand() % g->n;
        int d = hamming_portable(Matrix + u * n, Matrix + w * n, n);
        // if (d < g->D[(u * g->k) + (g->k - 1)])
        b_change += KNN_insert_safe_min(g, u, w, d);
        b_change += KNN_insert_safe_min(g, w, u, d);
    }
    for (int i = 0; i < g->k; i++)
    {
        int u = rand() % g->n;

        omp_set_lock(&g->L[u % LOCKS_COUNT]);
        for (int z = u * g->k; z < (u + 1) * g->k; z++)
            if (g->V[z] >= g->n)
            {
                g->V[z] -= g->n;
            }
        omp_unset_lock(&g->L[u % LOCKS_COUNT]);
    }
    return b_change;
}
int KNN_greedy_further_dist1_update(KNN *g, int u, const uint64_t *restrict Matrix, int m,
                                    int n)
{
    int b_change = 1;
    int jk = 0;
    int sqrt_K = sqrt(g->k);
    while (b_change > 0)
    {
        b_change = 0;
        for (int z2 = 1; z2 < sqrt_K; z2++)
        {
            int u2 = g->V[(u)*g->k + z2];
            if (u2 >= g->n)
            {
                u2 -= g->n;
            }
            for (int z = 1; z < sqrt_K; z++)
            {
                int u1 = g->V[(u2 + 1) * g->k - z];
                if (u1 >= g->n)
                {
                    u1 -= g->n;
                }
                for (int i = 1; i <= sqrt_K; i++)
                {

                    int v = g->V[(u1 + 1) * g->k - i];
                    if (v >= g->n) //&& my_gen < g->gen[v - g->n]
                    {
                        // omp_set_lock(&g->L[u]);
                        // if (v == g->V[(u + 1) * g->k - i])
                        // {
                        //     g->V[(u + 1) * g->k - i] -= g->n;
                        // }
                        // omp_unset_lock(&g->L[u]);
                        continue;
                    }
                    const int v1 = (v)*g->k;
                    int brought_change = 0;
                    for (int j = 0; j < sqrt_K; j++)
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
                        // omp_set_lock(&g->L[u]);
                        // if (v == g->V[(u + 1) * g->k - i])
                        // {
                        //     g->V[(u + 1) * g->k - i] += g->n;
                        // }
                        // omp_unset_lock(&g->L[u]);
                    }
                }
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
                if (u == v || ((u - g->n) == v))
                {
                    tp++;
                    break;
                }
            }
        }
    }
    // int16_t ma = INT16_MAX;
    // for (int i = 0; i < g->n; i++)
    // {
    //     if (g->Dmin[i] < ma)
    //     {
    //         ma = g->Dmin[i];
    //     }
    // }
    // printf("%d \n", ma);
    return (double)tp / (double)(m * k);
}