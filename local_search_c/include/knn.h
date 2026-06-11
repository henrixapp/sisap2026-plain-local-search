#pragma once

#include "constants.h"
#include <omp.h>
#include <stdint.h>
typedef struct KNN_struct
{
    int n;         // Number of vertices
    int k;         // Size of neighborhoods
    int *V;        // List of neighbors (n if empty)
    int *D;        // Sorted hamming distances (descending)
    omp_lock_t *L; // Locks for mutual exclusion
} KNN;

KNN *KNN_init(int n, int k, float ref);

void KNN_free(KNN *g);

int KNN_validate(KNN *g);
static inline void swapper(KNN *g, int pos_a, int pos_b)
{
    int D_temp = g->D[pos_a];
    g->D[pos_a] = g->D[pos_b];
    g->D[pos_b] = D_temp;
    int V_temp = g->V[pos_a];
    g->V[pos_a] = g->V[pos_b];
    g->V[pos_b] = V_temp;
}
static inline void KNN_insert_safe(KNN *g, int u, int v, int d)
{
#ifdef LOCAL_SEARCH_HEAPIFY
    int pos = (u * LOCAL_SEARCH_NEIGHBORHOOD_SIZE); // Implementing max heap
#else
    int pos = (u * LOCAL_SEARCH_NEIGHBORHOOD_SIZE) + (LOCAL_SEARCH_NEIGHBORHOOD_SIZE - 1);
#endif
    // Stop if d is worse than the worst neighbor
    if (u == v || g->D[pos] <= d)
        return;
    omp_set_lock(&g->L[u]);
    const int next_one = (u + 1) * LOCAL_SEARCH_NEIGHBORHOOD_SIZE;
    for (int i = u * LOCAL_SEARCH_NEIGHBORHOOD_SIZE; i < next_one; i++)
        if (g->V[i] == v)
            goto end;
#ifdef LOCAL_SEARCH_HEAPIFY
    int heapPos = 0;
    // Insert- override max value
    g->D[pos] = d;
    g->V[pos] = v;

    // Bubble down
    while (2 * heapPos + 2 < LOCAL_SEARCH_NEIGHBORHOOD_SIZE)
    {
        int left = 2 * heapPos + 1;
        int right = 2 * heapPos + 2;
        int better = (g->D[pos + left] > g->D[pos + right]) * left + (g->D[pos + left] <= g->D[pos + right]) * right;

        // left larger
        if (g->D[pos + better] > d)
        {
            // have to swap
            swapper(g, pos + better, pos + heapPos);
            heapPos = better;
            continue;
        }
        else
        {
            break;
        }
    }
#else
    // Finding insertion point (linear scan back)
    while (pos > (u * LOCAL_SEARCH_NEIGHBORHOOD_SIZE) && g->D[pos - 1] > d)
    {
        g->D[pos] = g->D[pos - 1];
        g->V[pos] = g->V[pos - 1];
        pos--;
    }

    // Insert
    g->D[pos] = d;
    g->V[pos] = v;
#endif
end:
    omp_unset_lock(&g->L[u]);
}

void KNN_greedy_dist1_update_simple(KNN *g, int u, const uint64_t *Matrix, int m, int n);

float KNN_compute_recall(KNN *g, const int *KNNs, int m, int n, int k);