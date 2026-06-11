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

static inline void KNN_insert_safe(KNN *g, int u, int v, int d)
{
    int pos = (u * LOCAL_SEARCH_NEIGHBORHOOD_SIZE) + (LOCAL_SEARCH_NEIGHBORHOOD_SIZE - 1);

    // Stop if d is worse than the worst neighbor
    if (u == v || g->D[pos] <= d)
        return;
    omp_set_lock(&g->L[u]);
    const int next_one = (u + 1) * LOCAL_SEARCH_NEIGHBORHOOD_SIZE;
    for (int i = u * LOCAL_SEARCH_NEIGHBORHOOD_SIZE; i < next_one; i++)
        if (g->V[i] == v)
            goto end;

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

end:
    omp_unset_lock(&g->L[u]);
}

void KNN_greedy_dist1_update_simple(KNN *g, int u, const uint64_t *Matrix, int m, int n);

float KNN_compute_recall(KNN *g, const int *KNNs, int m, int n, int k);