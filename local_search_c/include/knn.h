#pragma once
#include <stdint.h>
#include "algorithms.h"
#include <stdint.h>
#include <omp.h>

typedef struct KNN_struct
{
    int n;         // Number of vertices
    int k;         // Size of neighborhoods
    int *V;        // List of neighbors (n if empty)
    int16_t *D;    // Sorted dot products between vertices (descending)
    omp_lock_t *L; // Locks for mutual exclusion
} KNN;

KNN *KNN_init(int n, int k, float ref);

void KNN_free(KNN *g);

int KNN_validate(KNN *g);

static inline int KNN_contains_neighbor(KNN *g, int u, int v)
{
    for (int i = 0; i < g->k; i++)
        if (g->V[u * g->k + i] == v)
            return 1;
    return 0;
}

static inline void KNN_insert_safe_min(KNN *g, int u, int v, int16_t d)
{
    int pos = (u * g->k) + (g->k - 1);

    if ( g->D[pos] <= d  )
    {
        return;
    }
    omp_set_lock(&g->L[u]);

    if(KNN_contains_neighbor(g, u, v)){
    omp_unset_lock(&g->L[u]);
        
        return;}

    // Finding insertion point (linear scan back)
    while (pos > (u * g->k) && g->D[pos - 1] > d)
    {
        g->D[pos] = g->D[pos - 1];
        g->V[pos] = g->V[pos - 1];
        pos--;
    }

    // Insert
    g->D[pos] = d;
    g->V[pos] = v;

    omp_unset_lock(&g->L[u]);
}

void KNN_greedy_dist1_update(KNN *g, int u, const uint64_t *Matrix, int m, int n);

float KNN_compute_recall(KNN *g, const int *KNNs, int m, int n, int k);
