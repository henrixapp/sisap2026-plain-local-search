#pragma once

#include <omp.h>
#include <stdint.h>

typedef struct
{
    omp_lock_t lock;
    char padding[64 - sizeof(omp_lock_t)];
} PaddedLock;

typedef struct
{
    int v;
    int d;
} VDEntry;

typedef struct KNN_struct
{
    int n;              // Number of vertices
    int k;              // Size of neighborhoods
    VDEntry *VD;        // List of neighbors (n if empty, ascending in d)
    PaddedLock L[8192]; // Locks for mutual exclusion
} KNN;

KNN *KNN_init(int n, int k, float ref);

void KNN_free(KNN *g);

int KNN_validate(KNN *g);

void KNN_greedy_dist1_update_simple(KNN *g, int u, const uint64_t *Matrix, int m, int n);

void KNN_greedy_dist1_update_simple_round2(KNN *g, int u, const uint64_t *Matrix, int m, int n);

float KNN_compute_recall(KNN *g, const int *KNNs, int m, int n, int k);

static inline void KNN_insert_safe(KNN *g, int u, int v, int d)
{
    omp_set_lock(&g->L[u & 8191].lock);

    VDEntry *N = g->VD + (u * g->k);

    // Stop if d is worse than the worst neighbor
    if (N[g->k - 1].d <= d)
        goto end;

    int pos = -1;
    for (int i = 0; i < g->k; i++)
    {
        if (N[i].v == v)
            goto end;

        if (N[i].d <= d)
            pos = i;
    }
    pos++;

    for (int i = g->k - 1; i > pos; i--)
        N[i] = N[i - 1];

    // Insert
    N[pos].v = v;
    N[pos].d = d;

end:
    omp_unset_lock(&g->L[u & 8191].lock);
}

static inline void KNN_insert_unsafe(KNN *g, int u, int v, int d)
{
    VDEntry *N = g->VD + (u * g->k);

    // Stop if d is worse than the worst neighbor
    if (N[g->k - 1].d <= d)
        return;

    int pos = -1;
    for (int i = 0; i < g->k; i++)
    {
        if (N[i].v == v)
            return;

        if (N[i].d <= d)
            pos = i;
    }
    pos++;

    for (int i = g->k - 1; i > pos; i--)
        N[i] = N[i - 1];

    // Insert
    N[pos].v = v;
    N[pos].d = d;
}