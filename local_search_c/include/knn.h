#pragma once
#include "algorithms.h"
#include <omp.h>
#include <stdint.h>

typedef struct KNN_struct
{
    int n;         // Number of vertices
    int k;         // Size of neighborhoods
    int *V;        // List of neighbors (n if empty)
    int16_t *D;    // Sorted dot products between vertices (descending)
    int16_t *Dmin; // Sorted dot products between vertices (descending)

    omp_lock_t *L; // Locks for mutual exclusion
} KNN;

KNN *KNN_init(int n, int k, float ref);

void KNN_free(KNN *g);

int KNN_validate(KNN *g);

static inline int KNN_contains_neighbor(KNN *g, int u, int v, int d)
{

    const int z = u * g->k;
    int j = 1;
    if (g->D[z] < d || (INT16_MAX != g->Dmin[u] && g->Dmin[u] > d))
    {
        return 0;
    }
    for (; j < g->k; j++)
        if ((g->V[z + j] == v) || ((g->V[z + j] - g->n) == v))
            return 1;
    return 0;
}
static inline void swapper(KNN *g, int pos_a, int pos_b)
{
    int D_temp = g->D[pos_a];
    g->D[pos_a] = g->D[pos_b];
    g->D[pos_b] = D_temp;
    int V_temp = g->V[pos_a];
    g->V[pos_a] = g->V[pos_b];
    g->V[pos_b] = V_temp;
}
static inline int KNN_insert_safe_min(KNN *g, int u, int v, int16_t d)
{
    if (u == v)
    {
        return 0;
    }
    int pos = (u * g->k); // Implementing max heap
    int basePos = (u * g->k);
    if (g->D[pos] <= d)
    {
        return 0;
    }
    omp_set_lock(&g->L[u]);

    if (KNN_contains_neighbor(g, u, v, d))
    {
        omp_unset_lock(&g->L[u]);

        return 0;
    }
    int heapPos = 0;
    // Insert- override max value
    if (g->Dmin[u] > d)
    {
        g->Dmin[u] = d;
    }
    g->D[pos] = d;
    g->V[pos] = v;

    // Bubble down
    while (2 * heapPos + 2 < g->k)
    {
        int left = 2 * heapPos + 1;
        int right = 2 * heapPos + 2;
        if (g->D[basePos + left] > g->D[basePos + right])
        {
            // left larger
            if (g->D[basePos + left] > d)
            {
                // have to swap
                swapper(g, basePos + left, basePos + heapPos);
                heapPos = left;
                continue;
            }
            else
            {
                break;
            }
        }
        else
        {
            // right larger
            if (g->D[basePos + right] > d)
            {
                // have to swap
                swapper(g, basePos + right, basePos + heapPos);
                heapPos = right;
                continue;
            }
            else
            {
                break;
            }
        }
    }
    omp_unset_lock(&g->L[u]);
    return 1;
}

void KNN_greedy_dist1_update(KNN *g, int u, const uint64_t *Matrix, int m,
                             int n);

float KNN_compute_recall(KNN *g, const int *KNNs, int m, int n, int k);
#define MAX_PIDG 4
typedef struct Hole
{
    int i;
    int ks[MAX_PIDG];
} Hole;
static void pidgeonize(KNN *g, uint8_t *matrix_bin, size_t bytesPerEntry,
                       size_t bn)
{

    const int CHUNKS = 4;
    const int CHUNKS_MAX = 65536;
    Hole holes[CHUNKS][CHUNKS_MAX];
    for (int i = 0; i < CHUNKS; i++)
    {
        for (int j = 0; j < CHUNKS_MAX; j++)
        {
            holes[i][j].i = 0;
        }
    }
    // insert
    uint16_t words[CHUNKS];
    for (int u = 0; u < 10 * CHUNKS_MAX; u++)
    {
        for (int i = 0; i < CHUNKS; i++)
            words[i] = 0;
        int c = 0;
        for (int j = 0; j < bytesPerEntry; j += bytesPerEntry / CHUNKS)
        {
            for (int k = 0; k < bytesPerEntry / CHUNKS; k += 2)
            {
                words[c] ^= matrix_bin[u * bytesPerEntry + j + k]
                            << 8 + matrix_bin[u * bytesPerEntry + j + k + 1];
            }
            c++;
        }
        for (int i = 0; i < CHUNKS; i++)
        {
            Hole *hole_ptr = &holes[i][words[i]];
            int k = 0;
            if (hole_ptr->i >= MAX_PIDG)
                continue;
            hole_ptr->ks[hole_ptr->i++] = u;
        }
    }
    for (int u = 0; u < g->n; u++)
    {
        for (int i = 0; i < CHUNKS; i++)
            words[i] = 0;
        int c = 0;
        for (int j = 0; j < bytesPerEntry; j += bytesPerEntry / CHUNKS)
        {
            for (int k = 0; k < bytesPerEntry / CHUNKS; k += 2)
            {
                words[c] ^= matrix_bin[u * bytesPerEntry + j + k]
                            << 8 + matrix_bin[u * bytesPerEntry + j + k + 1];
            }
            c++;
        }
        for (int i = 0; i < CHUNKS; i++)
        {
            Hole *hole_ptr = &holes[i][words[i]];
            for (int j = 0; j < hole_ptr->i; j++)
            {
                const int v = hole_ptr->ks[j];
                int d = hamming_portable(matrix_bin + (size_t)u * bn,
                                         matrix_bin + (size_t)v * bn, bn);
                KNN_insert_safe_min(g, u, v, d);
            }
            hole_ptr = &holes[i][words[i] >> 1];
            for (int j = 0; j < hole_ptr->i; j++)
            {
                const int v = hole_ptr->ks[j];
                int d = hamming_portable(matrix_bin + (size_t)u * bn,
                                         matrix_bin + (size_t)v * bn, bn);
                KNN_insert_safe_min(g, u, v, d);
            }
        }
    }
}