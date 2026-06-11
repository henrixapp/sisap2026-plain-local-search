#pragma once

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

void KNN_insert_safe(KNN *g, int u, int v, int d);

void KNN_greedy_dist1_update_simple(KNN *g, int u, const uint64_t *Matrix, int m, int n);

float KNN_compute_recall(KNN *g, const int *KNNs, int m, int n, int k);