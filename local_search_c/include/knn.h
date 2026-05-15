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
static inline void swapper(KNN* g,int pos_a, int pos_b){
        int D_temp = g->D[pos_a];
        g->D[pos_a]= g->D[pos_b];
        g->D[pos_b] = D_temp;
         int V_temp = g->V[pos_a];
        g->V[pos_a]= g->V[pos_b];
        g->V[pos_b] = V_temp;
}
static inline void KNN_insert_safe_min(KNN *g, int u, int v, int16_t d)
{
    int pos = (u * g->k); // Implementing max heap
    int basePos = (u * g->k);
    if ( g->D[pos] < d  )
    {
        return;
    }
    omp_set_lock(&g->L[u]);

    if(KNN_contains_neighbor(g, u, v)){
    omp_unset_lock(&g->L[u]);
        
        return;
    }
    int heapPos =0;
    // Insert- override max value
    g->D[pos] = d;
    g->V[pos] = v;
    
    // Bubble down
    while(2*heapPos+2<g->k){
        int left = 2*heapPos+1;
        int right = 2*heapPos+2;
        if (g->D[basePos+left]>g->D[basePos+right]){
            //left larger
            if(g->D[basePos+left]>d){
                // have to swap
                swapper(g,basePos+left,basePos+heapPos);
                heapPos = left;
                continue;
            } else {
                break;
            }
        } else {
        //right larger
            if(g->D[basePos+right]>d){
                // have to swap
                swapper(g,basePos+right,basePos+heapPos);
                heapPos = right;
                continue;
            } else {
                break;
            }
        }
    }


    

    omp_unset_lock(&g->L[u]);
}

void KNN_greedy_dist1_update(KNN *g, int u, const uint64_t *Matrix, int m, int n);

float KNN_compute_recall(KNN *g, const int *KNNs, int m, int n, int k);
