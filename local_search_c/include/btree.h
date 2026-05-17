#pragma once
#include "knn.h"
#include <stdint.h>
#define MAX_BLEAF_SIZE 65300
typedef struct BNode
{
    int is_leaf;

    // --- Internal Node Data ---

    struct BNode **pivots;

    // --- Leaf Node Data ---
    int *indices; // Array of data indices
    int num_points;
} BNode;

BNode *build_rp_btree(uint64_t *data, int *indices, int num_points, int bn, int all_points_n, int B);
void b_tree_init_KNN(KNN *g, BNode *tree, uint64_t *data, int bytes);
void printBTreeStats(BNode *n);
void free_btree(BNode *);
