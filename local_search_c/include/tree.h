#pragma once
#include "knn.h"
#include <stdint.h>
#define MAX_LEAF_SIZE 64
typedef struct Node
{
    int is_leaf;

    // --- Internal Node Data ---
    int pivot1_index;
    int d;
    struct Node *left;
    struct Node *right;

    // --- Leaf Node Data ---
    int *indices; // Array of data indices
    int num_points;
} Node;

Node *build_binary_rp_tree(uint64_t *data, int *indices, int num_points, int num_bytes, int all_points_n);
Node *build_binary_rp_tree2(uint64_t *data, int *indices, int num_points, int num_bytes, int all_points_n);

void init_KNN(KNN *g, Node *tree, uint64_t *data, int bytes);
void printTreeStats(Node *n);