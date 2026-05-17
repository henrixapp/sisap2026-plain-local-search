#include "btree.h"
#include "algorithms.h"
#include <stdio.h>
BNode *create_bleaf(int *indices, int num_points)
{
    if (num_points > MAX_BLEAF_SIZE)
    {
        printf("WARNING %d\n", num_points);
    }
    BNode *node = (BNode *)malloc(sizeof(BNode));
    node->is_leaf = 1;
    node->num_points = num_points;
    node->indices = (int *)malloc(num_points * sizeof(int));
    for (int i = i; i < num_points; i++)
    {
        node->indices[i] = indices[i];
    }
    node->pivots = 0;
    return node;
}

// Recursive function to build the tree using Hamming distance
BNode *build_rp_btree(uint64_t *data, int *indices, int num_points, int bn, int all_points_n, int B)
{
    if (num_points <= MAX_BLEAF_SIZE)
    {
        return create_bleaf(indices, num_points);
    }

    // 1. Pick two random distinct pivot points
    int idxs[B];
    int k = 0;
    int min_diff = 10;
    uint64_t *ps[B];
    do
    {
        int idx1;
        int to_low = 0;
        int tries = 0;
        do
        {
            idx1 = indices[rand() % num_points]; // TODO not on all
            for (int i = 0; i < k; i++)
            {
                to_low |= hamming_portable(ps[i], data + bn * idx1, bn) < min_diff;
            }
        } while (to_low > 0 && tries++ < 3);

        ps[k] = data + bn * idx1;
        idxs[k] = idx1;
        k++;

        // If the two pivots are identical, we must stop to prevent infinite recursion
    } while (k < B);
    int counts[B];
    for (int i = 0; i < B; i++)
    {
        counts[i] = 0;
    }
    // 2. In-place Partitioning

    for (int i = 0; i < num_points; i++)
    {
        int min_k = 0;
        uint64_t *curr = data + bn * indices[i];

        int d = hamming_portable(ps[min_k], curr, bn);
        for (int k = 1; k < B; k++)
        {
            int d1 = hamming_portable(ps[k], curr, bn);
            if (d1 < d)
            {
                d = d1;
                min_k = k;
            }
        }
        counts[min_k]++;
    }
    int *sub_indices[B];
    for (int k = 0; k < B; k++)
    {
        sub_indices[k] = malloc(counts[k] * sizeof(int));
        counts[k] = 0;
    }
    for (int i = 0; i < num_points; i++)
    {
        int min_k = 0;
        uint64_t *curr = data + bn * indices[i];

        int d = hamming_portable(ps[min_k], curr, bn);
        for (int k = 1; k < B; k++)
        {
            int d1 = hamming_portable(ps[k], curr, bn);
            if (d1 < d)
            {
                min_k = k;
                d = d1;
            }
        }
        sub_indices[min_k][counts[min_k]++] = i;
    }
    BNode *node = (BNode *)malloc(sizeof(BNode));
    node->is_leaf = 0;
    node->num_points = 0;
    node->indices = malloc(sizeof(int) * B);
    node->pivots = (BNode **)malloc(sizeof(BNode *) * B);
    for (int k = 0; k < B; k++)
    {
        node->indices[k] = idxs[k];
        node->pivots[node->num_points++] = build_rp_btree(data, sub_indices[k], counts[k], bn, all_points_n, B);
        free(sub_indices[k]);
    }

    return node;
}

void free_btree(BNode *node)
{
    if (!node)
        return;
    free(node->indices);
    if (!node->is_leaf)
    {
        for (int k = 0; k < node->num_points; k++)
        {
            free_btree(node->pivots[k]);
        }
        free(node->pivots);
    }
    free(node);
}
void b_tree_init_KNN(KNN *g, BNode *tree, uint64_t *data, int bn)
{
    int errors = 0;
    int ties = 0;
#pragma omp parallel for reduction(+ : errors, ties)
    for (int u = 0; u < g->n; u++)
    {
        BNode *ptr = tree;
        uint64_t curr = data + bn * u;
        while (ptr != NULL)
        {
            if (ptr->is_leaf)
            {
                break;
            }
            int k_min = 0;
            int d1 = hamming_portable(data + ptr->indices[k_min] * bn, curr, bn);
            KNN_insert_safe_min(g, u, ptr->indices[k_min], d1);

            for (int k = 1; k < ptr->num_points; k++)
            {
                int d = hamming_portable(data + ptr->indices[k] * bn, curr, bn);
                if (d < d1)
                {
                    d = d1;
                    k_min = k;
                    KNN_insert_safe_min(g, u, ptr->indices[k], d);
                    KNN_insert_safe_min(g, ptr->indices[k], u, d);
                }
            }
            ptr = ptr->pivots[k_min];
        }
        if (ptr != NULL)
        {
            for (int j = 0; j < ptr->num_points; j++)
            {
                int d = hamming_portable(curr, data + ptr->indices[j] * bn, bn);
                KNN_insert_safe_min(g, u, ptr->indices[j], d);
                KNN_insert_safe_min(g, ptr->indices[j], u, d);
            }
        }
        else
        {
            errors++;
        }
    }
    printf("Errors: %d Ties: %d\n", errors, ties);
}
int bheight(BNode *n)
{
    if (n == NULL || n->is_leaf)
    {
        return 0;
    }
    int h = 0;
    for (int i = 0; i < n->num_points; i++)
    {
        int h1 = bheight(n->pivots[i]);
        if (h1 > h)
        {
            h = h1;
        }
    }
    return h + 1;
}
int bleaves(BNode *n)
{
    if (n == NULL)
    {
        return 0;
    }
    if (n->is_leaf)
    {
        return 1;
    }
    int leafes = 0;
    for (int i = 0; i < n->num_points; i++)
    {
        leafes += bleaves(n->pivots[i]);
    }
    return leafes;
}
void printBTreeStats(BNode *n)
{
    printf("leaves %d\n", bleaves(n));
    printf("height %d\n", bheight(n));
}