#include "tree.h"
#include "algorithms.h"
#include <stdio.h>
Node *create_leaf(int *indices, int num_points)
{
    if (num_points > MAX_LEAF_SIZE)
    {
        printf("WARNING %d\n", num_points);
    }
    Node *node = (Node *)malloc(sizeof(Node));
    node->is_leaf = 1;
    node->num_points = num_points;
    node->indices = (int *)malloc(num_points * sizeof(int));
    for (int i = i; i < num_points; i++)
    {
        node->indices[i] = indices[i];
    }
    node->pivot1_index = 0;
    node->pivot2_index = 0;
    node->left = NULL;
    node->right = NULL;
    return node;
}

// Recursive function to build the tree using Hamming distance
Node *build_binary_rp_tree(uint64_t *data, int *indices, int num_points, int bn, int all_points_n)
{
    if (num_points <= MAX_LEAF_SIZE)
    {
        return create_leaf(indices, num_points);
    }

    // 1. Pick two random distinct pivot points
    int idx1, idx2;
    int k = 0;
    uint64_t *p1, *p2;
    do
    {
        k++;
        idx1 = indices[rand() % num_points]; // TODO not on all
        idx2 = indices[rand() % num_points];
        if (idx1 == idx2)
        {
            idx2 = (idx1 + 1) % all_points_n;
        }

        p1 = data + bn * idx1;
        p2 = data + bn * idx2;

        // If the two pivots are identical, we must stop to prevent infinite recursion
    } while (hamming_portable(p1, p2, bn) == 0 && k < 10);
    if (k == 10)
    {
        return create_leaf(indices, num_points);
    }
    // 2. In-place Partitioning
    int left_ptr = 0;
    int right_ptr = num_points - 1;

    while (left_ptr <= right_ptr)
    {
        int dist1 = hamming_portable(data + bn * indices[left_ptr], p1, bn);
        int dist2 = hamming_portable(data + bn * indices[left_ptr], p2, bn);

        int go_left;
        if (dist1 < dist2)
        {
            go_left = 1;
        }
        else if (dist1 > dist2)
        {
            go_left = 0;
        }
        else
        {
            // CRITICAL: Random tie-breaking. In discrete binary spaces,
            // exact ties are very common. Without this, trees become highly unbalanced.
            go_left = rand() % 2;
        }

        if (go_left)
        {
            left_ptr++; // Keep on the left side
        }
        else
        {
            // Swap with the right pointer
            int temp = indices[left_ptr];
            indices[left_ptr] = indices[right_ptr];
            indices[right_ptr] = temp;
            right_ptr--;
        }
    }

    // 3. Handle degenerate splits (all points fell on one side due to tie-breaking luck)
    if (left_ptr == 0 || left_ptr == num_points)
    {
        return create_leaf(indices, num_points);
    }

    // 4. Create internal node
    Node *node = (Node *)malloc(sizeof(Node));
    node->is_leaf = 0;
    node->num_points = 0;
    node->indices = NULL;

    // Copy the pivots so the tree owns them (safe if underlying data shifts)
    node->pivot1_index = idx1;
    node->pivot2_index = idx2;

    // 5. Recurse
    node->left = build_binary_rp_tree(data, indices, left_ptr, bn, all_points_n);
    node->right = build_binary_rp_tree(data, indices + left_ptr, num_points - left_ptr, bn, all_points_n);

    return node;
}

void free_tree(Node *node)
{
    if (!node)
        return;
    if (node->is_leaf)
    {
        free(node->indices);
    }
    else
    {
        free_tree(node->left);
        free_tree(node->right);
    }
    free(node);
}
void init_KNN(KNN *g, Node *tree, uint64_t *data, int bn)
{
    int errors = 0;
    int ties = 0;
#pragma omp parallel for reduction(+ : errors, ties)
    for (int u = 0; u < g->n; u++)
    {
        Node *prev = NULL;
        Node *ptr = tree;
        uint64_t curr = data + bn * u;
        while (ptr != NULL)
        {
            if (ptr->is_leaf)
            {
                break;
            }
            int d1 = hamming_portable(data + ptr->pivot1_index * bn, curr, bn);
            int d2 = hamming_portable(data + ptr->pivot2_index * bn, curr, bn);
            if (d1 < d2)
            {
                ptr = ptr->left;
            }
            else if (d2 <= d1)
            {
                ptr = ptr->right;
            }
            else
            {
                ties++;
                ptr = rand() % 2 ? ptr->left : ptr->right;
            }
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
int height(Node *n)
{
    if (n == NULL)
    {
        return 0;
    }
    int l = height(n->left);
    int r = height(n->right);
    int h = l > r ? l : r;
    return h + 1;
}
int leaves(Node *n)
{
    if (n == NULL)
    {
        return 0;
    }
    if (n->is_leaf)
    {
        return 1;
    }
    int l = leaves(n->left);
    int r = leaves(n->right);
    return l + r;
}
void printTreeStats(Node *n)
{
    printf("leaves %d\n", leaves(n));
    printf("height %d\n", height(n));
}