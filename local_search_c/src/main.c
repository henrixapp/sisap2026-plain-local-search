
#include <stdint.h>
#define _GNU_SOURCE

#include <stdint.h>

#include "algorithms.h"
#include "btree.h"
#include "knn.h"
#include "matrix.h"
#include "tree.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

const int kEmm = 16;

int compare_alt(const void *a, const void *b, void *c)
{
    const int x = *(const int *)a;
    const int y = *(const int *)b;
    float *V = (float *)c;
    if (V[x * kEmm] < V[y * kEmm])
        return -1;
    return V[x * kEmm] > V[y * kEmm];
}

int main(int argc, char **argv)
{
    printf("%s\n", argv[1]);

    int n = 0, m = 0;
    uint16_t *Matrix = matrix_load_h5_data(argv[1], "/train", &m, &n);

    int kn = 0, km = 0;
    int *KNNs = matrix_load_h5_indices(argv[1], "/allknn/knns", &km, &kn);

    printf("%d %d\n", n, m);
    printf("%d %d\n", kn, km);

    int bn = n / 64;

    uint64_t *Matrix_bin = aligned_alloc(32, sizeof(uint64_t) * ((size_t)m * bn));
    uint64_t *Matrix_bin2 = aligned_alloc(32, sizeof(uint64_t) * ((size_t)m * bn));
    for (int i = 0; i < m; i++)
        quantize_1024_fp16_avx2(Matrix + (size_t)i * n, (uint8_t *)(Matrix_bin + (size_t)i * bn), _CMP_GT_OQ);
    int prej = rand() % m;
    for (int i = 0; i < m; i++)
        quantize_1024_fp16_avx2_with_pred(Matrix + (size_t)i * n, Matrix + (size_t)prej * n, (uint8_t *)(Matrix_bin2 + (size_t)i * bn), _CMP_GE_OQ);
    KNN *g = KNN_init(m, 64, n);
    // pidgeonize(g, Matrix_bin, bn * sizeof(uint64_t), bn);
    int tree_size = g->n;
    int *indics = (int *)malloc(sizeof(int) * tree_size);
    for (int i = 0; i < tree_size; i++)
    {
        indics[i] = i;
    }
    double t0 = omp_get_wtime();
    for (int i = 0; i < 8; i++)
    {
        int mask = rand() % 2;
        Node *tree = build_binary_rp_tree(mask ? Matrix_bin : Matrix_bin2, indics, tree_size, bn, g->n);
        printf("BUILD tree %f\n", omp_get_wtime() - t0);
        printTreeStats(tree);
        t0 = omp_get_wtime();

        init_KNN(g, tree, mask ? Matrix_bin : Matrix_bin2, bn);
        free_tree(tree);
    }
    printf("Tree by tree %f\n", omp_get_wtime() - t0);

    free(indics);
    printf("%f\n", KNN_compute_recall(g, KNNs, km, kn, 15));
    t0 = omp_get_wtime();
    float *X = compute_random_embedding(Matrix, m, n, kEmm);

    double t1 = omp_get_wtime();

    printf("Computed embedding in %lf\n", t1 - t0);

    const int dist = 64;

#pragma omp parallel for
    for (int i = 0; i < 16; i++)
    {
        int *O = malloc(sizeof(int) * g->n);
        for (int i = 0; i < g->n; i++)
            O[i] = i;

        qsort_r(O, g->n, sizeof(int), compare_alt, X + i);

        for (int j = 0; j < g->n; j++)
        {
            int u = O[j];
            for (int k = j + 1; k < g->n && k < j + dist + 1; k++)
            {
                int v = O[k];

                int d = hamming_portable(Matrix_bin + (size_t)u * bn, Matrix_bin + (size_t)v * bn, bn);

                KNN_insert_safe_min(g, u, v, d);

                KNN_insert_safe_min(g, v, u, d);
            }
        }

        free(O);
    }

    double t2 = omp_get_wtime();

    printf("Finished initial sorting and inserts in %lf\n", t2 - t1);
    if (!KNN_validate(g))
        printf("Error!\n");
    printf("%f\n", KNN_compute_recall(g, KNNs, km, kn, 15));
    int part = g->n;
    printf("START_MESSAGE\n");
    int *O = malloc(sizeof(int) * g->n);
    for (int i = 0; i < g->n; i++)
        O[i] = i;
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < g->n / 10; j++)
        {
            int a = rand() % g->n;
            int b = rand() % g->n;
            int tmp = O[a];
            O[a] = O[b];
            O[b] = tmp;
        }
#pragma omp parallel
        {
            int *T = malloc(sizeof(int) * g->k * g->k); // what is that for?

#pragma omp for schedule(dynamic, 256)
            for (int u = 0; u < g->n; u++)
            {

                if ((u % 100000) == 10)
                {

                    int succ = KNN_greedy2_dist1_update(g, O[u], Matrix_bin, m, bn);
                    KNN_greedy_dist1_update(g, O[u], Matrix_bin, m, bn);

                    float recall = KNN_compute_recall(g, KNNs, km, kn, 15);
                    printf("%f %f %d\n", recall, omp_get_wtime() - t2, succ);
                }
                else if (u < part)
                {
                    KNN_greedy_dist1_update_fast(g, O[u], Matrix_bin, m, bn, 32 + (u % 20));
                    KNN_greedy_dist1_update_fast(g, O[u], Matrix_bin2, m, bn, 32 + (u % 20));
                }
                else
                    KNN_greedy_dist1_update(g, O[u], Matrix_bin, m, bn);
            }

            free(T);
            // part /= 8;
        }
        for (size_t i = 0; i < g->n * g->k; i++)
        {
            if (g->V[i] >= g->n)
                g->V[i] -= g->n;
        }
        if (!KNN_validate(g))
            printf("Error!\n");

        float recall = KNN_compute_recall(g, KNNs, km, kn, 15);
        printf("%f %f -1 new gen\n", recall, omp_get_wtime() - t2);
        if (recall > 0.8)
            break;
    }
    free(O);
    printf("STOP_MESSAGE\n");

    double t3 = omp_get_wtime();

    printf("Finished refinement in %lf\n", t3 - t2);

    KNN_free(g);

    free(Matrix);
    free(KNNs);
    free(X);

    return 0;
}