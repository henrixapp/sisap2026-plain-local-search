#define _GNU_SOURCE

#include <stdint.h>

#include "algorithms.h"
#include "knn.h"
#include "matrix.h"

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

    for (int i = 0; i < m; i++)
        quantize_1024_fp16_avx2(Matrix + (size_t)i * n, (uint8_t *)(Matrix_bin + (size_t)i * bn));

    KNN *g = KNN_init(m, 64, n);

    printf("%f\n", KNN_compute_recall(g, KNNs, km, kn, 15));

    double t0 = omp_get_wtime();
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

    printf("%f\n", KNN_compute_recall(g, KNNs, km, kn, 15));

    for (int i = 0; i < 16; i++)
    {
#pragma omp parallel
        {
            int *T = malloc(sizeof(int) * g->k * g->k); // what is that for?

#pragma omp for schedule(dynamic, 256)
            for (int u = 0; u < g->n; u++)
            {
                KNN_greedy_dist1_update(g, u, Matrix_bin, m, bn);
            }

            free(T);
        }

        if (!KNN_validate(g))
            printf("Error!\n");

        float recall = KNN_compute_recall(g, KNNs, km, kn, 15);
        printf("%f\n", recall);
        if (recall > 0.8)
            break;
    }

    double t3 = omp_get_wtime();

    printf("Finished refinement in %lf\n", t3 - t2);

    KNN_free(g);

    free(Matrix);
    free(KNNs);
    free(X);

    return 0;
}