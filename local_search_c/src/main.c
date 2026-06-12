#include "algorithms.h"
#include "knn.h"
#include "matrix.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct pair_struct
{
    int u, d;
} pair;

/*
    Qsort-ish indexing:
    - pick random pivot point
    - partition data around median distance from pivot
    - recurse on each half
    - when a segment becomes small, point are likely neighbors
*/
static void qsort_recursive_knn(const uint64_t *Matrix, int n, pair *O, int *C, int m, KNN *g)
{
    // Base case, test all pairs and insert into KNN structure
    if (m <= 120)
    {
        for (int i = 0; i < m; i++)
        {
            int u = O[i].u;
            for (int j = i + 1; j < m; j++)
            {
                int v = O[j].u;

                int d = hamming_portable(Matrix + u * n, Matrix + v * n, n);

                if (d < g->VD[(u * g->k) + (g->k - 1)].d)
                    KNN_insert_safe(g, u, v, d);

                if (d < g->VD[(v * g->k) + (g->k - 1)].d)
                    KNN_insert_safe(g, v, u, d);
            }
        }

        return;
    }

    for (int i = 0; i < n * 64; i++)
        C[i] = 0;

    // Random pivot element
    int u = O[fast_rand() % m].u;

    // Compute hamming distances for all points to pivot
    for (int i = 0; i < m; i++)
    {
        int v = O[i].u;
        int d = hamming_portable(Matrix + u * n, Matrix + v * n, n);
        O[i].d = d;
        C[d]++;
    }

    // Find median distance
    int p = 0, pd = -1;
    for (int i = 0; i < n * 64; i++)
    {
        p += C[i];
        if (p >= m / 2)
        {
            pd = i;
            break;
        }
    }

    // Partition data
    p = 0;
    for (int i = 0; i < m; i++)
    {
        if (O[i].d > pd)
        {
            pair t = O[p];
            O[p] = O[i];
            O[i] = t;
            p++;
        }
    }

    // Protect against infinite recursion
    if (p == 0 || p == m)
        p = m / 2;

    qsort_recursive_knn(Matrix, n, O, C, p, g);
    qsort_recursive_knn(Matrix, n, O + p, C, m - p, g);
}

static inline void qsort_knn(const uint64_t *Matrix, int m, int n, KNN *g)
{
    pair *O = malloc(sizeof(pair) * m);
    int *C = malloc(sizeof(int) * n * 64);
    for (int i = 0; i < m; i++)
        O[i].u = i;

    qsort_recursive_knn(Matrix, n, O, C, m, g);

    free(O);
    free(C);
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

    prepare_random();

    KNN *g = KNN_init(m, 72, n);

    // Finished parsing and allocation

    // 1. Quantization to 1 bit
    // Currently this step assumes that n / 8 is an integer

    double t0 = omp_get_wtime();

    int bn = (n + 63) / 64;
    uint64_t *Matrix_bin = aligned_alloc(32, sizeof(uint64_t) * ((size_t)m * bn));
    for (int i = 0; i < m; i++)
        quantize_8_fp16_avx2(Matrix + (size_t)i * n, (uint8_t *)(Matrix_bin + (size_t)i * bn), n);

    double t1 = omp_get_wtime();

    printf("1-bit quantization in %lf\n", t1 - t0);

    // 2. Initialize ANN with qsort idea
    int num_qsort_tries = 64;

#pragma omp parallel
    {
        pair *O = malloc(sizeof(pair) * m);
        int *C = malloc(sizeof(int) * bn * 64);

        for (int i = 0; i < m; i++)
            O[i].u = i;

#pragma omp for
        for (int i = 0; i < num_qsort_tries; i++)
        {
            qsort_recursive_knn(Matrix_bin, bn, O, C, m, g);
        }

        free(O);
        free(C);
    }

    double t2 = omp_get_wtime();

    printf("Initializedd ANN in %lf\n", t2 - t1);

    if (KNNs != NULL)
        printf("Recall %f\n", KNN_compute_recall(g, KNNs, km, kn, 15));

    if (!KNN_validate(g))
        printf("Error in KNN\n");

    // 3. Local search refinement

    for (int i = 0; i < 3; i++)
    {
        double tt = omp_get_wtime();

#pragma omp parallel
        {

#pragma omp for schedule(dynamic, 256)
            for (int u = 0; u < g->n; u++)
            {
                KNN_greedy_dist1_update_simple(g, u, Matrix_bin, m, bn);
            }
        }

        if (!KNN_validate(g))
            printf("Error!\n");

        if (KNNs != NULL)
        {
            float recall = KNN_compute_recall(g, KNNs, km, kn, 15);
            printf("It %d: Recall %f %lf\n", i + 1, recall, omp_get_wtime() - tt);
            if (recall > 0.8)
                break;
        }
    }

    double t3 = omp_get_wtime();

    printf("Finished refinement in %lf\n", t3 - t2);

    int *k_NN = calloc(g->n * 15, sizeof(int));
    float *k_NN_d = calloc(g->n * 15, sizeof(float));

#pragma omp parallel for
    for (int u = 0; u < g->n; u++)
    {
        int *N = k_NN + (u * 15);
        float *Nd = k_NN_d + (u * 15);

        for (int i = 0; i < g->k; i++)
        {
            int v = g->VD[u * g->k + i].v;
            float d = compute_dot_product(Matrix, m, n, u, v);

            if (d < Nd[14])
                continue;

            int p = 13;
            while (p >= 0 && Nd[p] < d)
            {
                N[p + 1] = N[p];
                Nd[p + 1] = Nd[p];
                p--;
            }
            N[p + 1] = v + 1;
            Nd[p + 1] = d;
        }
    }

    matrix_store_h5_results(argv[2], m, 15, k_NN, k_NN_d, t2 - t0, t3 - t2, argv[3]);

    free(Matrix);
    free(KNNs);

    return 0;
}