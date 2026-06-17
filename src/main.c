#include "algorithms.h"
#include "knn.h"
#include "matrix.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int Qb = 264;
int Qn = 32;
int k = 64;
int Ln = 2;

/*
    Qsort-ish indexing:
    - pick random pivot point
    - partition data around median distance from pivot
    - recurse on each half
    - when a segment becomes small, points are likely neighbors
*/

static void qsort_recursive_knn(const uint64_t *Matrix, int n, VDEntry *O, int *C, int m, KNN *g, int qb)
{
    // Base case, test all pairs and insert into KNN structure
    if (m < qb)
    {
        for (int i = 0; i < m; i++)
        {
            int u = O[i].v;

            for (int j = i + 1; j < m; j++)
            {
                int v = O[j].v;

                int d = hamming_avx2_lut(Matrix + u * n, Matrix + v * n, n);

                if (d < g->D[(u * g->k) + (g->k - 1)])
                    KNN_insert_safe(g, u, v, d);

                if (d < g->D[(v * g->k) + (g->k - 1)])
                    KNN_insert_safe(g, v, u, d);
            }
        }

        return;
    }

    // memset(C, 0, sizeof(int) * n * 64);
    for (int i = 0; i < n * 64; i++)
        C[i] = 0;

    // Random pivot element
    int u = O[fast_rand() % m].v;

    // Compute hamming distances for all points to pivot
    for (int i = 0; i < m; i++)
    {
        int v = O[i].v;
        int d = hamming_avx2_lut(Matrix + u * n, Matrix + v * n, n);
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
        if (O[i].d >= pd)
        {
            VDEntry t = O[p];
            O[p] = O[i];
            O[i] = t;
            p++;
        }
    }

    // Protect against infinite recursion
    if (p == 0 || p == m)
        p = m / 2;

    qsort_recursive_knn(Matrix, n, O, C, p, g, qb);
    qsort_recursive_knn(Matrix, n, O + p, C, m - p, g, qb);
}

static inline void insert_safe_rerank(KNN *g, int *N, float *Nd, int u, int v, float d)
{
    omp_set_lock(&g->L[u & 8191].lock);

    if (d < Nd[14])
        goto end;

    for (int i = 0; i < 15; i++)
        if (N[i] == v + 1)
            goto end;

    int p = 13;
    while (p >= 0 && Nd[p] < d)
    {
        N[p + 1] = N[p];
        Nd[p + 1] = Nd[p];
        p--;
    }
    N[p + 1] = v + 1;
    Nd[p + 1] = d;

end:
    omp_unset_lock(&g->L[u & 8191].lock);
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

    KNN *g = KNN_init(m, k, n);

    // Finished parsing and allocation

    // 1. Quantization to 1 bit
    // Currently this step assumes that n / 8 is an integer

    double t0 = omp_get_wtime();

    int bn = (n + 63) / 64;
    if ((bn % 4) > 0)
        bn += 4 - (bn % 4);

    uint64_t *Matrix_bin = aligned_alloc(32, sizeof(uint64_t) * ((size_t)m * bn));

#pragma omp parallel for
    for (int i = 0; i < m * bn; i++)
        Matrix_bin[i] = 0;

#pragma omp parallel for
    for (int i = 0; i < m; i++)
        quantize_8_fp16_avx2(Matrix + (size_t)i * n, (uint8_t *)(Matrix_bin + (size_t)i * bn), n);

    double t1 = omp_get_wtime();

    printf("1-bit quantization in %lf\n", t1 - t0);

    // 2. Initialize ANN with qsort idea

#pragma omp parallel
    {
        VDEntry *O = malloc(sizeof(VDEntry) * m);
        int *C = malloc(sizeof(int) * bn * 64);

#pragma omp for
        for (int i = 0; i < Qn; i++)
        {
            for (int i = 0; i < m; i++)
                O[i].v = i;
            qsort_recursive_knn(Matrix_bin, bn, O, C, m, g, Qb - i);
        }

        free(O);
        free(C);
    }

    double t2 = omp_get_wtime();

    printf("Initialized ANN in %lf\n", t2 - t1);

    if (KNNs != NULL)
        printf("Recall %f\n", KNN_compute_recall(g, KNNs, km, kn, 15));

    // 3. Local search refinement

    {
        double tt = omp_get_wtime();

#pragma omp parallel
        {
            const int target = g->n / 2;
#pragma omp for schedule(dynamic, 256)
            for (int u = 0; u < target; u++)
            {
                KNN_greedy_dist1_update_simple(g, u, Matrix_bin, m, bn);
                KNN_greedy_dist1_update_simple(g, g->n - 1 - u, Matrix_bin, m, bn);
            }
        }
        if (KNNs != NULL)
        {
            float recall = KNN_compute_recall(g, KNNs, km, kn, 15);
            printf("It %d: Recall %f %lf\n", 1, recall, omp_get_wtime() - tt);
            if (recall > 0.8)
                goto endwork;
        }
    }

    for (int i = 1; i < Ln; i++)
    {
        double tt = omp_get_wtime();

#pragma omp parallel
        {

#pragma omp for schedule(dynamic, 256)
            for (int u = 0; u < g->n; u++)
            {
                KNN_greedy_dist1_update_simple_round2(g, u, Matrix_bin, m, bn);
            }
        }
        if (KNNs != NULL)
        {
            float recall = KNN_compute_recall(g, KNNs, km, kn, 15);
            printf("It %d: Recall %f %lf\n", 2, recall, omp_get_wtime() - tt);
            if (recall > 0.8)
                goto endwork;
        }
    }

endwork:;
    double t3 = omp_get_wtime();

    printf("Finished refinement in %lf\n", t3 - t2);

    int *k_NN = calloc(g->n * 15, sizeof(int));
    float *k_NN_d = calloc(g->n * 15, sizeof(float));

#pragma omp parallel for schedule(dynamic, 256)
    for (int u = 0; u < g->n; u++)
    {
        int *N = k_NN + (u * 15);
        float *Nd = k_NN_d + (u * 15);

        for (int i = 0; i < g->k; i++)
        {
            int v = g->V[u * g->k + i];

            float d = compute_dot_product(Matrix, m, n, u, v);

            if (d > Nd[14])
                insert_safe_rerank(g, N, Nd, u, v, d);

            if (d > k_NN_d[v * 15 + 14])
                insert_safe_rerank(g, k_NN + (v * 15), k_NN_d + (v * 15), v, u, d);
        }
    }

    double t4 = omp_get_wtime();

    printf("Finished reranking in %lf\n", t4 - t3);

    if (KNNs != NULL)
        printf("Recall %f\n", KNN_compute_recall_standalone(k_NN, KNNs, km, kn, 15));

    matrix_store_h5_results(argv[2], m, 15, k_NN, k_NN_d, t2 - t0, t4 - t2, argv[3], Qb, Qn, k, Ln);

    free(Matrix);
    free(KNNs);
    free(k_NN);
    free(k_NN_d);

    return 0;
}
