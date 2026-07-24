#include "algorithms.h"
#include "alloc.h"
#include "kernels.h"
#include "knn.h"
#include "matrix.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int Qb = 256; // Base case size of the vantage point partition
int Qn = 12;  // Number of randomized partition rounds
int k = 96;   // Size of the approximate neighborhoods carried into reranking
int Ln = 2;   // Local search refinement rounds
int Du = 32;  // Neighbors of u expanded during local search
int Dv = 40;  // Neighbors of each of those expanded in turn
int R = 18;   // Shortlist kept by the int8 scan and rescored exactly
int Ob = 64;  // Leaf size of the partition that defines the processing order
int Sy = 1;   // Rounds that also insert the reverse edge

// Dispatch table, bound once in main to the best flavour this CPU supports.
static void (*init_pass)(KNN *, const uint64_t *, int, int, int, int);
static void (*build_order)(const uint64_t *, int, int, int, int *);
static void (*refine)(KNN *, const uint64_t *, const int *, int, int, int, int, int);
static void (*rerank)(KNN *, const uint16_t *, const int8_t *, const float *, const int *, int,
                      int, int, int *, float *);

static int env_int(const char *name, int def)
{
    const char *s = getenv(name);
    return s ? atoi(s) : def;
}

int main(int argc, char **argv)
{
    printf("%s\n", argv[1]);

    Qb = env_int("QB", Qb);
    Qn = env_int("QN", Qn);
    k = env_int("KK", k);
    Ln = env_int("LN", Ln);
    Du = env_int("DU", Du);
    Dv = env_int("DV", Dv);
    R = env_int("RR", R);
    Ob = env_int("OB", Ob);
    Sy = env_int("SY", Sy);
    if (R < 15)
        R = 15;

    if (cpu_has_avx512_popcnt() && !env_int("NOAVX512", 0))
    {
        init_pass = knn_init_pass_v4;
        build_order = knn_build_order_v4;
        refine = knn_refine_v4;
        rerank = knn_rerank_v4;
        printf("Using AVX-512 kernels\n");
    }
    else
    {
        init_pass = knn_init_pass_v3;
        build_order = knn_build_order_v3;
        refine = knn_refine_v3;
        rerank = knn_rerank_v3;
        printf("Using AVX2 kernels\n");
    }

    int n = 0, m = 0;
    uint16_t *Matrix = matrix_load_h5_data(argv[1], "/train", &m, &n);

    int kn = 0, km = 0;
    int *KNNs = matrix_load_h5_indices(argv[1], "/allknn/knns", &km, &kn);

    printf("%d %d\n", n, m);
    printf("%d %d\n", kn, km);

    prepare_random();

    KNN *g = KNN_init(m, k, n);

    // Finished parsing and allocation

    // 1. Build the two compressed views of the data

    double t0 = omp_get_wtime();

    // Pad to a multiple of 8 words so the AVX-512 kernel can consume the row
    // in whole 512 bit chunks.
    int bn = (n + 63) / 64;
    if ((bn % 8) > 0)
        bn += 8 - (bn % 8);

    uint64_t *Matrix_bin = alloc64(sizeof(uint64_t) * ((size_t)m * bn));

#pragma omp parallel for
    for (size_t i = 0; i < (size_t)m * bn; i++)
        Matrix_bin[i] = 0;

    // Symmetric int8 codes for the reranking scan: one byte per dimension
    // instead of two, which is exactly what that bandwidth bound pass costs.
    int8_t *Q = alloc64((size_t)m * n);
    float *S = alloc64(sizeof(float) * m);

    // Both views are derived in one sweep so each fp16 row is read once.
#pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
        const uint16_t *src = Matrix + (size_t)i * n;

        quantize_8_fp16_avx2(src, (uint8_t *)(Matrix_bin + (size_t)i * bn), n);

        float mx = 0.0f;
        for (int j = 0; j < n; j++)
        {
            float v = _cvtsh_ss(src[j]);
            float a = v < 0.0f ? -v : v;
            if (a > mx)
                mx = a;
        }

        float scale = mx > 0.0f ? mx / 127.0f : 1.0f;
        float inv = 1.0f / scale;

        int8_t *dst = Q + (size_t)i * n;
        for (int j = 0; j < n; j++)
        {
            int q = (int)lrintf(_cvtsh_ss(src[j]) * inv);
            dst[j] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
        }

        S[i] = scale;
    }

    double t1 = omp_get_wtime();

    printf("Quantization in %lf\n", t1 - t0);

    // 2. Initialize ANN with randomized vantage point partitions

    init_pass(g, Matrix_bin, m, bn, Qb, Qn);

    // Order the remaining passes for cache reuse
    int *Ord = malloc(sizeof(int) * m);
    build_order(Matrix_bin, m, bn, Ob, Ord);

    double t2 = omp_get_wtime();

    printf("Initialized ANN in %lf\n", t2 - t1);

    if (KNNs != NULL)
        printf("Recall %f\n", KNN_compute_recall(g, KNNs, km, kn, 15));

    // 3. Local search refinement

    for (int i = 0; i < Ln; i++)
    {
        double tt = omp_get_wtime();

        // Only the first round pays for symmetric insertion: it is what turns
        // the sparse initial sketch into a connected graph. Later rounds find
        // almost nothing that way and the extra locking is not worth it.
        refine(g, Matrix_bin, Ord, m, bn, Du, Dv, i < Sy);

        double te = omp_get_wtime();
        if (KNNs != NULL)
            printf("It %d: Recall %f %lf\n", i + 1, KNN_compute_recall(g, KNNs, km, kn, 15),
                   te - tt);
    }

    double t3 = omp_get_wtime();

    printf("Finished refinement in %lf\n", t3 - t2);

    // 4. Rerank the candidates with exact fp16 inner products

    int *Shortlist = alloc64(sizeof(int) * (size_t)g->n * R);
    float *Shortlist_d = alloc64(sizeof(float) * (size_t)g->n * R);

#pragma omp parallel for
    for (size_t i = 0; i < (size_t)g->n * R; i++)
    {
        Shortlist[i] = 0;
        Shortlist_d[i] = 0.0f;
    }

    rerank(g, Matrix, Q, S, Ord, m, n, R, Shortlist, Shortlist_d);

    // Keep the 15 best of every shortlist
    int *k_NN = malloc(sizeof(int) * (size_t)g->n * 15);
    float *k_NN_d = malloc(sizeof(float) * (size_t)g->n * 15);

#pragma omp parallel for
    for (int u = 0; u < g->n; u++)
        for (int i = 0; i < 15; i++)
        {
            k_NN[(size_t)u * 15 + i] = Shortlist[(size_t)u * R + i];
            k_NN_d[(size_t)u * 15 + i] = Shortlist_d[(size_t)u * R + i];
        }

    free(Shortlist);
    free(Shortlist_d);

    double t4 = omp_get_wtime();

    printf("Finished reranking in %lf\n", t4 - t3);
    printf("Total %lf\n", t4 - t0);

    if (KNNs != NULL)
        printf("Recall %f\n", KNN_compute_recall_standalone(k_NN, KNNs, km, kn, 15));

    if (argc > 2)
        matrix_store_h5_results(argv[2], m, 15, k_NN, k_NN_d, t2 - t0, t4 - t2,
                                argc > 3 ? argv[3] : "unknown", Qb, Qn, k, Ln);

    free(Matrix);
    free(Matrix_bin);
    free(Ord);
    free(Q);
    free(S);
    free(KNNs);
    free(k_NN);
    free(k_NN_d);
    KNN_free(g);

    return 0;
}
