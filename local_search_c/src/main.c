#include "matrix.h"
#include "algorithms.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

static inline float find_min(int *V, float *D, int n)
{
    float min = 1000000.0f;
    for (int i = 0; i < n; i++)
    {
        if (D[i] > min)
            continue;

        min = D[i];
    }
    return min;
}

static inline void remove_min(int *V, float *D, int n)
{
    int p = -1;
    float min = 1000000.0f;
    for (int i = 0; i < n; i++)
    {
        if (D[i] > min)
            continue;

        p = i;
        min = D[i];
    }

    assert(p >= 0);

    for (int i = p; i < n - 1; i++)
    {
        V[i] = V[i + 1];
        D[i] = D[i + 1];
    }
}

static inline void insert_new(int *V, float *D, int n, int v, float d)
{
    int p = n - 1;
    for (int i = 0; i < n - 1; i++)
    {
        if (V[i] < v)
            continue;

        p = i;
        break;
    }

    for (int i = n - 1; i > p; i--)
    {
        V[i] = V[i - 1];
        D[i] = D[i - 1];
    }

    V[p] = v;
    D[p] = d;
}

static inline void swap(int **A, int **B)
{
    int *t = *A;
    *A = *B;
    *B = t;
}

void greedy_update(const uint16_t *Matrix, int m, int n, int *V, float *D, int k, int u, int *T1, int *T2)
{
    int c = 0;
    for (int i = 0; i < k; i++)
    {
        int v = V[u * k + i];
        c = set_union(T1, c, V + v * k, k, T2);
        swap(&T1, &T2);
    }

    c = remove_duplicates(T1, c);
    c = set_difference(T1, c, V + u * k, k, T2);
    float min = find_min(V + u * k, D + u * k, k);

    for (int i = 0; i < c; i++)
    {
        int v = T2[i];
        if (v == u)
            continue;

        float d = compute_dot_product(Matrix, m, n, u, v);
        if (d > min)
        {
            remove_min(V + u * k, D + u * k, k);
            insert_new(V + u * k, D + u * k, k, v, d);
            min = find_min(V + u * k, D + u * k, k);
        }
    }
}

float compute_recall(int *V, float *D, int k, const int *KNNs, int m, int n)
{
    int tp = 0;

#pragma omp parallel for reduction(+ : tp)
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            int u = V[i * k + j];

            // Using the first 16 elements since the input includes self loops
            for (int l = 0; l < 16; l++)
            {
                int v = KNNs[i * n + l] - 1;
                if (u == v)
                {
                    tp++;
                    break;
                }
            }
        }
    }
    return (double)tp / (double)(m * 15);
}

int compare(const void *a, const void *b)
{
    const int x = *(const int *)a;
    const int y = *(const int *)b;
    return x - y;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Usage %s [input.h5]\n", argv[0]);
        return 0;
    }

    printf("%s\n", argv[1]);

    int n = 0, m = 0;
    uint16_t *Matrix = matrix_load_h5_data(argv[1], "/train", &m, &n);

    int kn = 0, km = 0;
    int *KNNs = matrix_load_h5_indices(argv[1], "/allknn/knns", &km, &kn);

    printf("%d %d\n", n, m);
    printf("%d %d\n", kn, km);

    int k = 32;

    int *V = malloc(sizeof(int) * m * k);
    float *D = malloc(sizeof(float) * m * k);

#pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
            V[i * k + j] = (i + 1 + j) % m;

        qsort(V + i * k, k, sizeof(int), compare);

        for (int j = 0; j < k; j++)
            D[i * k + j] = compute_dot_product(Matrix, m, n, i, V[i * k + j]);

        if (!set_is_sorted(V + i * k, k))
            printf("Error!\n");
    }

    float recall = compute_recall(V, D, k, KNNs, km, kn);
    printf("%f\n", recall);

    int **T1, **T2;

#pragma omp parallel
    {
        int nt = omp_get_num_threads();
        int tid = omp_get_thread_num();
#pragma omp single
        {
            T1 = malloc(sizeof(int *) * nt);
            T2 = malloc(sizeof(int *) * nt);
        }
        T1[tid] = malloc(sizeof(int) * k * k);
        T2[tid] = malloc(sizeof(int) * k * k);

        // This next stuff was to initialize the whole KNN structure, seems to be slower but a bit better

        // int tries = 128;
        // unsigned int seed = tid;

        // #pragma omp for schedule(dynamic, 256)
        //         for (int u = 0; u < m; u++)
        //         {
        //             float min = find_min(V + u * k, D + u * k, k);

        //             for (int i = 0; i < tries; i++)
        //             {
        //                 int v = rand_r(&seed) % m;
        //                 if (v == u)
        //                     continue;
        //                 int in = 0;
        //                 for (int j = 0; j < k; j++)
        //                 {
        //                     if (V[u * k + j] == v)
        //                         in = 1;
        //                 }
        //                 if (in)
        //                     continue;

        //                 float d = compute_dot_product(Matrix, m, n, u, v);
        //                 if (d > min)
        //                 {
        //                     remove_min(V + u * k, D + u * k, k);
        //                     insert_new(V + u * k, D + u * k, k, v, d);
        //                     min = find_min(V + u * k, D + u * k, k);
        //                 }
        //             }
        //         }
    }

    recall = compute_recall(V, D, k, KNNs, km, kn);
    printf("%f\n", recall);

    while (1)
    {
#pragma omp parallel
        {
            int *_T1 = T1[omp_get_thread_num()];
            int *_T2 = T2[omp_get_thread_num()];
#pragma omp for schedule(dynamic, 256)
            for (int u = 0; u < m; u++)
            {
                greedy_update(Matrix, m, n, V, D, k, u, _T1, _T2);
            }
        }

        float recall = compute_recall(V, D, k, KNNs, km, kn);
        printf("%f\n", recall);
    }

    free(Matrix);
    free(KNNs);

    free(V);
    free(D);

    return 0;
}
