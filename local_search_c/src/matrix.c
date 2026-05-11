#include "matrix.h"
#include "hdf5.h"

#include <stdlib.h>
#include <immintrin.h>
#include <math.h>

uint16_t *matrix_load_h5_data(const char *Path, const char *Dataset, int *m, int *n)
{
    hid_t file = H5Fopen(Path, H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t dataset = H5Dopen2(file, Dataset, H5P_DEFAULT);

    hid_t dataspace = H5Dget_space(dataset);
    int n_dims = H5Sget_simple_extent_ndims(dataspace);
    hsize_t *Dims = malloc(sizeof(hsize_t) * n_dims);
    H5Sget_simple_extent_dims(dataspace, Dims, NULL);

    if (n_dims != 2)
    {
        fprintf(stderr, "Can't parse matrix with %d dims!\n", n_dims);
        return NULL;
    }

    *m = Dims[0], *n = Dims[1];

    uint16_t *Matrix = aligned_alloc(32, sizeof(uint16_t) * *m * *n);

    if (Matrix == NULL)
    {
        fprintf(stderr, "Failed to allocate memory for matrix");
        return NULL;
    }

    hid_t file_type = H5Dget_type(dataset);

    H5Dread(dataset, file_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, Matrix);

    H5Dclose(dataset);
    H5Fclose(file);
    free(Dims);

    return Matrix;
}

int *matrix_load_h5_indices(const char *Path, const char *Dataset, int *m, int *n)
{
    hid_t file = H5Fopen(Path, H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t dataset = H5Dopen2(file, Dataset, H5P_DEFAULT);

    hid_t dataspace = H5Dget_space(dataset);
    int n_dims = H5Sget_simple_extent_ndims(dataspace);
    hsize_t *Dims = malloc(sizeof(hsize_t) * n_dims);
    H5Sget_simple_extent_dims(dataspace, Dims, NULL);

    if (n_dims != 2)
    {
        fprintf(stderr, "Can't parse matrix with %d dims!\n", n_dims);
        return NULL;
    }

    *m = Dims[0], *n = Dims[1];

    int *Matrix = aligned_alloc(32, sizeof(int) * *m * *n);

    if (Matrix == NULL)
    {
        fprintf(stderr, "Failed to allocate memory for matrix");
        return NULL;
    }

    H5Dread(dataset, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, Matrix);

    H5Dclose(dataset);
    H5Fclose(file);
    free(Dims);

    return Matrix;
}

float compute_dot_product(const uint16_t *Matrix, int m, int n, int rowA, int rowB)
{
    const uint16_t *vecA = Matrix + ((size_t)rowA * n);
    const uint16_t *vecB = Matrix + ((size_t)rowB * n);

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    size_t i = 0;

    for (; i <= n - 16; i += 16)
    {
        __m256 a0 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(vecA + i)));
        __m256 b0 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(vecB + i)));
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);

        __m256 a1 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(vecA + i + 8)));
        __m256 b1 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(vecB + i + 8)));
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
    }

    acc0 = _mm256_add_ps(acc0, acc1);

    float res[8];
    _mm256_storeu_ps(res, acc0);
    float total_sum = 0.0;
    for (int j = 0; j < 8; j++)
    {
        total_sum += res[j];
    }

    // Clean up remaining elements (if n is not a multiple of 16)
    for (; i < n; i++)
    {
        float fa = _cvtsh_ss(vecA[i]);
        float fb = _cvtsh_ss(vecB[i]);
        total_sum += fa * fb;
    }

    return total_sum;
}

float rand_gaussian(float mu, float sigma)
{
    static float V1, V2, S;
    static int phase = 0;
    float X;

    if (phase == 0)
    {
        do
        {
            float U1 = (float)rand() / RAND_MAX;
            float U2 = (float)rand() / RAND_MAX;
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while (S >= 1 || S == 0);
        X = V1 * sqrt(-2 * log(S) / S);
    }
    else
    {
        X = V2 * sqrt(-2 * log(S) / S);
    }
    phase = 1 - phase;
    return (X * sigma + mu);
}

float *compute_random_embedding(const uint16_t *Matrix, int m, int n)
{
    const int k = 16;

    float *B = aligned_alloc(32, sizeof(float) * n * k);

    float sigma = 1.0f / sqrtf((float)k);
    for (int i = 0; i < n * k; i++)
        B[i] = rand_gaussian(0.0f, sigma);

    float *C = aligned_alloc(32, sizeof(float) * m * k);

#pragma omp parallel for
    for (size_t i = 0; i < m; i += 2)
    {
        __m256 acc00 = _mm256_setzero_ps();
        __m256 acc01 = _mm256_setzero_ps();
        __m256 acc10 = _mm256_setzero_ps();
        __m256 acc11 = _mm256_setzero_ps();

        for (size_t j = 0; j < n; j++)
        {
            __m256 b0 = _mm256_load_ps(B + j * k + 0);
            __m256 b1 = _mm256_load_ps(B + j * k + 8);

            __m256 a0 = _mm256_set1_ps(_cvtsh_ss(Matrix[(i + 0) * n + j]));

            acc00 = _mm256_fmadd_ps(b0, a0, acc00);
            acc01 = _mm256_fmadd_ps(b1, a0, acc01);

            __m256 a1 = _mm256_set1_ps(_cvtsh_ss(Matrix[(i + 1) * n + j]));

            acc10 = _mm256_fmadd_ps(b0, a1, acc10);
            acc11 = _mm256_fmadd_ps(b1, a1, acc11);
        }

        _mm256_store_ps(C + (i + 0) * k + 0, acc00);
        _mm256_store_ps(C + (i + 0) * k + 8, acc01);
        _mm256_store_ps(C + (i + 1) * k + 0, acc10);
        _mm256_store_ps(C + (i + 1) * k + 8, acc11);
    }

    free(B);

    return C;
}
