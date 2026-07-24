#include "matrix.h"

#include "alloc.h"
#include "hdf5.h"

#include <stdlib.h>
#include <immintrin.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>

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

    uint16_t *Matrix = alloc64(sizeof(uint16_t) * (size_t)*m * *n);

    if (Matrix == NULL)
    {
        fprintf(stderr, "Failed to allocate memory for matrix");
        return NULL;
    }

    hid_t file_type = H5Dget_type(dataset);

    // Check if the file data type is a 32-bit little-endian float (or matches native float)
    if (H5Tequal(file_type, H5T_IEEE_F32LE) > 0 || H5Tequal(file_type, H5T_NATIVE_FLOAT) > 0)
    {
        // 1. Allocate temporary buffer for standard 32-bit floats
        float *temp_f32 = malloc(sizeof(float) * (size_t)*m * *n);
        if (temp_f32 == NULL)
        {
            fprintf(stderr, "Failed to allocate temporary buffer for F32 parsing\n");
            free(Matrix);
            Matrix = NULL;
        }
        else
        {
            // 2. Read dataset out as standard native 32-bit floats
            H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp_f32);

            // 3. Convert elements inline to 16-bit floats using hardware intrinsics
            size_t total = (size_t)*m * *n;
            size_t blocks = total / 8;

#pragma omp parallel for
            for (size_t b = 0; b < blocks; b++)
                _mm_storeu_si128((__m128i *)(Matrix + b * 8),
                                 _mm256_cvtps_ph(_mm256_loadu_ps(temp_f32 + b * 8), 0));

            for (size_t i = blocks * 8; i < total; i++)
                Matrix[i] = _cvtss_sh(temp_f32[i], 0);

            free(temp_f32);
        }
    }
    else
    {
        // Fallback: If it's already a 16-bit raw representation, do a direct binary copy
        H5Dread(dataset, file_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, Matrix);
    }

    // H5Dread(dataset, file_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, Matrix);

    H5Dclose(dataset);
    H5Fclose(file);
    free(Dims);

    return Matrix;
}

int *matrix_load_h5_indices(const char *Path, const char *Dataset, int *m, int *n)
{
    hid_t file = H5Fopen(Path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (H5Lexists(file, Dataset, H5P_DEFAULT) == 0)
    {
        H5Fclose(file);
        return NULL;
    }
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

    int *Matrix = alloc64(sizeof(int) * (size_t)*m * *n);

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

// Helper function to create missing local directories (similar to mkdir -p)
void ensure_directory(const char *path)
{
    char tmp[512];
    char *p = NULL;
    size_t len;

    snprintf(tmp, sizeof(tmp), "%s", path);
    len = strlen(tmp);
    if (tmp[len - 1] == '/')
    {
        tmp[len - 1] = 0;
    }
    for (p = tmp + 1; *p; p++)
    {
        if (*p == '/')
        {
            *p = 0;
            mkdir(tmp, 0755);
            *p = '/';
        }
    }
    mkdir(tmp, 0755);
}

// Function to attach a string attribute at the root level
void write_root_string_attr(hid_t loc_id, const char *attr_name, const char *value)
{
    // 1. Strings require a Scalar dataspace structure
    hid_t dataspace_id = H5Screate(H5S_SCALAR);

    // 2. Derive a custom string datatype copying the standard C string baseline
    hid_t type_id = H5Tcopy(H5T_C_S1);
    H5Tset_size(type_id, strlen(value) + 1); // +1 handles the null terminator
    H5Tset_strpad(type_id, H5T_STR_NULLTERM);

    // 3. Create the attribute attached directly to the root location
    hid_t attr_id = H5Acreate2(loc_id, attr_name, type_id, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);

    if (attr_id >= 0)
    {
        // 4. Write the string buffer text downstream
        H5Awrite(attr_id, type_id, value);
        H5Aclose(attr_id);
    }
    else
    {
        fprintf(stderr, "Warning: Failed to write attribute %s\n", attr_name);
    }

    // 5. Free resources
    H5Tclose(type_id);
    H5Sclose(dataspace_id);
}

// Function to attach a float (double precision) attribute at the root level
void write_root_float_attr(hid_t loc_id, const char *attr_name, double value)
{
    hid_t dataspace_id = H5Screate(H5S_SCALAR);

    // Use standard 64-bit IEEE data layout rules
    hid_t attr_id = H5Acreate2(loc_id, attr_name, H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);

    if (attr_id >= 0)
    {
        // Map native local double storage parameters directly to the attribute layout
        H5Awrite(attr_id, H5T_NATIVE_DOUBLE, &value);
        H5Aclose(attr_id);
    }
    else
    {
        fprintf(stderr, "Warning: Failed to write attribute %s\n", attr_name);
    }

    H5Sclose(dataspace_id);
}

void matrix_store_h5_results(const char *Path, int m, int n, int *N, float *D, float t_build, float t_query, const char *Dataset, int p_qb, int p_qn, int p_k, int p_ln)
{
    hid_t file_id = H5Fcreate(Path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0)
    {
        fprintf(stderr, "Error creating HDF5 file.\n");
        return;
    }

    // 3. Define the 2D dimensions and create the DataSpaces
    hsize_t dims[2] = {m, n};
    hid_t dataspace_knns_id = H5Screate_simple(2, dims, NULL);

    // 4. Create the Datasets inside the file
    hid_t dataset_knns_id = H5Dcreate2(file_id, "knns", H5T_STD_I32LE,
                                       dataspace_knns_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // 5. Write the memory buffers to the file datasets
    herr_t status_knns = H5Dwrite(dataset_knns_id, H5T_NATIVE_INT32, H5S_ALL,
                                  H5S_ALL, H5P_DEFAULT, N);
    if (status_knns < 0)
    {
        fprintf(stderr, "Error writing matrix data to dataset.\n");
    }

    H5Dclose(dataset_knns_id);
    H5Sclose(dataspace_knns_id);

    hid_t dataspace_dists_id = H5Screate_simple(2, dims, NULL);

    hid_t dataset_dists_id = H5Dcreate2(file_id, "dists", H5T_IEEE_F32LE,
                                        dataspace_dists_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    herr_t status_dists = H5Dwrite(dataset_dists_id, H5T_NATIVE_FLOAT, H5S_ALL,
                                   H5S_ALL, H5P_DEFAULT, D);
    if (status_dists < 0)
    {
        fprintf(stderr, "Error writing matrix data to dataset.\n");
    }

    H5Dclose(dataset_dists_id);
    H5Sclose(dataspace_dists_id);

    char Params[512];
    sprintf(Params, "Qb=%d, Qn=%d, k=%d, Ln=%d", p_qb, p_qn, p_k, p_ln);

    write_root_string_attr(file_id, "algo", "PlainLocalSearch");
    write_root_string_attr(file_id, "task", "task1");
    write_root_string_attr(file_id, "params", Params);
    write_root_string_attr(file_id, "dataset", Dataset);
    write_root_float_attr(file_id, "buildtime", t_build);
    write_root_float_attr(file_id, "querytime", t_query);

    // 6. Close and release resources in reverse order of creation

    H5Fclose(file_id);

    printf("Matrix successfully written to %s\n", Path);
}