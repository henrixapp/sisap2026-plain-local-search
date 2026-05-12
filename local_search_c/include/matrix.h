#pragma once

#include <stdint.h>

uint16_t *matrix_load_h5_data(const char *Path, const char *Dataset, int *m, int *n);

int *matrix_load_h5_indices(const char *Path, const char *Dataset, int *m, int *n);

float compute_dot_product(const uint16_t *Matrix, int m, int n, int rowA, int rowB);

// returns a 16 * n matrix
float *compute_random_embedding(const uint16_t *Matrix, int m, int n,int k);