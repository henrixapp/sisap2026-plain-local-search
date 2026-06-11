#pragma once

#include <stdint.h>

uint16_t *matrix_load_h5_data(const char *Path, const char *Dataset, int *m, int *n);

int *matrix_load_h5_indices(const char *Path, const char *Dataset, int *m, int *n);

float compute_dot_product(const uint16_t *Matrix, int m, int n, int rowA, int rowB);

void matrix_store_h5_results(const char *Path, int m, int n, int *N, float *D, float t_build, float t_query, const char *Dataset);
