#pragma once

#include "knn.h"

#include <stdint.h>

/*
    The three hot passes exist in two flavours: a baseline x86-64-v3 (AVX2)
    build and an AVX-512 + VPOPCNTDQ build. Both are generated from
    src/kernels.inc and main picks between them once, at startup, based on a
    cpuid probe. Everything stays fully inlined inside each flavour, which a
    function pointer on the distance kernel could never achieve.
*/

#define KERNEL_DECLS(SUF)                                                                    \
    void knn_init_pass##SUF(KNN *g, const uint64_t *B, int m, int bn, int Qb, int Qn);       \
    void knn_build_order##SUF(const uint64_t *B, int m, int bn, int Qb, int *Ord);            \
    void knn_refine##SUF(KNN *g, const uint64_t *B, const int *Ord, int m, int bn, int Du,    \
                         int Dv, int sym);                                                    \
    void knn_rerank##SUF(KNN *g, const uint16_t *M, const int8_t *Q, const float *S,          \
                         const int *Ord, int m, int n, int R, int *N, float *Nd);

KERNEL_DECLS(_v3)
KERNEL_DECLS(_v4)

#undef KERNEL_DECLS
