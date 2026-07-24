#include "kernels.h"

#include "algorithms.h"
#include "knn.h"
#include "matrix.h"

#include <stdlib.h>
#include <string.h>

// ---- baseline x86-64-v3 flavour -------------------------------------------

#define SUF _v3
#define TGT
#define HAM(a, b, n) hamming_avx2_lut((a), (b), (n))
#define DOT(a, b, n) dot_fp16_avx2((a), (b), (n))
#define DOTI8(a, b, n) dot_i8_avx2((a), (b), (n))
#include "kernels.inc"
#undef SUF
#undef TGT
#undef HAM
#undef DOT
#undef DOTI8

// ---- AVX-512 / VPOPCNTDQ flavour ------------------------------------------

#define SUF _v4
#define TGT __attribute__((target("avx512f,avx512bw,avx512vl,avx512vpopcntdq")))
#define HAM(a, b, n) hamming_avx512((a), (b), (n))
#define DOT(a, b, n) dot_fp16_avx512((a), (b), (n))
#define DOTI8(a, b, n) dot_i8_avx512((a), (b), (n))
#include "kernels.inc"
#undef SUF
#undef TGT
#undef HAM
#undef DOT
#undef DOTI8
