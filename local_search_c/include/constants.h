#pragma once
#define LOCAL_SEARCH_DIMS 1024
#define LOCAL_SEARCH_DIMS_IN_BYTE 128
#define LOCAL_SEARCH_DIMS_IN_64BIT 16
// #define LS_AVX_512_POP_COUNT 1 // uncomment if faster hamming distance should be used
#define LOCAL_SEARCH_SIZE_OF_NEIGHBORS 64 // 96 -->0.882826 330.407677
#define MAX_LEAF_SIZE 64