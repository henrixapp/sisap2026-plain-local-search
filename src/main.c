#include "algorithms.h"
#include "alloc.h"
#include "kernels.h"
#include "knn.h"
#include "matrix.h"

#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
    const char *input_path;
    const char *output_path;
    const char *dataset_name;
    const char *train_key;
    const char *gt_key;

    int Qb;       // Base case size of the vantage point partition
    int Qn;       // Number of randomized partition rounds
    int k;        // Size of the approximate neighborhoods carried into reranking
    int Ln;       // Local search refinement rounds
    int Du;       // Neighbors of u expanded during local search
    int Dv;       // Neighbors of each of those expanded in turn
    int R;        // Shortlist kept by the int8 scan and rescored exactly
    int Ob;       // Leaf size of the partition that defines the processing order
    int Sy;       // Rounds that also insert the reverse edge
    int target_k; // Final number of nearest neighbors to output/evaluate

    uint32_t seed;
    int threads;
    int no_avx512;
    int quiet;
} Config;

// Dispatch table, bound once in main to the best flavour this CPU supports.
static void (*init_pass)(KNN *, const uint64_t *, int, int, int, int);
static void (*build_order)(const uint64_t *, int, int, int, int *);
static void (*refine)(KNN *, const uint64_t *, const int *, int, int, int, int, int);
static void (*rerank)(KNN *, const uint16_t *, const int8_t *, const float *, const int *, int,
                      int, int, int *, float *);

enum
{
    OPT_QB = 256,
    OPT_QN,
    OPT_LN,
    OPT_DU,
    OPT_DV,
    OPT_OB,
    OPT_SY,
    OPT_TRAIN_KEY,
    OPT_GT_KEY,
    OPT_NO_AVX512,
};

static void print_usage(FILE *out, const char *prog)
{
    fprintf(out,
            "Usage: %s [OPTIONS] <input.h5> [output.h5]\n"
            "\n"
            "Compute an approximate k-nearest neighbor graph using Plain Local Search.\n"
            "\n"
            "Algorithm options:\n"
            "      --qb <int>          Vantage-point partition base case size (default: 256)\n"
            "      --qn <int>          Number of randomized partition rounds (default: 12)\n"
            "  -k, --knn-k <int>       Approximate neighborhood size for reranking (default: 96)\n"
            "      --ln <int>          Local search refinement rounds (default: 2)\n"
            "      --du <int>          Neighbors of u expanded in local search (default: 32)\n"
            "      --dv <int>          Second-hop neighbors expanded in local search (default: 40)\n"
            "  -r, --shortlist <int>   Shortlist size kept by int8 scan for exact fp16 rescoring\n"
            "                          (default: 18, clamped to >= target-k)\n"
            "      --ob <int>          Leaf size for cache-friendly processing order (default: 64)\n"
            "      --sy <int>          Refinement rounds with symmetric insertion (default: 1)\n"
            "  -K, --target-k <int>    Final number of nearest neighbors to output (default: 15)\n"
            "\n"
            "I/O and runtime options:\n"
            "  -o, --output <path>     Output HDF5 file path (alternative to positional arg)\n"
            "  -d, --dataset <name>    Dataset name stored in output metadata (default: unknown)\n"
            "      --train-key <path>  HDF5 dataset path for input vectors (default: /train)\n"
            "      --gt-key <path>     HDF5 dataset path for ground truth (default: /allknn/knns)\n"
            "  -s, --seed <uint>       Global RNG seed (default: 42)\n"
            "  -t, --threads <int>     Number of OpenMP threads (default: OpenMP default)\n"
            "      --no-avx512         Disable AVX-512 kernels and force AVX2\n"
            "  -q, --quiet             Suppress all stdout output\n"
            "  -h, --help              Show this help message and exit\n",
            prog);
}

static int parse_int(const char *s, const char *opt_name, int min_val, int *out)
{
    errno = 0;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v < min_val || v > INT_MAX)
    {
        fprintf(stderr, "Invalid value for %s: '%s' (expected integer >= %d)\n", opt_name, s,
                min_val);
        return -1;
    }
    *out = (int)v;
    return 0;
}

static int parse_uint32(const char *s, const char *opt_name, uint32_t *out)
{
    errno = 0;
    char *end = NULL;
    if (*s == '-')
    {
        fprintf(stderr, "Invalid value for %s: '%s' (expected unsigned 32-bit integer)\n", opt_name,
                s);
        return -1;
    }
    unsigned long v = strtoul(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v > UINT32_MAX)
    {
        fprintf(stderr, "Invalid value for %s: '%s' (expected unsigned 32-bit integer)\n", opt_name,
                s);
        return -1;
    }
    *out = (uint32_t)v;
    return 0;
}

// Returns 0 on success, 1 if --help was requested, -1 on error.
static int parse_args(int argc, char **argv, Config *cfg)
{
    static const struct option long_opts[] = {
        {"qb", required_argument, NULL, OPT_QB},
        {"qn", required_argument, NULL, OPT_QN},
        {"knn-k", required_argument, NULL, 'k'},
        {"ln", required_argument, NULL, OPT_LN},
        {"du", required_argument, NULL, OPT_DU},
        {"dv", required_argument, NULL, OPT_DV},
        {"shortlist", required_argument, NULL, 'r'},
        {"ob", required_argument, NULL, OPT_OB},
        {"sy", required_argument, NULL, OPT_SY},
        {"target-k", required_argument, NULL, 'K'},
        {"output", required_argument, NULL, 'o'},
        {"dataset", required_argument, NULL, 'd'},
        {"train-key", required_argument, NULL, OPT_TRAIN_KEY},
        {"gt-key", required_argument, NULL, OPT_GT_KEY},
        {"seed", required_argument, NULL, 's'},
        {"threads", required_argument, NULL, 't'},
        {"no-avx512", no_argument, NULL, OPT_NO_AVX512},
        {"quiet", no_argument, NULL, 'q'},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0},
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "k:r:K:o:d:s:t:qh", long_opts, NULL)) != -1)
    {
        switch (opt)
        {
        case OPT_QB:
            if (parse_int(optarg, "--qb", 1, &cfg->Qb) < 0)
                return -1;
            break;
        case OPT_QN:
            if (parse_int(optarg, "--qn", 1, &cfg->Qn) < 0)
                return -1;
            break;
        case 'k':
            if (parse_int(optarg, "-k/--knn-k", 1, &cfg->k) < 0)
                return -1;
            break;
        case OPT_LN:
            if (parse_int(optarg, "--ln", 0, &cfg->Ln) < 0)
                return -1;
            break;
        case OPT_DU:
            if (parse_int(optarg, "--du", 1, &cfg->Du) < 0)
                return -1;
            break;
        case OPT_DV:
            if (parse_int(optarg, "--dv", 1, &cfg->Dv) < 0)
                return -1;
            break;
        case 'r':
            if (parse_int(optarg, "-r/--shortlist", 1, &cfg->R) < 0)
                return -1;
            break;
        case OPT_OB:
            if (parse_int(optarg, "--ob", 1, &cfg->Ob) < 0)
                return -1;
            break;
        case OPT_SY:
            if (parse_int(optarg, "--sy", 0, &cfg->Sy) < 0)
                return -1;
            break;
        case 'K':
            if (parse_int(optarg, "-K/--target-k", 1, &cfg->target_k) < 0)
                return -1;
            break;
        case 'o':
            cfg->output_path = optarg;
            break;
        case 'd':
            cfg->dataset_name = optarg;
            break;
        case OPT_TRAIN_KEY:
            cfg->train_key = optarg;
            break;
        case OPT_GT_KEY:
            cfg->gt_key = optarg;
            break;
        case 's':
            if (parse_uint32(optarg, "-s/--seed", &cfg->seed) < 0)
                return -1;
            break;
        case 't':
            if (parse_int(optarg, "-t/--threads", 1, &cfg->threads) < 0)
                return -1;
            break;
        case OPT_NO_AVX512:
            cfg->no_avx512 = 1;
            break;
        case 'q':
            cfg->quiet = 1;
            break;
        case 'h':
            print_usage(stdout, argv[0]);
            return 1;
        default:
            print_usage(stderr, argv[0]);
            return -1;
        }
    }

    int remaining = argc - optind;
    if (remaining < 1)
    {
        fprintf(stderr, "Error: missing required <input.h5> argument.\n\n");
        print_usage(stderr, argv[0]);
        return -1;
    }

    cfg->input_path = argv[optind];
    if (remaining >= 2)
    {
        if (cfg->output_path != NULL)
        {
            fprintf(stderr, "Error: output path specified both via -o/--output and positional "
                            "argument.\n");
            return -1;
        }
        cfg->output_path = argv[optind + 1];
    }
    if (remaining > 2)
    {
        fprintf(stderr, "Error: unexpected extra positional argument '%s'.\n\n",
                argv[optind + 2]);
        print_usage(stderr, argv[0]);
        return -1;
    }

    if (cfg->R < cfg->target_k)
        cfg->R = cfg->target_k;
    if (cfg->k < cfg->target_k)
        cfg->k = cfg->target_k;

    return 0;
}

#define LOG(...)               \
    do                         \
    {                          \
        if (!cfg.quiet)        \
            printf(__VA_ARGS__); \
    } while (0)

int main(int argc, char **argv)
{
    Config cfg = {
        .input_path = NULL,
        .output_path = NULL,
        .dataset_name = "unknown",
        .train_key = "/train",
        .gt_key = "/allknn/knns",
        .Qb = 256,
        .Qn = 12,
        .k = 96,
        .Ln = 2,
        .Du = 32,
        .Dv = 40,
        .R = 18,
        .Ob = 64,
        .Sy = 1,
        .target_k = 15,
        .seed = 42,
        .threads = 0,
        .no_avx512 = 0,
        .quiet = 0,
    };

    int parse_status = parse_args(argc, argv, &cfg);
    if (parse_status > 0)
        return 0;
    if (parse_status < 0)
        return 1;

    if (cfg.threads > 0)
        omp_set_num_threads(cfg.threads);

    LOG("%s\n", cfg.input_path);

    if (cpu_has_avx512_popcnt() && !cfg.no_avx512)
    {
        init_pass = knn_init_pass_v4;
        build_order = knn_build_order_v4;
        refine = knn_refine_v4;
        rerank = knn_rerank_v4;
        LOG("Using AVX-512 kernels\n");
    }
    else
    {
        init_pass = knn_init_pass_v3;
        build_order = knn_build_order_v3;
        refine = knn_refine_v3;
        rerank = knn_rerank_v3;
        LOG("Using AVX2 kernels\n");
    }

    int n = 0, m = 0;
    uint16_t *Matrix = matrix_load_h5_data(cfg.input_path, cfg.train_key, &m, &n);
    if (Matrix == NULL)
        return 1;

    int kn = 0, km = 0;
    int *KNNs = matrix_load_h5_indices(cfg.input_path, cfg.gt_key, &km, &kn);
    if (KNNs != NULL && km != m)
    {
        free(KNNs);
        KNNs = NULL;
    }

    LOG("%d %d\n", n, m);
    if (KNNs != NULL)
        LOG("%d %d\n", kn, km);

    prepare_random(cfg.seed);

    KNN *g = KNN_init(m, cfg.k, n);

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

    LOG("Quantization in %lf\n", t1 - t0);

    // 2. Initialize ANN with randomized vantage point partitions

    init_pass(g, Matrix_bin, m, bn, cfg.Qb, cfg.Qn);

    // Order the remaining passes for cache reuse
    int *Ord = malloc(sizeof(int) * m);
    build_order(Matrix_bin, m, bn, cfg.Ob, Ord);

    double t2 = omp_get_wtime();

    LOG("Initialized ANN in %lf\n", t2 - t1);

    if (KNNs != NULL && !cfg.quiet)
        printf("Recall %f\n", KNN_compute_recall(g, KNNs, km, kn, cfg.target_k));

    // 3. Local search refinement

    for (int i = 0; i < cfg.Ln; i++)
    {
        double tt = omp_get_wtime();

        // Only the first round pays for symmetric insertion: it is what turns
        // the sparse initial sketch into a connected graph. Later rounds find
        // almost nothing that way and the extra locking is not worth it.
        refine(g, Matrix_bin, Ord, m, bn, cfg.Du, cfg.Dv, i < cfg.Sy);

        double te = omp_get_wtime();
        if (KNNs != NULL && !cfg.quiet)
            printf("It %d: Recall %f %lf\n", i + 1,
                   KNN_compute_recall(g, KNNs, km, kn, cfg.target_k), te - tt);
    }

    double t3 = omp_get_wtime();

    LOG("Finished refinement in %lf\n", t3 - t2);

    // 4. Rerank the candidates with exact fp16 inner products

    int *Shortlist = alloc64(sizeof(int) * (size_t)g->n * cfg.R);
    float *Shortlist_d = alloc64(sizeof(float) * (size_t)g->n * cfg.R);

#pragma omp parallel for
    for (size_t i = 0; i < (size_t)g->n * cfg.R; i++)
    {
        Shortlist[i] = 0;
        Shortlist_d[i] = 0.0f;
    }

    rerank(g, Matrix, Q, S, Ord, m, n, cfg.R, Shortlist, Shortlist_d);

    // Keep the target_k best of every shortlist
    int *k_NN = malloc(sizeof(int) * (size_t)g->n * cfg.target_k);
    float *k_NN_d = malloc(sizeof(float) * (size_t)g->n * cfg.target_k);

#pragma omp parallel for
    for (int u = 0; u < g->n; u++)
        for (int i = 0; i < cfg.target_k; i++)
        {
            k_NN[(size_t)u * cfg.target_k + i] = Shortlist[(size_t)u * cfg.R + i];
            k_NN_d[(size_t)u * cfg.target_k + i] = Shortlist_d[(size_t)u * cfg.R + i];
        }

    free(Shortlist);
    free(Shortlist_d);

    double t4 = omp_get_wtime();

    LOG("Finished reranking in %lf\n", t4 - t3);
    LOG("Total %lf\n", t4 - t0);

    if (KNNs != NULL && !cfg.quiet)
        printf("Recall %f\n", KNN_compute_recall_standalone(k_NN, KNNs, km, kn, cfg.target_k));

    int exit_code = 0;
    if (cfg.output_path != NULL)
    {
        char params[512];
        snprintf(params, sizeof(params),
                 "Qb=%d, Qn=%d, k=%d, Ln=%d, Du=%d, Dv=%d, R=%d, Ob=%d, Sy=%d, target_k=%d, "
                 "seed=%u",
                 cfg.Qb, cfg.Qn, cfg.k, cfg.Ln, cfg.Du, cfg.Dv, cfg.R, cfg.Ob, cfg.Sy,
                 cfg.target_k, cfg.seed);

        if (matrix_store_h5_results(cfg.output_path, m, cfg.target_k, k_NN, k_NN_d,
                                    (float)(t2 - t0), (float)(t4 - t2), cfg.dataset_name,
                                    params) == 0)
            LOG("Matrix successfully written to %s\n", cfg.output_path);
        else
            exit_code = 1;
    }

    free(Matrix);
    free(Matrix_bin);
    free(Ord);
    free(Q);
    free(S);
    free(KNNs);
    free(k_NN);
    free(k_NN_d);
    KNN_free(g);

    return exit_code;
}
