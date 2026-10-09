#include "algorithms.h"
#include "knn.h"
#include "matrix.h"

#include <fcntl.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int Qb = 264;
int Qn = 32;
int k = 64;
int Ln = 2;

/*
    Everything below is overridable from the environment for experiments; the
    defaults are the tuned configuration.

      KNN_QB / KNN_QN / KNN_K / KNN_LN   leaf size, tree count, k, refinement rounds
      KNN_RR                             candidates to rerank (0 = all k)
      KNN_RERANK_F16                     1 = exact fp16 rerank, 0 = int8 codes
      KNN_SEQ                            1 = plain sweep in refinement round 1
      KNN_NOPERM                         1 = skip the spatial relabelling
      KNN_CEILING=<n>                    diagnostic: brute-force the exact top-k by
                                         hamming for n sampled points, print the
                                         recall that bounds this pipeline, and exit
      KNN_CACHE=<path>                   development only: raw cache of the parsed
                                         input, so repeated runs skip the HDF5
                                         parse. Loading is outside the timed
                                         sections either way.
*/

void *knn_alloc_huge(size_t bytes);

static double eval_time = 0.0;

static int env_int(const char *name, int fallback)
{
    const char *s = getenv(name);
    return s == NULL ? fallback : atoi(s);
}

/*
    Qsort-ish indexing:
    - pick random pivot point
    - partition data around median distance from pivot
    - recurse on each half
    - when a segment becomes small, points are likely neighbors

    The partitioning and the all-pairs leaf work are split into two phases. All
    leaves of one tree are disjoint point sets, so the leaf phase needs no locks
    at all -- the previous version ran whole trees concurrently, which forced
    every insert through the lock table.
*/

typedef struct
{
    int start;
    int len;
} Leaf;

typedef struct
{
    const uint64_t *M;
    int bn;
    VDEntry *O;
    Leaf *leaves;
    int nleaves;
    int cap;
    int qb;
} TreeCtx;

static void tree_partition(TreeCtx *T, int base, int m)
{
    if (m < T->qb)
    {
        int idx;
#pragma omp atomic capture
        idx = T->nleaves++;
        if (idx < T->cap)
        {
            T->leaves[idx].start = base;
            T->leaves[idx].len = m;
        }
        return;
    }

    const int bn = T->bn;
    const uint64_t *M = T->M;
    VDEntry *O = T->O + base;

    // Random pivot element
    const uint64_t *pivot = M + (size_t)O[fast_rand() % m].v * bn;

    // Compute hamming distances for all points to pivot. The rows are scattered
    // across 813 MB and the ids are all known up front, so run the loads ahead.
#define PF_PART 12
    int dmin = 1 << 30, dmax = -1;
    for (int i = 0; i < PF_PART && i < m; i++)
    {
        const char *pf = (const char *)(M + (size_t)O[i].v * bn);
        _mm_prefetch(pf, _MM_HINT_T0);
        _mm_prefetch(pf + 64, _MM_HINT_T0);
    }
    for (int i = 0; i < m; i++)
    {
        if (i + PF_PART < m)
        {
            const char *pf = (const char *)(M + (size_t)O[i + PF_PART].v * bn);
            _mm_prefetch(pf, _MM_HINT_T0);
            _mm_prefetch(pf + 64, _MM_HINT_T0);
        }

        int d = hamming_avx2_lut(pivot, M + (size_t)O[i].v * bn, bn);
        O[i].d = d;
        if (d < dmin)
            dmin = d;
        if (d > dmax)
            dmax = d;
    }

    // Histogram over the observed range only; distances concentrate hard in high
    // dimension, so this is a small window out of the 0..bn*64 space.
    int C[1088];
    for (int i = dmin; i <= dmax; i++)
        C[i] = 0;
    for (int i = 0; i < m; i++)
        C[O[i].d]++;

    // Find median distance
    int p = 0, pd = dmax;
    for (int i = dmin; i <= dmax; i++)
    {
        p += C[i];
        if (p >= m / 2)
        {
            pd = i;
            break;
        }
    }

    // Partition data
    p = 0;
    for (int i = 0; i < m; i++)
    {
        if (O[i].d >= pd)
        {
            VDEntry t = O[p];
            O[p] = O[i];
            O[i] = t;
            p++;
        }
    }

    // Protect against infinite recursion
    if (p == 0 || p == m)
        p = m / 2;

    if (m > 200000)
    {
#pragma omp task firstprivate(base, p)
        tree_partition(T, base, p);
        tree_partition(T, base + p, m - p);
    }
    else
    {
        tree_partition(T, base, p);
        tree_partition(T, base + p, m - p);
    }
}

static int build_tree(TreeCtx *T, int m)
{
    T->nleaves = 0;
#pragma omp parallel
    {
#pragma omp single
        tree_partition(T, 0, m);
    }
    return T->nleaves < T->cap ? T->nleaves : T->cap;
}

/*
    All pairs inside one leaf. The codes are gathered into a scratch buffer first
    so the inner loop streams over ~30 KB instead of chasing 128-byte rows spread
    across 813 MB.
*/
static void process_leaf(const uint64_t *M, int bn, const VDEntry *O, int len, KNN *g, uint64_t *buf)
{
    for (int i = 0; i < len; i++)
    {
        if (i + 8 < len)
        {
            const char *pf = (const char *)(M + (size_t)O[i + 8].v * bn);
            _mm_prefetch(pf, _MM_HINT_T0);
            _mm_prefetch(pf + 64, _MM_HINT_T0);
        }
        memcpy(buf + (size_t)i * bn, M + (size_t)O[i].v * bn, (size_t)bn * 8);
    }

    for (int i = 0; i < len; i++)
    {
        const uint64_t *a = buf + (size_t)i * bn;
        int u = O[i].v;

        int j = i + 1;
        for (; j + 1 < len; j += 2)
        {
            int d0, d1;
            hamming_avx2_lut_x2(a, buf + (size_t)j * bn, buf + (size_t)(j + 1) * bn, bn, &d0, &d1);

            int v0 = O[j].v, v1 = O[j + 1].v;

            if (d0 < g->Dmax[u])
                KNN_offer_max0(g, u, v0, d0);
            if (d0 < g->Dmax[v0])
                KNN_offer_max0(g, v0, u, d0);
            if (d1 < g->Dmax[u])
                KNN_offer_max0(g, u, v1, d1);
            if (d1 < g->Dmax[v1])
                KNN_offer_max0(g, v1, u, d1);
        }
        for (; j < len; j++)
        {
            int d = hamming_avx2_lut(a, buf + (size_t)j * bn, bn);
            int v = O[j].v;

            if (d < g->Dmax[u])
                KNN_offer_max0(g, u, v, d);
            if (d < g->Dmax[v])
                KNN_offer_max0(g, v, u, d);
        }
    }
}

/*
    Rmin mirrors the 15th-best similarity of every point in a compact array. The
    reverse insert has to test that value for each candidate, and reading it out
    of the full result table pulls a cache line from 381 MB; here it is 4 bytes
    out of 25 MB.
*/
static inline void insert_safe_rerank(KNN *g, int *N, float *Nd, float *Rmin, int u, int v, float d)
{
    knn_lock(&g->L[u & 8191]);

    if (d < Nd[14])
        goto end;

    for (int i = 0; i < 15; i++)
        if (N[i] == v + 1)
            goto end;

    int p = 13;
    while (p >= 0 && Nd[p] < d)
    {
        N[p + 1] = N[p];
        Nd[p + 1] = Nd[p];
        p--;
    }
    N[p + 1] = v + 1;
    Nd[p + 1] = d;
    Rmin[u] = Nd[14];

end:
    knn_unlock(&g->L[u & 8191]);
}

static void report(KNN *g, const int *KNNs, int km, int kn, const char *tag, double t)
{
    if (KNNs == NULL)
    {
        printf("%s in %lf\n", tag, t);
        fflush(stdout);
        return;
    }
    double e0 = omp_get_wtime();
    float recall = KNN_compute_recall(g, KNNs, km, kn, 15);
    eval_time += omp_get_wtime() - e0;
    printf("%s in %lf (candidate recall %f)\n", tag, t, recall);
    fflush(stdout);
}

/* ---- development helper: raw cache of the parsed input ------------------- */

static int read_full(int fd, void *p, size_t n)
{
    char *c = p;
    while (n > 0)
    {
        ssize_t r = read(fd, c, n > (1u << 30) ? (1u << 30) : n);
        if (r <= 0)
            return 0;
        c += r;
        n -= r;
    }
    return 1;
}

static void write_full(int fd, const void *p, size_t n)
{
    const char *c = p;
    while (n > 0)
    {
        ssize_t r = write(fd, c, n > (1u << 30) ? (1u << 30) : n);
        if (r <= 0)
            return;
        c += r;
        n -= r;
    }
}

int main(int argc, char **argv)
{
    printf("%s\n", argv[1]);

    Qb = env_int("KNN_QB", Qb);
    Qn = env_int("KNN_QN", Qn);
    k = env_int("KNN_K", k);
    Ln = env_int("KNN_LN", Ln);
    int rerank_f16 = env_int("KNN_RERANK_F16", 1);
    int seq_pass1 = env_int("KNN_SEQ", 0);
    int rr = env_int("KNN_RR", 0); // candidates to rerank, 0 = all of them

    int n = 0, m = 0, kn = 0, km = 0;
    uint16_t *Matrix = NULL;
    int *KNNs = NULL;

    const char *cache = getenv("KNN_CACHE");
    int cfd = cache != NULL ? open(cache, O_RDONLY) : -1;

    double tl = omp_get_wtime();

    if (cfd >= 0)
    {
        int hdr[4];
        read_full(cfd, hdr, sizeof(hdr));
        m = hdr[0], n = hdr[1], km = hdr[2], kn = hdr[3];
        Matrix = knn_alloc_huge(sizeof(uint16_t) * (size_t)m * n);
        read_full(cfd, Matrix, sizeof(uint16_t) * (size_t)m * n);
        if (km > 0)
        {
            KNNs = knn_alloc_huge(sizeof(int) * (size_t)km * kn);
            read_full(cfd, KNNs, sizeof(int) * (size_t)km * kn);
        }
        close(cfd);
    }
    else
    {
        Matrix = matrix_load_h5_data(argv[1], "/train", &m, &n);
        KNNs = matrix_load_h5_indices(argv[1], "/allknn/knns", &km, &kn);

        if (cache != NULL)
        {
            int fd = open(cache, O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd >= 0)
            {
                int hdr[4] = {m, n, KNNs ? km : 0, KNNs ? kn : 0};
                write_full(fd, hdr, sizeof(hdr));
                write_full(fd, Matrix, sizeof(uint16_t) * (size_t)m * n);
                if (KNNs)
                    write_full(fd, KNNs, sizeof(int) * (size_t)km * kn);
                close(fd);
            }
        }
    }

    printf("%d %d\n", n, m);
    printf("%d %d\n", kn, km);
    printf("Loaded input in %lf\n", omp_get_wtime() - tl);
    fflush(stdout);

    prepare_random();

    // Finished parsing and allocation

    // 1. Quantization to 1 bit
    // Currently this step assumes that n / 8 is an integer

    double t0 = omp_get_wtime();

    int bn = (n + 63) / 64;
    if ((bn % 4) > 0)
        bn += 4 - (bn % 4);

    // One spare row so a sentinel neighbour id can be dereferenced harmlessly
    uint64_t *Matrix_bin = knn_alloc_huge(sizeof(uint64_t) * ((size_t)m + 1) * bn);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < ((size_t)m + 1) * bn; i++)
        Matrix_bin[i] = 0;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++)
        quantize_8_fp16_avx2(Matrix + (size_t)i * n, (uint8_t *)(Matrix_bin + (size_t)i * bn), n);

    double t1 = omp_get_wtime();

    printf("1-bit quantization in %lf\n", t1 - t0);
    fflush(stdout);

    /*
        2. Relabel the points along a spatial ordering.

        Refinement and reranking both walk hundreds of neighbour candidates per
        point, each a random row of an 813 MB / 13 GB table, and are limited by
        memory latency rather than arithmetic. Reordering the data so that points
        close in space are close in memory turns many of those into cache hits.
        The ordering is the leaf order of one extra partition pass, which costs a
        fraction of the all-pairs leaf work it later feeds.
    */

    VDEntry *O = knn_alloc_huge(sizeof(VDEntry) * (size_t)m);
    int leaf_cap = 16 * m / (Qb > 32 ? Qb : 32) + 4096;
    Leaf *leaves = knn_alloc_huge(sizeof(Leaf) * (size_t)leaf_cap);

    TreeCtx T = {.M = Matrix_bin, .bn = bn, .O = O, .leaves = leaves, .cap = leaf_cap, .qb = Qb};

#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++)
        O[i].v = i;

    if (!env_int("KNN_NOPERM", 0))
        build_tree(&T, m);

    int *perm = knn_alloc_huge(sizeof(int) * (size_t)m);
    int *iperm = knn_alloc_huge(sizeof(int) * (size_t)m);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++)
    {
        perm[i] = O[i].v;
        iperm[O[i].v] = i;
    }

    uint64_t *Matrix_bin_p = knn_alloc_huge(sizeof(uint64_t) * ((size_t)m + 1) * bn);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++)
        memcpy(Matrix_bin_p + (size_t)i * bn, Matrix_bin + (size_t)perm[i] * bn, (size_t)bn * 8);

    for (int i = 0; i < bn; i++)
        Matrix_bin_p[(size_t)m * bn + i] = 0;

    free(Matrix_bin);
    Matrix_bin = Matrix_bin_p;

    // Codes for reranking, built directly in the new order.
    int8_t *Matrix_i8 = NULL;
    float *Scale = NULL;
    uint16_t *Matrix_p = NULL;

    if (rerank_f16)
    {
        Matrix_p = knn_alloc_huge(sizeof(uint16_t) * (size_t)m * n);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++)
            memcpy(Matrix_p + (size_t)i * n, Matrix + (size_t)perm[i] * n, sizeof(uint16_t) * n);
    }
    else
    {
        Matrix_i8 = knn_alloc_huge(sizeof(int8_t) * (size_t)m * n);
        Scale = knn_alloc_huge(sizeof(float) * (size_t)m);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++)
            Scale[i] = quantize_i8_fp16_avx2(Matrix + (size_t)perm[i] * n, Matrix_i8 + (size_t)i * n, n);
    }
    free(Matrix);
    Matrix = NULL;

    double t1b = omp_get_wtime();
    printf("Reordering + rerank codes in %lf\n", t1b - t1);
    fflush(stdout);

    // Ground truth is only read by the recall checks; move it into the new
    // labelling once, outside the measured sections.
    if (KNNs != NULL)
    {
        double e0 = omp_get_wtime();
        int *KP = knn_alloc_huge(sizeof(int) * (size_t)km * kn);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < km; i++)
            for (int l = 0; l < kn; l++)
            {
                int v = KNNs[(size_t)perm[i] * kn + l];
                KP[(size_t)i * kn + l] = v > 0 ? iperm[v - 1] + 1 : v;
            }
        free(KNNs);
        KNNs = KP;
        eval_time += omp_get_wtime() - e0;
    }

    free(iperm);
    iperm = NULL;

    KNN *g = KNN_init(m, k, n);

    /*
        Diagnostic: brute-force the exact top-k by hamming for a sample of points
        and score it against the ground truth. That is the ceiling this pipeline
        can reach -- the candidate set is chosen in the 1-bit space, so whatever
        the binary codes cannot separate is lost no matter how good the search is.
    */
    int ceil_sample = env_int("KNN_CEILING", 0);
    if (ceil_sample > 0 && KNNs != NULL)
    {
        double tc = omp_get_wtime();
        long tp = 0;
        const int K = g->k;
        const long stride = m / ceil_sample;

#pragma omp parallel for schedule(dynamic, 1) reduction(+ : tp)
        for (int s = 0; s < ceil_sample; s++)
        {
            int u = (int)(s * stride);
            const uint64_t *a = Matrix_bin + (size_t)u * bn;
            int bd[256], bv[256];
            for (int i = 0; i < K; i++)
            {
                bd[i] = 1 << 20;
                bv[i] = -1;
            }
            int worst = 1 << 20;

            for (int x = 0; x < m; x++)
            {
                if (x == u)
                    continue;
                int d = hamming_avx2_lut(a, Matrix_bin + (size_t)x * bn, bn);
                if (d >= worst)
                    continue;
                int p = 0;
                for (int i = 1; i < K; i++)
                    if (bd[i] > bd[p])
                        p = i;
                bd[p] = d;
                bv[p] = x;
                worst = bd[0];
                for (int i = 1; i < K; i++)
                    if (bd[i] > worst)
                        worst = bd[i];
            }

            for (int l = 0; l < 15; l++)
            {
                int t = KNNs[(size_t)u * kn + l] - 1;
                if (t == u)
                    t = KNNs[(size_t)u * kn + 15] - 1;
                if (t < 0)
                    continue;
                for (int i = 0; i < K; i++)
                    if (bv[i] == t)
                    {
                        tp++;
                        break;
                    }
            }
        }

        printf("CEILING: exact top-%d hamming over %d sampled points -> recall %f (%.1lfs)\n",
               K, ceil_sample, (double)tp / ((double)ceil_sample * 15), omp_get_wtime() - tc);
        fflush(stdout);
        return 0;
    }

    // 3. Initialize ANN with qsort idea

    T.M = Matrix_bin;

    double t_part = 0.0, t_leaf = 0.0;

    for (int t = 0; t < Qn; t++)
    {
        T.qb = Qb - t;

        double ta = omp_get_wtime();

#pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++)
            O[i].v = i;

        int nl = build_tree(&T, m);
        if (T.nleaves > T.cap)
            fprintf(stderr, "WARNING: leaf list overflow (%d > %d), coverage lost\n", T.nleaves, T.cap);

        double tb = omp_get_wtime();

#pragma omp parallel
        {
            uint64_t *buf = aligned_alloc(64, sizeof(uint64_t) * (size_t)T.qb * bn);
#pragma omp for schedule(dynamic, 4)
            for (int l = 0; l < nl; l++)
                process_leaf(Matrix_bin, bn, O + leaves[l].start, leaves[l].len, g, buf);
            free(buf);
        }

        double tc = omp_get_wtime();
        t_part += tb - ta;
        t_leaf += tc - tb;
    }

    printf("  (tree partitioning %lf, leaf all-pairs %lf)\n", t_part, t_leaf);
    fflush(stdout);

    free(O);
    O = NULL;
    free(leaves);
    leaves = NULL;

    // Refinement walks the lists in distance order, indexing was allowed to keep
    // them unsorted.
    KNN_sort_lists(g);

    double t2 = omp_get_wtime();

    report(g, KNNs, km, kn, "Initialized ANN", t2 - t1b);

    // 4. Local search refinement

    {
        double tt = omp_get_wtime();

#pragma omp parallel
        {
            uint64_t *seen = calloc(KNN_SEEN_SIZE, sizeof(uint64_t));

            if (seq_pass1)
            {
                // Straight sweep: with the spatial relabelling, consecutive u share
                // most of their candidate set, so the working set stays hot.
#pragma omp for schedule(dynamic, 256)
                for (int u = 0; u < g->n; u++)
                    KNN_greedy_dist1_update_simple(g, u, Matrix_bin, m, bn, seen);
            }
            else
            {
                const int target = g->n / 2;
#pragma omp for schedule(dynamic, 256)
                for (int u = 0; u < target; u++)
                {
                    KNN_greedy_dist1_update_simple(g, u, Matrix_bin, m, bn, seen);
                    KNN_greedy_dist1_update_simple(g, g->n - 1 - u, Matrix_bin, m, bn, seen);
                }
            }

            free(seen);
        }
        report(g, KNNs, km, kn, "It 1", omp_get_wtime() - tt);
    }

    for (int i = 1; i < Ln; i++)
    {
        double tt = omp_get_wtime();

#pragma omp parallel
        {
            uint64_t *seen = calloc(KNN_SEEN_SIZE, sizeof(uint64_t));
#pragma omp for schedule(dynamic, 256)
            for (int u = 0; u < g->n; u++)
            {
                KNN_greedy_dist1_update_simple_round2(g, u, Matrix_bin, m, bn, seen);
            }
            free(seen);
        }
        char tag[32];
        snprintf(tag, sizeof(tag), "It %d", i + 1);
        report(g, KNNs, km, kn, tag, omp_get_wtime() - tt);
    }

    double t3 = omp_get_wtime();

    printf("Finished refinement in %lf\n", t3 - t2 - eval_time);
    fflush(stdout);

    free(Matrix_bin);
    Matrix_bin = NULL;
    free(g->D);
    g->D = NULL;
    free(g->Dmax);
    g->Dmax = NULL;

    // 5. Rerank the candidates with the exact (fp16) inner product

    int *k_NN = knn_alloc_huge(sizeof(int) * (size_t)g->n * 15);
    float *k_NN_d = knn_alloc_huge(sizeof(float) * (size_t)g->n * 15);
    float *Rmin = knn_alloc_huge(sizeof(float) * (size_t)g->n);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < (size_t)g->n * 15; i++)
    {
        k_NN[i] = 0;
        k_NN_d[i] = -1.0e30f;
    }

#pragma omp parallel for schedule(static)
    for (int i = 0; i < g->n; i++)
        Rmin[i] = -1.0e30f;

#pragma omp parallel for schedule(dynamic, 256)
    for (int u = 0; u < g->n; u++)
    {
        int *N = k_NN + ((size_t)u * 15);
        float *Nd = k_NN_d + ((size_t)u * 15);
        const int8_t *au = rerank_f16 ? NULL : Matrix_i8 + (size_t)u * n;
        const uint16_t *fu = rerank_f16 ? Matrix_p + (size_t)u * n : NULL;
        const float su = rerank_f16 ? 1.0f : Scale[u];
        const int *Vu = g->V + (size_t)u * g->k;
        const int nrr = (rr > 0 && rr < g->k) ? rr : g->k;

        // Scoring the candidates in index order instead of distance order was
        // measured and did not pay for itself: after the relabelling the band of
        // indices is still wide enough that the sort cost more than it saved.
        for (int i = 0; i < nrr; i++)
        {
            int v = Vu[i];
            if (v >= g->n)
                continue;

            if (i + 3 < nrr && Vu[i + 3] < g->n)
            {
                int vp = Vu[i + 3];
                const char *p = rerank_f16 ? (const char *)(Matrix_p + (size_t)vp * n)
                                           : (const char *)(Matrix_i8 + (size_t)vp * n);
                size_t sz = rerank_f16 ? sizeof(uint16_t) * (size_t)n : (size_t)n;
                for (size_t off = 0; off < sz; off += 128)
                    _mm_prefetch(p + off, _MM_HINT_T0);
            }

            float d = rerank_f16 ? dot_f16_avx2(fu, Matrix_p + (size_t)v * n, n)
                                 : (float)dot_i8_avx2(au, Matrix_i8 + (size_t)v * n, n) * su * Scale[v];

            if (d > Nd[14])
                insert_safe_rerank(g, N, Nd, Rmin, u, v, d);

            if (d > Rmin[v])
                insert_safe_rerank(g, k_NN + ((size_t)v * 15), k_NN_d + ((size_t)v * 15), Rmin, v, u, d);
        }
    }

    double t4 = omp_get_wtime();

    printf("Finished reranking in %lf\n", t4 - t3);
    printf("Total (build + query) %lf\n", t4 - t0 - eval_time);
    fflush(stdout);

    if (KNNs != NULL)
        printf("Recall %f\n", KNN_compute_recall_standalone(k_NN, KNNs, km, kn, 15));

    free(Rmin);
    Rmin = NULL;
    free(Matrix_p);
    Matrix_p = NULL;
    free(Matrix_i8);
    Matrix_i8 = NULL;
    free(Scale);
    Scale = NULL;
    KNN_free(g);
    g = NULL;

    // Back to the input labelling for the output file
    int *out_NN = knn_alloc_huge(sizeof(int) * (size_t)m * 15);
    float *out_D = knn_alloc_huge(sizeof(float) * (size_t)m * 15);

#pragma omp parallel for schedule(static)
    for (int u = 0; u < m; u++)
        for (int j = 0; j < 15; j++)
        {
            int v = k_NN[(size_t)u * 15 + j];
            out_NN[(size_t)perm[u] * 15 + j] = v > 0 ? perm[v - 1] + 1 : 0;
            out_D[(size_t)perm[u] * 15 + j] = k_NN_d[(size_t)u * 15 + j];
        }

    free(k_NN);
    k_NN = NULL;
    free(k_NN_d);
    k_NN_d = NULL;

    matrix_store_h5_results(argv[2], m, 15, out_NN, out_D, t2 - t0, t4 - t2 - eval_time, argv[3], Qb, Qn, k, Ln);

    free(out_NN);
    free(out_D);
    free(perm);
    free(KNNs);

    return 0;
}
