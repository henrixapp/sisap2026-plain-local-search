/*
    Template body for the three hot passes. Included once per ISA flavour from
    kernels.c with SUF, TGT, HAM and DOT bound to the matching variant.

    The OpenMP pragmas deliberately live in plain, untargeted wrapper functions:
    GCC outlines every parallel region into a helper of its own and that helper
    would not inherit an ISA target attribute. The per vertex worker is the
    targeted part, which costs one ordinary call per vertex and keeps the whole
    inner loop, distance kernel included, inlined at the right ISA level.
*/

#ifndef KERNELS_INC_HELPERS
#define KERNELS_INC_HELPERS
#define KCAT2(a, b) a##b
#define KCAT(a, b) KCAT2(a, b)

#define RF_PD 6     // local search prefetch distance, in candidates
#define RF_HBITS 12 // dedup table size, 2^RF_HBITS slots
#ifndef RR_PD
#ifndef RF_CHUNK
#define RF_CHUNK 16 // vertices per scheduling chunk in the refinement pass
#endif
#ifndef RR_CHUNK
#define RR_CHUNK 16 // vertices per scheduling chunk in the reranking pass
#endif
#define RR_PD 2 // rerank prefetch distance, in candidate rows
#endif
#ifndef RR_STRIDE
#define RR_STRIDE 128 // bytes between prefetch hints inside a candidate row
#endif
#endif

#define FN(name) KCAT(name, SUF)

/*
    Randomized vantage point partition.

    Pick a random pivot, split the segment at the median Hamming distance to it
    and recurse. Once a segment is small enough every pair inside it is
    evaluated; points that repeatedly land in the same small cell are likely
    neighbors. Repeating the whole thing with fresh pivots accumulates a first
    sketch of the graph.
*/
TGT static void FN(qsort_recursive_knn)(const uint64_t *Matrix, int n, VDEntry *O, int *C, int m,
                                        KNN *g, int Qb)
{
    // Base case, test all pairs and insert into KNN structure
    if (m < Qb)
    {
        for (int i = 0; i < m; i++)
        {
            int u = O[i].v;
            const uint64_t *cu = Matrix + (size_t)u * n;

            // Keep the rejection bound of u in a register: it only ever
            // decreases and only this loop can lower it from here.
            int wu = g->W[u];

            for (int j = i + 1; j < m; j++)
            {
                int v = O[j].v;

                int d = HAM(cu, Matrix + (size_t)v * n, n);

                if (d < wu)
                {
                    KNN_insert_safe(g, u, v, d);
                    wu = g->W[u];
                }

                if (d < g->W[v])
                    KNN_insert_safe(g, v, u, d);
            }
        }

        return;
    }

    const int nd = n * 64 + 1;
    memset(C, 0, sizeof(int) * nd);

    // Random pivot element
    int u = O[fast_rand() % m].v;
    const uint64_t *cu = Matrix + (size_t)u * n;

    // Compute hamming distances for all points to pivot
    for (int i = 0; i < m; i++)
    {
        int d = HAM(cu, Matrix + (size_t)O[i].v * n, n);
        O[i].d = d;
        C[d]++;
    }

    // Find median distance
    int p = 0, pd = nd - 1;
    for (int i = 0; i < nd; i++)
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
        if (O[i].d > pd)
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

    FN(qsort_recursive_knn)(Matrix, n, O, C, p, g, Qb);
    FN(qsort_recursive_knn)(Matrix, n, O + p, C, m - p, g, Qb);
}

void FN(knn_init_pass)(KNN *g, const uint64_t *B, int m, int bn, int Qb, int Qn)
{
#pragma omp parallel
    {
        VDEntry *O = malloc(sizeof(VDEntry) * m);
        int *C = malloc(sizeof(int) * (bn * 64 + 1));

        for (int i = 0; i < m; i++)
            O[i].v = i;

#pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < Qn; i++)
            FN(qsort_recursive_knn)(B, bn, O, C, m, g, Qb);

        free(O);
        free(C);
    }
}

/*
    Produce a cache friendly processing order.

    The refinement and reranking passes both spend most of their time chasing
    rows of a matrix far bigger than the last level cache, and the rows they
    chase for a given vertex are exactly that vertex's neighborhood. Visiting
    vertices in an order that keeps nearby points together therefore lets a
    whole chunk of consecutive vertices share one small working set instead of
    each pulling its own from DRAM.

    The vantage point recursion already produces such an order as a side effect,
    so this is one extra partition pass with the base case doing nothing: only
    the O(m log m) pivot distances are paid, no pair evaluation at all.
*/
TGT static void FN(order_recursive)(const uint64_t *Matrix, int n, VDEntry *O, int *C, int m,
                                    int Qb)
{
    if (m < Qb)
        return;

    const int nd = n * 64 + 1;
    memset(C, 0, sizeof(int) * nd);

    int u = O[fast_rand() % m].v;
    const uint64_t *cu = Matrix + (size_t)u * n;

    for (int i = 0; i < m; i++)
    {
        int d = HAM(cu, Matrix + (size_t)O[i].v * n, n);
        O[i].d = d;
        C[d]++;
    }

    int p = 0, pd = nd - 1;
    for (int i = 0; i < nd; i++)
    {
        p += C[i];
        if (p >= m / 2)
        {
            pd = i;
            break;
        }
    }

    p = 0;
    for (int i = 0; i < m; i++)
    {
        if (O[i].d > pd)
        {
            VDEntry t = O[p];
            O[p] = O[i];
            O[i] = t;
            p++;
        }
    }

    if (p == 0 || p == m)
        p = m / 2;

    FN(order_recursive)(Matrix, n, O, C, p, Qb);
    FN(order_recursive)(Matrix, n, O + p, C, m - p, Qb);
}

void FN(knn_build_order)(const uint64_t *B, int m, int bn, int Qb, int *Ord)
{
    VDEntry *O = malloc(sizeof(VDEntry) * m);
    int *C = malloc(sizeof(int) * (bn * 64 + 1));

    for (int i = 0; i < m; i++)
        O[i].v = i;

    FN(order_recursive)(B, bn, O, C, m, Qb);

    for (int i = 0; i < m; i++)
        Ord[i] = O[i].v;

    free(O);
    free(C);
}

/*
    Plain local search: compare u against the neighbors of its own neighbors.

    Two things dominate the cost here and both are addressed below.

    The neighborhoods of nearby points overlap heavily, so the raw Du * Dv
    candidate stream contains a large fraction of repeats. A small open
    addressed set, cleared per vertex by bumping a generation counter, filters
    them out before any memory is touched on their behalf.

    What survives is a random row of the binary matrix, i.e. a block far larger
    than the last level cache. One iteration of lookahead is nowhere near a DRAM
    round trip, so candidates are prefetched RF_PD iterations ahead.
*/

typedef struct
{
    int *key;
    int *stamp;
    int *snap; // snapshot of the neighbors being expanded
    int gen;
    int fill;
} FN(Seen);

static inline int FN(seen_test_set)(FN(Seen) * s, int w)
{
    const unsigned cap = 1u << RF_HBITS;

    // The probe below relies on there always being an empty slot. Well within
    // the sane parameter range the table never gets close to full, but a large
    // Du * Dv must not be able to turn that into a spin.
    if ((unsigned)s->fill >= cap - (cap >> 2))
        return 0;

    unsigned h = ((unsigned)w * 2654435761u) >> (32 - RF_HBITS);

    while (s->stamp[h] == s->gen)
    {
        if (s->key[h] == w)
            return 1;
        h = (h + 1) & (cap - 1);
    }

    s->stamp[h] = s->gen;
    s->key[h] = w;
    s->fill++;
    return 0;
}

TGT static void FN(refine_one)(KNN *g, int u, const uint64_t *B, int m, int bn, int Du, int Dv,
                               int sym, FN(Seen) * seen)
{
    const int k = g->k;
    const uint64_t *cu = B + (size_t)u * bn;
    const int *Vu = g->V + ((size_t)u * k);
    int *snap = seen->snap;

    seen->gen++;
    seen->fill = 0;
    FN(seen_test_set)(seen, u);

    // Expand a snapshot of the current neighbors rather than the live row.
    // Insertions made below, and by other threads working on other vertices,
    // shift this row around underneath us; chasing those shifts is what the
    // loop used to do and it is not guaranteed to terminate.
    int ns = 0;
    for (int i = 0; i < Du; i++)
    {
        int v = Vu[i];
        if (v < m)
            snap[ns++] = v;
    }

    for (int i = ns - 1; i >= 0; i--)
    {
        const int *Vv = g->V + ((size_t)snap[i] * k);

        for (int j = 0; j < Dv; j++)
        {
            // Pull a candidate vector in well ahead of when it is needed.
            if (j + RF_PD < Dv)
            {
                int wn = Vv[j + RF_PD];
                if (wn < m)
                {
                    const char *p = (const char *)(B + (size_t)wn * bn);
                    __builtin_prefetch(p, 0, 1);
                    __builtin_prefetch(p + 64, 0, 1);
                    __builtin_prefetch((const char *)&g->W[wn], 0, 1);
                }
            }

            int w = Vv[j];
            if (w >= m)
                continue;

            if (FN(seen_test_set)(seen, w))
                continue;

            int d = HAM(cu, B + (size_t)w * bn, bn);

            if (d < g->W[u])
                KNN_insert_safe(g, u, w, d);

            if (sym && d < g->W[w])
                KNN_insert_safe(g, w, u, d);
        }
    }
}

void FN(knn_refine)(KNN *g, const uint64_t *B, const int *Ord, int m, int bn, int Du, int Dv,
                    int sym)
{
    if (Du > g->k)
        Du = g->k;
    if (Dv > g->k)
        Dv = g->k;

#pragma omp parallel
    {
        FN(Seen) seen;
        seen.key = malloc(sizeof(int) << RF_HBITS);
        seen.stamp = calloc((size_t)1 << RF_HBITS, sizeof(int));
        seen.snap = malloc(sizeof(int) * g->k);
        seen.gen = 0;
        seen.fill = 0;

#pragma omp for schedule(dynamic, RF_CHUNK)
        for (int t = 0; t < g->n; t++)
            FN(refine_one)(g, Ord[t], B, m, bn, Du, Dv, sym, &seen);

        free(seen.key);
        free(seen.stamp);
        free(seen.snap);
    }
}

static inline void FN(rr_insert)(KNN *g, int *N, float *Nd, int R, int u, int v, float d)
{
    KNN_lock(g, u);

    if (d > Nd[R - 1])
    {
        int dup = 0;
        for (int i = 0; i < R; i++)
            if (N[i] == v + 1)
            {
                dup = 1;
                break;
            }

        if (!dup)
        {
            int p = R - 2;
            while (p >= 0 && Nd[p] < d)
            {
                N[p + 1] = N[p];
                Nd[p + 1] = Nd[p];
                p--;
            }
            N[p + 1] = v + 1;
            Nd[p + 1] = d;
        }
    }

    KNN_unlock(g, u);
}

/*
    Rerank the approximate neighborhoods.

    Two passes. The first scans every candidate with the int8 codes and keeps a
    shortlist of R; each product is used in both directions, so the reverse
    edges widen the effective candidate pool at no extra distance cost. The
    second recomputes the whole shortlist from the fp16 originals and reorders
    it, so the 15 that survive and the similarities reported for them are exact.

    Splitting it this way is what makes the pass affordable: it is purely
    bandwidth bound, the wide scan now touches one byte per dimension instead of
    two, and full precision is only paid for R of the k candidates. R sits a
    little above 15 so that a pair the int8 estimate ranks slightly wrong is
    still caught by the exact pass.
*/
TGT static void FN(select_one)(KNN *g, int u, const int8_t *Q, const float *S, int m, int n, int R,
                               int *N, float *Nd)
{
    const int k = g->k;
    const int8_t *cu = Q + ((size_t)u * n);
    const float su = S[u];
    const int *Vu = g->V + ((size_t)u * k);
    int *Nu = N + ((size_t)u * R);
    float *Ndu = Nd + ((size_t)u * R);

    for (int i = 0; i < k; i++)
    {
        int v = Vu[i];
        if (v >= m)
            continue;

        if (i + RR_PD < k)
        {
            int vn = Vu[i + RR_PD];
            if (vn < m)
            {
                const char *p = (const char *)(Q + (size_t)vn * n);
                for (int o = 0; o < n; o += RR_STRIDE)
                    __builtin_prefetch(p + o, 0, 1);
            }
        }

        float d = (float)DOTI8(cu, Q + ((size_t)v * n), n) * su * S[v];

        if (d > Ndu[R - 1])
            FN(rr_insert)(g, Nu, Ndu, R, u, v, d);

        if (d > Nd[(size_t)v * R + R - 1])
            FN(rr_insert)(g, N + ((size_t)v * R), Nd + ((size_t)v * R), R, v, u, d);
    }
}

TGT static void FN(exact_one)(int u, const uint16_t *M, int n, int R, int *N, float *Nd)
{
    const uint16_t *cu = M + ((size_t)u * n);
    int *Nu = N + ((size_t)u * R);
    float *Ndu = Nd + ((size_t)u * R);

    int cnt = 0;
    for (; cnt < R && Nu[cnt] != 0; cnt++)
        Ndu[cnt] = DOT(cu, M + ((size_t)(Nu[cnt] - 1) * n), n);

    // Insertion sort by descending similarity over a short, nearly sorted list.
    for (int i = 1; i < cnt; i++)
    {
        int v = Nu[i];
        float d = Ndu[i];
        int j = i - 1;
        while (j >= 0 && Ndu[j] < d)
        {
            Nu[j + 1] = Nu[j];
            Ndu[j + 1] = Ndu[j];
            j--;
        }
        Nu[j + 1] = v;
        Ndu[j + 1] = d;
    }
}

void FN(knn_rerank)(KNN *g, const uint16_t *M, const int8_t *Q, const float *S, const int *Ord,
                    int m, int n, int R, int *N, float *Nd)
{
#pragma omp parallel for schedule(dynamic, RR_CHUNK)
    for (int t = 0; t < g->n; t++)
        FN(select_one)(g, Ord[t], Q, S, m, n, R, N, Nd);

#pragma omp parallel for schedule(dynamic, RR_CHUNK)
    for (int t = 0; t < g->n; t++)
        FN(exact_one)(Ord[t], M, n, R, N, Nd);
}

#undef FN
