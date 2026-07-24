# Plain Local Search - SISAP 2026 Indexing Challenge

![Language](https://img.shields.io/badge/Language-C-blue)

This repository contains the software implementation for our submission to Task 1 of the SISAP 2026 Indexing Challenge. The primary goal of this software is to efficiently compute the k-nearest neighbor graph (k-NNG). It is specifically designed to tackle the problem for $k=15$ on the provided Wikipedia embeddings set, which consists of 6.5 million embeddings.

## Algorithm Pipeline

Our approach to constructing the k-NN graph relies on a four-step algorithm designed to reduce memory usage and accelerate distance computations:

* **Aggressive 1-bit Quantization**: Converts the original 16-bit floating-point vectors into 1-bit binary vectors based on a greater-than-zero rule. This allows distances to be measured as Hamming distances using efficient bitwise XOR and popcount operations. The same sweep also derives a symmetric 8-bit code per vector, used later during reranking.
* **Vantage Point Initialization**: Constructs an initial sketch of the k-NN graph using a randomized, multi-try vantage-point tree partition strategy.
* **Plain Local Search (PLS)**: Refines the initialized graph through a greedy local search. The algorithm compares points that are a distance of two apart in the current graph (neighbors of neighbors) and updates the neighborhood if they are closer.
* **Reranking**: Scans the approximate neighborhoods with the 8-bit codes, then recomputes the resulting shortlist with full 16-bit exact distance computations to construct the final k-NNG, correcting any inexact results caused by the earlier quantization.

## What the cost is actually made of

Past the distance kernel itself, every phase of this pipeline is bound by
irregular memory traffic rather than arithmetic. The design notes below follow
from that.

**The candidate list size is the accuracy knob.** Reranking with exact
distances recovers the true ordering of whatever it is given, so the recall of
the final graph is capped by how many of the true 15 neighbors appear among the
`k` Hamming candidates. Measured against exact Hamming rankings on the dev set,
that cap is 0.90 at `k=80`, 0.92 at `k=96` and 0.94 at `k=112`. Nothing
downstream can beat it, and a randomized Hadamard rotation before sign
quantization does not move it either (these embeddings are already close to
isotropic), so raising recall means carrying more candidates and paying for them
everywhere else.

**Rejection has to be cheap.** Almost every candidate the search produces is
immediately discarded for being worse than the current worst neighbor. `KNN.W`
keeps that per vertex bound in an array of its own, so the common case reads one
small, cache resident block instead of touching the much larger distance block.

**Reranking is bandwidth, not arithmetic.** At `k=96` the pass streams tens of
gigabytes of vectors and spends a few percent of its time on the inner products
themselves, which is why it scans with the 8-bit codes and only pays full 16-bit
precision for the shortlist that survives. Prefetch tuning does nothing for it;
fewer bytes is the only lever.

**Visit order is worth as much as any kernel.** The refinement and reranking
passes chase exactly the neighborhood of the vertex they are working on, so
visiting vertices in vantage point leaf order lets a chunk of consecutive
vertices share one small working set. That reordering costs one extra partition
pass and removes about 10% of the total runtime on its own.

## Build and run

```sh
make
./KNN <input.h5> [output.h5] [dataset-name]
```

The three hot passes are compiled twice, once for x86-64-v3 and once for
AVX-512 with VPOPCNTDQ, and `main` binds to the better one after a cpuid probe;
the AVX-512 Hamming kernel has about 1.7x the throughput of the AVX2 lookup
table version. Every tuning parameter listed at the top of `src/main.c` can be
overridden from the environment (`QB`, `QN`, `KK`, `LN`, `DU`, `DV`, `RR`, `OB`,
`SY`), which is how the table below was produced.

## Speed versus recall

Dev set (`benchmark-dev-wikipedia-bge-m3-small.h5`, 200k x 1024), 12 threads,
recall of the 15 reported neighbors against the bundled ground truth. Times are
build time only, excluding HDF5 loading.

| `QN` | `KK` | `DU` | `DV` | time  | recall |
| ---- | ---- | ---- | ---- | ----- | ------ |
| 12   | 96   | 24   | 32   | 3.2 s | 0.876  |
| 12   | 96   | 32   | 40   | 3.6 s | 0.905  |
| 20   | 96   | 32   | 40   | 3.9 s | 0.912  |
| 20   | 112  | 32   | 40   | 4.3 s | 0.920  |
| 32   | 112  | 32   | 48   | 4.9 s | 0.931  |
| 48   | 128  | 40   | 48   | 6.1 s | 0.946  |

The defaults are the second row. The 8-bit codes cost one extra byte per
dimension of memory (about +200 MB on the dev set, and it would scale with the
data), which is the one resource this pipeline trades away for speed.
