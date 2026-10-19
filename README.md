# Plain Local Search - SISAP 2026 Indexing Challenge

![Language](https://img.shields.io/badge/Language-C-blue)

This repository contains the software implementation for our submission to Task 1 of the SISAP 2026 Indexing Challenge. The primary goal of this software is to efficiently compute the k-nearest neighbor graph (k-NNG). It is specifically designed to tackle the problem for $k=15$ on the provided Wikipedia embeddings set, which consists of 6.5 million embeddings.
You can find the leaderboard [here](https://sisap-challenges.github.io/sisap26-leaderboard/leaderboard.html). The post-challenge optimizations can be found on branch [knn-optimisation](https://github.com/henrixapp/sisap2026-plain-local-search/tree/knn-optimisation).

## Algorithm Pipeline

Our approach to constructing the k-NN graph relies on a four-step algorithm designed to reduce memory usage and accelerate distance computations:

* **Aggressive 1-bit Quantization**: Converts the original 16-bit floating-point vectors into 1-bit binary vectors based on a greater-than-zero rule. This allows distances to be measured as Hamming distances using efficient bitwise XOR and popcount operations. The same sweep also derives a symmetric 8-bit code per vector, used later during reranking.
* **Vantage Point Initialization**: Constructs an initial sketch of the k-NN graph using a randomized, multi-try vantage-point tree partition strategy.
* **Plain Local Search (PLS)**: Refines the initialized graph through a greedy local search. The algorithm compares points that are a distance of two apart in the current graph (neighbors of neighbors) and updates the neighborhood if they are closer.
* **Reranking**: Scans the approximate neighborhoods with the 8-bit codes, then recomputes the resulting shortlist with full 16-bit exact distance computations to construct the final k-NNG, correcting any inexact results caused by the earlier quantization.


## Build and run

```sh
make
./KNN <input.h5> [output.h5] [dataset-name]
```


