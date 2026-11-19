# Plain Local Search — SISAP 2026 Indexing Challenge
![Language](https://img.shields.io/badge/Language-C-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![SISAP 2026](https://img.shields.io/badge/SISAP-2026-brightgreen.svg)
[![Heidelberg University](https://img.shields.io/badge/Heidelberg-University-c1002a)](https://www.uni-heidelberg.de)

**Authors:** Kenneth Langedal, Henrik Reinstädtler

This repository contains the C implementation of **Plain Local Search (PLS)** for our submission to Task 1 of the [SISAP 2026 Indexing Challenge](https://sisap-challenges.github.io/sisap26-leaderboard/leaderboard.html). The solver computes an approximate $k$-nearest neighbor graph ($k$-NNG) under inner-product similarity from 16-bit or 32-bit floating-point HDF5 embeddings. Originally developed for $k = 15$ on the 6.5M-vector Wikipedia embeddings benchmark, all algorithm hyperparameters and runtime settings are configurable from the command line.

See the [SISAP 2026 leaderboard](https://sisap-challenges.github.io/sisap26-leaderboard/leaderboard.html) for challenge results, and the [knn-optimisation](https://github.com/henrixapp/sisap2026-plain-local-search/tree/knn-optimisation) branch for post-challenge optimization history.

## Algorithm Pipeline

The $k$-NNG is constructed in four stages designed to minimize memory bandwidth and accelerate distance computations with AVX2 / AVX-512 SIMD kernels:

1. **Quantization**: Reads each input vector once and derives two compressed representations: a **1-bit binary code** (sign bit, $> 0$) for fast XOR + popcount Hamming distances, and a **symmetric 8-bit integer code** (`int8` with a per-vector float scale) for bandwidth-efficient candidate scanning.
2. **Vantage Point Initialization**: Builds an initial sketch of the $k$-NNG over `Qn` randomized vantage-point tree partitions, recursively splitting at the median Hamming distance down to leaves of size `Qb` and evaluating all pairs within each leaf. One additional partition pass (leaf size `Ob`) orders vertices for cache locality in the remaining passes.
3. **Plain Local Search (PLS)**: Refines the graph for `Ln` rounds by comparing each vertex against the neighbors of its neighbors (`Du` first-hop $\times$ `Dv` second-hop candidates), using per-vertex deduplication and software prefetching. The first `Sy` rounds also insert reverse edges to improve graph connectivity.
4. **Two-Stage Reranking**: Scans each vertex's `k` approximate neighbors (and reverse edges) using the `int8` codes to keep a shortlist of size `R`, then rescores the shortlist with exact `fp16` inner products and outputs the top `target_k` neighbors.

## Dependencies & Installation

### Requirements
* **CPU**: x86-64 processor with AVX2, FMA, and F16C support (`x86-64-v3`). CPUs with AVX-512 and `AVX512_VPOPCNTDQ` are detected automatically at runtime for faster kernels.
* **Compiler & Build Tools**: GCC (with C17 and OpenMP support) and GNU `make`.
* **HDF5**: HDF5 C development library providing the `h5cc` compiler wrapper.

### Installing Dependencies

On Ubuntu / Debian:
```sh
sudo apt-get update
sudo apt-get install -y build-essential libhdf5-dev
```

### Building

```sh
git clone https://github.com/henrixapp/sisap2026-plain-local-search.git
cd sisap2026-plain-local-search
make
```

This produces the `./KNN` executable in the repository root. To clean build artifacts, run `make clean`.

### Docker (Optional)

If you prefer not to install HDF5 locally, you can build and run via Docker:

```sh
docker build -t sisap2026-pls .
```

## Usage

```sh
./KNN [OPTIONS] <input.h5> [output.h5]
```

**Example:**
```sh
./KNN --qn 12 --ln 2 -k 96 -K 15 -d wiki2026 data.h5 results.h5
```

If `output.h5` is omitted, the solver runs and reports timings (and recall, if ground truth is present in the input HDF5 file) without writing an output file.

### Command-Line Options

#### Algorithm Hyperparameters

| Option | Default | Description |
| :--- | :--- | :--- |
| `--qb <int>` | `256` | Vantage-point partition base case size ($Q_b$) |
| `--qn <int>` | `12` | Number of randomized partition rounds ($Q_n$) |
| `-k`, `--knn-k <int>` | `96` | Approximate neighborhood size carried into reranking (padded to a multiple of 16) |
| `--ln <int>` | `2` | Local search refinement rounds ($L_n$) |
| `--du <int>` | `32` | First-hop neighbors of $u$ expanded during local search ($D_u$) |
| `--dv <int>` | `40` | Second-hop neighbors expanded per first-hop neighbor ($D_v$) |
| `-r`, `--shortlist <int>` | `18` | Shortlist size kept by the `int8` scan for exact `fp16` rescoring ($R$, clamped to $\ge$ `target-k`) |
| `--ob <int>` | `64` | Leaf size of the partition defining the cache-friendly processing order ($O_b$) |
| `--sy <int>` | `1` | Number of initial refinement rounds that also insert reverse edges ($S_y$) |
| `-K`, `--target-k <int>` | `15` | Final number of nearest neighbors to output and evaluate |

#### I/O & Runtime Options

| Option | Default | Description |
| :--- | :--- | :--- |
| `-o`, `--output <path>` | *(none)* | Output HDF5 file path (alternative to the positional `[output.h5]` argument) |
| `-d`, `--dataset <name>` | `unknown` | Dataset name stored in the output HDF5 metadata |
| `--train-key <path>` | `/train` | HDF5 dataset path for input vectors (`float16` or `float32`) |
| `--gt-key <path>` | `/allknn/knns` | Optional HDF5 dataset path for ground-truth neighbors (used to report recall) |
| `-s`, `--seed <uint>` | `42` | Global RNG seed |
| `-t`, `--threads <int>` | *(OpenMP default)* | Number of OpenMP threads |
| `--no-avx512` | `false` | Disable AVX-512 kernels and force AVX2 |
| `-q`, `--quiet` | `false` | Suppress all stdout output |
| `-h`, `--help` | | Print usage information and exit |

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
