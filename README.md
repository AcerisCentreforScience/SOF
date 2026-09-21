# SOF: A Diagnostic Framework for Non-Stationary Bandits

This repository contains the code used to verify the four structural facts in the paper
"On Memory Time-Scales and Parameter Robustness in Non-Stationary Multi-Armed Bandits"
(under review at *Machine Learning*).

**SOF** (Second-Order Framework) is a unified upper-confidence framework that combines
second-order variance correction, adjustable temporal memory, and sparse-structure
exploration. In the paper it is used as a diagnostic instrument, not as a benchmarked
algorithm; the contributions are the four structural facts, not the framework itself.

## Build

Header-only, no external dependencies. Requires a C++17 compiler (GCC 9+ or Clang 10+).

```bash
mkdir -p build
g++ -std=c++17 -O2 -I include src/main_explore_bound.cpp   -o build/explore_floor
g++ -std=c++17 -O2 -I include src/main_delta_scan.cpp      -o build/delta_sweep
g++ -std=c++17 -O2 -I include src/main_drift_scan.cpp      -o build/drift_sweep
g++ -std=c++17 -O2 -I include src/main_stationary.cpp     -o build/stationary
g++ -std=c++17 -O2 -I include src/main_benchmark.cpp       -o build/benchmark
```

Each experiment runs in under a minute on a modern CPU.

## Reproduce paper results

```bash
./build/explore_floor    > results/explore_bound_new.txt   # Theorem 1 (Fig. 7)
./build/delta_sweep      > results/delta_new.txt            # Table 3 (Theorem 2 boundary)
./build/drift_sweep      > results/drift_new.txt            # Theorem 3 (Fig. 4)
./build/stationary       > results/stationary_new.txt        # Table 2, Theorem 1
./build/benchmark        > results/benchmark_new.txt        # Full baseline comparison
```

Reference outputs used in the paper are in `results/*.txt`. Numerical agreement
should be within seed noise (5--10 seeds).

## Repository layout

```
include/   header-only implementation (sof.hpp is the core)
src/       one main file per experiment
results/   reference outputs
```

| File | Paper section |
|---|---|
| `main_explore_bound.cpp` | Theorem 1 (Fig. 7) |
| `main_delta_scan.cpp` | Theorem 2 / Table 3 |
| `main_drift_scan.cpp` | Theorem 3 (Fig. 4) |
| `main_stationary.cpp` | Table 2, Theorem 1 |
| `main_benchmark.cpp` | Full baseline comparison |

## Notes

- The Open Bandit Dataset requires a separate download (see
  https://github.com/st-tech/zr-obp); it is not bundled here.
- Random seeds are fixed for reproducibility.
- This code is released under the MIT License (see `LICENSE`).
