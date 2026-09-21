# HCB3 Bandit Reproducibility Artifact

This repository contains the code used to verify the four structural facts in the paper
"On Memory Time-Scales and Parameter Robustness in Non-Stationary Multi-Armed Bandits"
(under review at *Machine Learning*).

The framework is referred to as **SOF** (Second-Order Framework) in the paper; the
internal implementation name is **HCB3**. The code is provided as a reproducibility
artifact, not as a benchmark-oriented algorithm release.

## Build

Header-only, no external dependencies. Requires a C++17 compiler.

```bash
mkdir -p build
g++ -std=c++17 -O2 -I include src/main_explore_bound.cpp   -o build/explore_floor
g++ -std=c++17 -O2 -I include src/main_delta_scan.cpp      -o build/delta_sweep
g++ -std=c++17 -O2 -I include src/main_drift_scan.cpp      -o build/drift_sweep
g++ -std=c++17 -O2 -I include src/main_stationary.cpp     -o build/stationary
g++ -std=c++17 -O2 -I include src/main_benchmark.cpp       -o build/benchmark
```

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
include/   header-only implementation (hcb3.hpp is the core)
src/       one main file per experiment
results/   reference outputs
```

## Notes

- The Open Bandit Dataset requires a separate download (see
  https://github.com/st-tech/zr-obp); it is not bundled here.
- Random seeds are fixed for reproducibility.
