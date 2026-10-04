# HPC-MIX Bench

[![Documentation](https://readthedocs.org/projects/hpc-mix-bench/badge/?version=latest)](https://hpc-mix-bench.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e?style=flat-square)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776ab?style=flat-square&logo=python&logoColor=white)](docs/source/installation.rst)
[![C/C++ benchmarks](https://img.shields.io/badge/benchmarks-C%2FC%2B%2B-2563eb?style=flat-square&logo=cplusplus&logoColor=white)](docs/source/applications.rst)
[![Docker ready](https://img.shields.io/badge/Docker-ready-0ea5e9?style=flat-square&logo=docker&logoColor=white)](Dockerfile)
[![40+ numerical workloads](https://img.shields.io/badge/workloads-40%2B-7c3aed?style=flat-square)](docs/source/applications.rst)
[![Last commit](https://img.shields.io/github/last-commit/PEQUAN/hpc-mix-bench?style=flat-square&color=0f766e)](https://github.com/PEQUAN/hpc-mix-bench/commits/main/)

HPC-MIX Bench is a collection of C/C++ numerical benchmarks for evaluating PROMISE mixed-precision tuning.  The repository includes benchmark programs, shared run-setting templates, Docker support, helper scripts for large benchmark sweeps, and post-processing tools for precision-count and one-bit precision analyses.

## Why HPC-MIX Bench?

Most numerical simulations default to double precision (IEEE 754 binary64) even though many variables tolerate much lower precision without affecting the accuracy of the final result. Manually discovering which variables can be safely narrowed is tedious and error-prone, yet the payoff is real: lower precision reduces compute time, memory bandwidth, and energy consumption, and it is a prerequisite for exploiting the low-precision arithmetic units (FP16, BF16, FP8/E5M2, FP8/E4M3) available on modern accelerators.

HPC-MIX Bench provides a curated, reproducible testbed of 40+ numerical and machine-learning kernels (linear solvers, iterative methods, numerical integration, Rodinia-style physical simulations, and classic ML algorithms) for exercising [PROMISE](cadnaPromise/) — a floating-point precision auto-tuning tool built on delta debugging and the [CADNA](https://cadna.lip6.fr/) library for rigorous round-off error estimation via Discrete Stochastic Arithmetic. This lets researchers and practitioners:

- Reproduce and extend published mixed-precision tuning results across a broad benchmark suite instead of a handful of toy examples.
- Quantify, per variable and per required accuracy (1-10 correct significant digits), how much of a program can run in 8/16-bit formats before results become unreliable.
- Compare four precision search spaces (E5M2/E4M3 x FP16/BF16, alongside FP32/FP64) to study the accuracy/performance trade-offs of emerging low-precision hardware formats.
- Feed real precision-assignment data into research on autotuning, compiler transformations, and energy-aware HPC.

**Explore the documentation:** [Start here](docs/source/index.rst) · [Precision tuning concepts](docs/source/precision_tuning.rst) · [Benchmark methodology and use cases](docs/source/methodology.rst) · [Results and interpretation](docs/source/benchmark_results.rst)

The suite separates *precision-search cost* from *application performance*: PROMISE sweeps record the time spent searching for a valid assignment, while the H100 validation artifacts report measured CUDA execution time and device allocation. See the [methodology guide](docs/source/methodology.rst) before comparing these numbers.

## Repository layout

```text
hpc-mix-bench/
├── 1-bit-exps/            # One-bit-granularity precision sweeps and figures
├── cadnaPromise/          # Bundled CADNA/PROMISE Python package
├── data/                  # Input datasets and data-query utilities
├── docs/                  # Sphinx documentation
├── mp_tests/              # Benchmark folders and runner scripts
├── papers/                # Plot/statistics helpers used for paper artifacts
├── run_settings/          # Shared run_setting_*.py, run_debug_*.sh, and fp.json templates
├── src/                   # Additional benchmark/source experiments
├── Dockerfile
└── docker-compose.yml
```

Each runnable benchmark under `mp_tests/` is a directory that contains at least:

- `promise.yml` with the PROMISE compile/run configuration.
- One or more `run_setting_*.py` files.
- C/C++ source and any data files needed by the benchmark.

The current `mp_tests/` tree includes benchmarks such as `backprop`, `dense_lu`, `hotspot`, `particle_filter`, `srad_v2`, `adaboost`, `bicgstab`, `cg`, `dbscan`, `gmres_tol1`, `kmeans`, `mlp`, `pca`, `qr`, `randomforest`, `sparse_lu`, `svm`, and others.


## Setup

### Local environment

Install compilers and Python dependencies needed by PROMISE and the plotting scripts:

```bash
python3 -m pip install -e ./cadnaPromise numpy matplotlib
activate-promise
```

See [`cadnaPromise/`](cadnaPromise/) for details about PROMISE, CADNA activation, `promise.yml`, and `fp.json`.

To add a benchmark, create a new folder under `mp_tests/` with the benchmark source code, `promise.yml`, and an `fp.json` file or use the shared template from `run_settings/fp.json`.

### Docker

The Docker image installs the bundled `cadnaPromise` package in a virtual environment.  `matplotlib` is installed by default for plot generation.

On macOS Apple Silicon and Windows on ARM, build for `linux/amd64`:

```bash
docker buildx build --platform linux/amd64 -t hpc-mix-cadna .
docker run --platform linux/amd64 -it --rm hpc-mix-cadna
```

On Linux x86_64 and Windows on Intel/AMD:

```bash
docker build -t hpc-mix-cadna .
docker run -it --rm hpc-mix-cadna
```

You can also use Docker Compose:

```bash
docker compose run --rm promise-env
```

Inside the container, run `activate-promise` if you need to refresh the PROMISE environment.

## Configure benchmark settings

Global run templates live in `run_settings/`:

- `run_setting_1.py` to `run_setting_4.py`
- `run_debug_1.sh` to `run_debug_4.sh`
- `fp.json`

To copy these templates into every top-level benchmark folder under `mp_tests/` that contains `promise.yml`:

```bash
cd mp_tests
bash sync_settings.sh
```

`sync_settings.sh` options:

| Option | Description |
|:--|:--|
| `--delete`, `-d` | Delete existing `run_setting_*.py`, `run_debug_*.sh`, and `fp.json` files from benchmark folders. |
| `--broadcast`, `-b` | Copy files from `../run_settings/` into benchmark folders. |
| `--advanced`, `-a` | Also copy `../run_settings/advanced/run_setting_*.py` when that directory exists. |
| `--help`, `-h` | Show usage help. |

If no option is given, the script runs both delete and broadcast.

## Run benchmarks

Run the main automation script from `mp_tests/`:

```bash
cd mp_tests
chmod +x run_benchmarks.sh
./run_benchmarks.sh [run_exp] [run_plot] [run_debug] [folders...] [--parallel] [--jobs N]
```

Arguments:

- `run_exp`: run PROMISE experiments. Accepted true values are `1`, `true`, `y`, `yes`; false values are `0`, `false`, `n`, `no`. Default: `true`.
- `run_plot`: generate plots from results. Uses saved results when experiments are skipped. Default: `true`.
- `run_debug`: run matching `run_debug_i.sh` scripts after `run_setting_i.py`. Default: `false`.
- `folders...`: optional benchmark folders. When omitted, the script auto-detects valid folders up to two levels deep.
- `--parallel`: run setting/debug tasks in parallel when GNU Parallel is installed.
- `--jobs N` or `--jobs=N`: number of parallel jobs. Default is `$JOBS`, then `nproc`, then `4`.

Common examples:

| Command | Description |
|:--|:--|
| `./run_benchmarks.sh` | Run experiments and plots for all detected benchmarks, skip debug, sequential mode. |
| `./run_benchmarks.sh 1 0` | Run experiments only. |
| `./run_benchmarks.sh 0 1` | Generate plots only from existing results. |
| `./run_benchmarks.sh 1 1 1` | Run experiments, plots, and debug scripts. |
| `./run_benchmarks.sh true true false hotspot dense_lu` | Run selected folders sequentially. |
| `./run_benchmarks.sh 1 1 false --parallel --jobs 4` | Run setting tasks in parallel with four workers. |

The script writes logs under `mp_tests/logs/<folder>/run_<i>.log`.  It also sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` to `1` unless those variables are already defined.

For repeatable comparisons, run settings that share one benchmark directory sequentially or give each setting an isolated copy of that directory. The `--parallel` option dispatches setting tasks, which may otherwise share generated binaries and output paths.

### MPI runner

For MPI-based distribution, use `mpi_runner.py` from `mp_tests/` after installing `mpi4py` and an MPI runtime:

```bash
mpirun -np 4 python3 mpi_runner.py true true false hotspot dense_lu
```

The positional arguments are the same boolean flags used by `run_benchmarks.sh`: `run_exp`, `run_plot`, and `run_debug`, followed by optional benchmark folders.

## Outputs and summaries

Each `run_setting_i.py` template can produce:

- `prec_setting_<i>.json` with PROMISE precision assignments.
- `precision<i>_with_runtime.jpg` plots.
- Runtime and status output in the benchmark log.

To summarize precision assignment counts across completed benchmarks:

```bash
cd mp_tests
python3 calculate_stats.py backprop dense_lu hotspot
```

This writes:

- `fp_counts_summary.csv`
- `fp_ratio_averages.csv`

For paper-style plot organization, use the helper under `papers/`:

```bash
cd papers
bash organize_plots.sh [folder1 folder2 ...]
```

## One-bit precision analysis

`1-bit-exps/onebit_precision_analysis1.py` sweeps a custom PROMISE floating-point format against double precision with one-bit granularity. It can collect fresh PROMISE data or regenerate figures from existing CSV files. The `onebit_precision_analysis2.py` through `onebit_precision_analysis4.py` variants provide alternative visualizations.

Run from `1-bit-exps/`:

```bash
python3 onebit_precision_analysis1.py --run
python3 onebit_precision_analysis1.py --run --benchmark hotspot --benchmark dense_lu
python3 onebit_precision_analysis1.py --digits 1-10
python3 onebit_precision_analysis1.py --nb-digits 6
```

Run from the repository root:

```bash
python3 1-bit-exps/onebit_precision_analysis1.py --repo-root mp_tests --run
```

Useful options:

| Option | Description |
|:--|:--|
| `--benchmark NAME` | Select a benchmark folder under `--repo-root`. Repeat this option to select multiple folders. Defaults to `backprop` and `dense_lu`. |
| `--run` | Run PROMISE sweeps before plotting. Without it, existing `onebit_precision_results.csv` files are used. |
| `--digits LIST` | Significant digits to sweep, such as `1-10` or `2,4,6`. |
| `--nb-digits N` | Run one significant-digit target when `--digits` is not set. |
| `--cadna-path PATH` | Optional `CADNA_PATH` override. If omitted, the script uses the environment or bundled CADNA from `cadnaPromise`. |
| `--timeout SECONDS` | Per-PROMISE-run timeout. Default: `300`. |

Outputs are written to:

- `<benchmark>/onebit_precision_results.csv`
- `<repo-root>/figures/*_1bit_sweep.{pdf,png}`
- `<repo-root>/figures/*_digit_precision_counts.{pdf,png}`
- `<repo-root>/figures/onebit_precision_summary.txt`

## References

HPC-MIX Bench builds on the PROMISE precision auto-tuning tool and the CADNA library for round-off error control. If you use this benchmark suite, please consider citing the underlying research:

- S. Graillat, F. Jézéquel, R. Picot, F. Févotte, B. Lathuilière. *Auto-tuning for floating-point precision with Discrete Stochastic Arithmetic*. Journal of Computational Science, 36, 101017, 2019. [HAL:hal-01331917](https://hal.archives-ouvertes.fr/hal-01331917)
- F. Jézéquel and J.-M. Chesneaux. *CADNA: a library for estimating round-off error propagation*. Computer Physics Communications, 178(12):933-955, 2008.
- P. Eberhart, J. Brajard, P. Fortin, F. Jézéquel. *High Performance Numerical Validation using Stochastic Arithmetic*. Reliable Computing, 21, 35-52, 2015.
- A. Zeller. *Why Programs Fail*, 2nd ed., Morgan Kaufmann, 2009. (Delta debugging algorithm used by PROMISE's search.)

See [`cadnaPromise/docs/source/index.rst`](cadnaPromise/docs/source/index.rst) for the complete PROMISE/CADNA bibliography, and cite this repository itself as:

```bibtex
@software{hpc_mix_bench,
  title  = {HPC-MIX Bench: Benchmarks for Mixed-Precision Emulations},
  author = {PEQUAN Team},
  year   = {2026},
  url    = {https://github.com/PEQUAN/hpc-mix-bench}
}
```

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details. The bundled [`cadnaPromise/`](cadnaPromise/) package is distributed separately under the **GNU LGPLv3**; see [`cadnaPromise/LICENSE`](cadnaPromise/LICENSE) for details.
