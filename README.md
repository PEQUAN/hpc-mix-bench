# HPC-MIX Benchmarks

[![Docs](https://readthedocs.org/projects/hpc-mix-bench/badge/?version=latest)](https://hpc-mix-bench.readthedocs.io/)
[![Python](https://img.shields.io/badge/python-3.10+-4B8BBE?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/github/license/PEQUAN/hpc-mix-bench)](https://github.com/PEQUAN/hpc-mix-bench)

HPC-MIX Bench is a collection of C/C++ numerical benchmarks for evaluating PROMISE mixed-precision tuning.  The repository includes benchmark programs, shared run-setting templates, Docker support, helper scripts for large benchmark sweeps, and post-processing tools for precision-count and one-bit precision analyses.

## Repository layout

```text
hpc-mix-bench/
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
python3 -m pip install cadnaPromise matplotlib numpy
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

`mp_tests/onebit_precision_analysis.py` sweeps a custom PROMISE floating-point format against double precision with one-bit granularity.  It can either collect fresh PROMISE data or regenerate figures from existing CSV files.

Run from `mp_tests/`:

```bash
python3 onebit_precision_analysis.py --run
python3 onebit_precision_analysis.py --run --benchmark hotspot --benchmark dense_lu
python3 onebit_precision_analysis.py --digits 1-10
python3 onebit_precision_analysis.py --nb-digits 6
```

Run from the repository root:

```bash
python3 mp_tests/onebit_precision_analysis.py --repo-root mp_tests --run
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

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
