# H100 CUDA mixed-precision ports

This directory contains CUDA ports for the transformed mixed-precision
programs under:

- `mp_tests/backprop/digit*_*`
- `mp_tests/dense_lu/digit*_*`
- `mp_tests/hotspot/digit*_*`

It also contains one double-precision CUDA baseline for each benchmark:

- `mp_tests/backprop/double_precision_cuda.cu`
- `mp_tests/dense_lu/double_precision_cuda.cu`
- `mp_tests/hotspot/double_precision_cuda.cu`

Each transformed directory contains a small CUDA entry point and a
`Makefile.cuda`. The entry point defines macros that mirror the
`flx::floatx<e,t>` choices found in the corresponding CPU transformed
program and includes the shared benchmark implementation. The baseline
entry points define the same macros as `floatx<11,52>` so the same CUDA
kernel structure runs in FP64.

## H100 low-precision mapping

The shared CUDA layer is `mp_tests/cuda_common/h100_mixed_precision.cuh`.
It maps the FloatX-style `(e,t)` format descriptions to CUDA/H100 types
when CUDA provides a native storage type:

- `floatx<5,2>` -> `__nv_fp8_e5m2` when `<cuda_fp8.h>` is available
- `floatx<4,3>` -> `__nv_fp8_e4m3` when `<cuda_fp8.h>` is available
- `floatx<5,10>` -> `__half`
- `floatx<8,7>` -> `__nv_bfloat16`
- `floatx<8,23>` -> `float`
- `floatx<11,52>` -> `double`

For unsupported custom `(e,t)` pairs, the layer falls back to a compact
software quantizer so that the CUDA source still expresses the selected
precision explicitly.

The CUDA benchmark implementations store the selected mixed-precision
state in the mapped CUDA storage type where that transformed variable is
part of the benchmark state. This makes the reported memory footprint
comparable against the FP64 baseline instead of only measuring transient
rounding in registers.

For Hotspot, the H100-optimized CUDA backend maps the grid storage
(`temp`, `power`, and `result`) to `HS_FIELD_*`, which defaults to the
PROMISE-selected `HS_DELTA_*` precision for each `digit<i>_<j>` wrapper.
The FP64 baseline still maps every field to `floatx<11,52>`. This gives a
hardware-storage interpretation of the PROMISE configuration rather than
a purely register-rounding emulation.

For Dense LU, pivot selection, row swap, scaling, and trailing updates are
kept on the device across each factorization step. The implementation
copies the singularity flag back to the host once at the end, avoiding a
per-pivot host/device synchronization while preserving the same partial
pivoting control flow.

## Building

Build every generated H100 CUDA port and both double-precision baselines:

```sh
cd mp_tests
./build_cuda_h100.sh
```

Build one transformed directory:

```sh
cd mp_tests/backprop/digit2_5
make -f Makefile.cuda

cd mp_tests/dense_lu/digit2_5
make -f Makefile.cuda

cd mp_tests/hotspot/digit2_5
make -f Makefile.cuda
```

Build only a double-precision baseline:

```sh
cd mp_tests/backprop
make -f Makefile.cuda

cd mp_tests/dense_lu
make -f Makefile.cuda

cd mp_tests/hotspot
make -f Makefile.cuda
```

The default target architecture is `sm_90`, suitable for H100. Override
it with `CUDA_ARCH=sm_90a` or another architecture if needed.

## Running

Backprop accepts the input layer size, matching the original benchmark
interface:

```sh
./backprop/backprop_cuda_double 65536
./backprop/digit2_5/backprop_cuda 65536
```

The default H100 scripts use `BACKPROP_SIZE=65536`.  For a modestly larger but
still lightweight run, use `BACKPROP_SIZE=131072`; larger values are optional
and can make the backprop sweep less convenient without necessarily improving
mixed-precision speedups.

Dense LU accepts an optional matrix size and defaults to `5000`:

```sh
./dense_lu/dense_lu_cuda_double 5000
./dense_lu/digit2_5/dense_lu_cuda 5000
```

Hotspot keeps the original benchmark interface:

```sh
python3 hotspot/generate_hotspot_input.py 1024 1024 hotspot/temp_1024 hotspot/power_1024
./hotspot/hotspot_cuda_double 1024 1024 2 hotspot/temp_1024 hotspot/power_1024 hotspot/output_cuda_double.out
./hotspot/digit2_5/hotspot_cuda 1024 1024 2 hotspot/temp_1024 hotspot/power_1024 hotspot/output_digit2_5.out
```

Both the mixed-precision and double-precision executables print timing
and memory fields with the same names:

- Backprop: `kernel_time_ms`
- Dense LU: `factorization_time_ms`
- Hotspot: `kernel_time_ms`
- All benchmarks: `device_allocation_bytes` and `device_allocation_mib`
- Backprop additionally reports the MSE and relative L2/Linf error of the
  PROMISE-marked `output_delta` array relative to the FP64 CUDA baseline in
  the comparison CSV.
- Dense LU additionally reports `relative_residual`, `relative_error`, and
  solution-vector errors relative to the FP64 CUDA baseline in the comparison
  CSV.
- Hotspot additionally reports output-field L2/Linf errors relative to the
  FP64 CUDA output in the comparison CSV.

## Comparing mixed precision with double precision

After building, run all comparisons and emit CSV:

```sh
cd papers
./run_cuda_h100_comparison.sh > cuda_h100_comparison.csv
```

By default, each executable is first run once as a warm-up and then run
three measured times. The reported `time_ms` is the arithmetic mean of
the measured runs. The raw CSV also includes `time_ms_stddev`,
`time_ms_min`, `time_ms_max`, `time_ms_runs`, `warmup_runs`, and
`measured_runs`.

Optional environment variables:

```sh
COMBINATIONS="1 2" BACKPROP_SIZE=65536 DENSE_LU_SIZE=5000 HOTSPOT_ROWS=1024 HOTSPOT_COLS=1024 HOTSPOT_ITERS=2 ./run_cuda_h100_comparison.sh
BENCHMARKS=hotspot ./run_cuda_h100_comparison.sh
WARMUP_RUNS=1 MEASURED_RUNS=5 ./run_cuda_h100_comparison.sh
sbatch --account=${IDRPROJ}@h100 --export=ALL,PLOT_FONT_SIZE=12,RUN_PLOTS=1 submit_hpc_mix_h100.sh
```

For Hotspot, the comparison script generates deterministic input files when
the requested `HOTSPOT_TEMP_FILE` and `HOTSPOT_POWER_FILE` are missing.  The
Jean Zay batch script stores these generated inputs under
`h100_results/<job-id>/hotspot_inputs` so large 1024-by-1024 files do not have
to be committed to the repository.  Set `HOTSPOT_GENERATE_INPUTS=0` to require
pre-existing input files instead.

## Running on Jean Zay H100

Submit from `hpc-mix-bench/papers`:

```sh
sbatch --account=${IDRPROJ}@h100 \
  --export=ALL,REPO_DIR="$(pwd)/..",BENCHMARKS="dense_lu hotspot",COMBINATIONS="1 2",DENSE_LU_SIZE=5000,HOTSPOT_ROWS=1024,HOTSPOT_COLS=1024,HOTSPOT_ITERS=2,FORCE_REBUILD=1,WARMUP_RUNS=1,MEASURED_RUNS=3,RUN_PLOTS=1 \
  submit_hpc_mix_h100.sh
```

The job writes CSVs, generated Hotspot inputs, benchmark outputs, and figures
to:

```text
papers/h100_results/<slurm-job-id>/
```

For each mixed run, compute:

- speedup = double mean `time_ms` / mixed mean `time_ms`
- time ratio = mixed mean `time_ms` / double mean `time_ms`
- memory ratio = mixed `device_allocation_bytes` / double `device_allocation_bytes`

## Plotting

The Jean Zay batch script writes plots automatically to
`h100_results/<job-id>/figures`.  To regenerate them manually from
`papers/h100_results`, use:

```sh
python3 plot_h100_ratios.py <job-id> --font-size 12
python3 plot_h100_ratios_combined.py <job-id> --font-size 12
python3 plot_dense_lu_solution_errors.py <job-id> --font-size 12
python3 plot_h100_accuracy_errors.py <job-id> --font-size 12
```

`plot_dense_lu_solution_errors.py` reads
`dense_lu_solutions/double_solution.txt` and the corresponding
`digit<i>_<j>_solution.txt` files, writes `dense_lu_solution_errors.csv`,
and plots relative L2/Linf errors against the FP64 CUDA solution.
`plot_h100_ratios_combined.py` also writes
`<benchmark>_ratio_error_combinations_1_2_dual_axis.*` when the ratio CSV
contains accuracy columns, combining the performance ratios and validation
errors with left/right y-axes.
`plot_h100_accuracy_errors.py` reads `cuda_h100_ratios.csv` and plots the
recorded Backprop output-delta errors, Dense LU solution errors, and Hotspot
output-field errors.

These CUDA ports are intended for performance validation of the PROMISE
mixed-precision configurations on H100-class hardware. They do not
depend on CADNA/PROMISE runtime macros; the precision choices are baked
into the generated CUDA wrappers.
