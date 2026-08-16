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

Dense LU accepts an optional matrix size and defaults to `500`:

```sh
./dense_lu/dense_lu_cuda_double 500
./dense_lu/digit2_5/dense_lu_cuda 500
```

Hotspot keeps the original benchmark interface:

```sh
./hotspot/hotspot_cuda_double 512 512 2 hotspot/temp_512 hotspot/power_512 hotspot/output_cuda_double.out
./hotspot/digit2_5/hotspot_cuda 512 512 2 hotspot/temp_512 hotspot/power_512 hotspot/output_digit2_5.out
```

Both the mixed-precision and double-precision executables print timing
and memory fields with the same names:

- Backprop: `kernel_time_ms`
- Dense LU: `factorization_time_ms`
- Hotspot: `kernel_time_ms`
- All benchmarks: `device_allocation_bytes` and `device_allocation_mib`

## Comparing mixed precision with double precision

After building, run all comparisons and emit CSV:

```sh
cd mp_tests
./run_cuda_h100_comparison.sh > cuda_h100_comparison.csv
```

By default, each executable is first run once as a warm-up and then run
three measured times. The reported `time_ms` is the arithmetic mean of
the measured runs. The raw CSV also includes `time_ms_stddev`,
`time_ms_min`, `time_ms_max`, `time_ms_runs`, `warmup_runs`, and
`measured_runs`.

Optional environment variables:

```sh
BACKPROP_SIZE=65536 DENSE_LU_SIZE=500 HOTSPOT_ITERS=2 ./run_cuda_h100_comparison.sh
BENCHMARKS=hotspot ./run_cuda_h100_comparison.sh
WARMUP_RUNS=1 MEASURED_RUNS=5 ./run_cuda_h100_comparison.sh
```

For each mixed run, compute:

- speedup = double mean `time_ms` / mixed mean `time_ms`
- time ratio = mixed mean `time_ms` / double mean `time_ms`
- memory ratio = mixed `device_allocation_bytes` / double `device_allocation_bytes`

These CUDA ports are intended for performance validation of the PROMISE
mixed-precision configurations on H100-class hardware. They do not
depend on CADNA/PROMISE runtime macros; the precision choices are baked
into the generated CUDA wrappers.
