# Complementary H100 Experiments

This directory contains reviewer-facing complementary experiments for the
PROMISE-derived CUDA ports under `papers/`.

The main H100 results intentionally use direct CUDA ports of the
PROMISE-transformed mixed-precision programs.  They preserve the source-level
precision choices produced by PROMISE and do not use vendor linear-algebra
libraries, WMMA, or Tensor Core APIs.  Therefore, a time ratio close to or above
one is possible when the selected low-precision variables do not dominate the
runtime, when casts/rounding are frequent, or when the kernel is memory-bound.

The scripts here provide two additional pieces of evidence:

- **T1 profiling/decomposition**: rerun the direct CUDA implementations for
  every available `digit1_*` and `digit2_*` case in the manifest, summarize
  kernel time, end-to-end executable time, non-kernel overhead, memory
  footprint, dominant storage types, and identical/different generated
  configurations.
- **T2 Tensor Core complement**: for source cases whose PROMISE-derived
  algorithmic structure contains a Tensor-Core-suitable dense update, run the
  corresponding accelerated kernel and record the originating `digit<i>_<j>`
  case.  For cases such as Hotspot stencils or the small-output Backprop
  configuration, the output CSV records that Tensor Cores are not applicable
  instead of pretending that a GEMM surrogate is the same experiment.

## Contents

```text
complement/
├── analyze_h100_results.py
├── digit_case_manifest.csv
├── plot_complement_h100.py
├── run_complement_h100.sh
├── submit_complement_h100.sh
├── dense_lu_tensorcore/
│   ├── dense_lu_panel_update_tc.cu
│   └── Makefile
└── backprop_tensorcore/
    ├── backprop_wmma.cu
    └── Makefile
```

## Running on Jean Zay H100

Submit from `hpc-mix-bench/papers`:

```sh
sbatch --account=${IDRPROJ}@h100 \
  --export=ALL,REPO_DIR="$(pwd)/..",COMBINATIONS="1 2",BACKPROP_SIZE=65536,DENSE_LU_SIZE=5000,HOTSPOT_ROWS=1024,HOTSPOT_COLS=1024,HOTSPOT_ITERS=2,WARMUP_RUNS=1,MEASURED_RUNS=5,RUN_PLOTS=1 \
  complement/submit_complement_h100.sh
```

For Hotspot, deterministic 1024-by-1024 input files are generated under
`papers/complement/results/<slurm-job-id>/hotspot_inputs` when they are not
already present.  Set `HOTSPOT_GENERATE_INPUTS=0` to require pre-existing input
files instead.

The job writes outputs to:

```text
papers/complement/results/<slurm-job-id>/
```

Important output files:

- `existing_h100_summary.csv`: existing direct-port ratios plus precision
  wrapper information.
- `reviewer_notes.md`: concise interpretation of the reviewer questions.
- `direct_case_rerun.csv`: rerun of all `digit1_*` and `digit2_*` cases listed
  in `digit_case_manifest.csv`, plus FP64 baselines.  Backprop rows include
  MSE and relative L2/Linf error for the PROMISE-marked output-delta array
  relative to the FP64 CUDA baseline.  Dense LU rows include residual/error
  metrics and L2/Linf solution error relative to the FP64 CUDA solution.
  Hotspot rows include L2/Linf output-field error relative to the FP64 CUDA
  output.
- `direct_case_ratios.csv`: ratio table with the same schema as
  `cuda_h100_ratios.csv`, with the Backprop, Dense LU, and Hotspot accuracy
  columns preserved for result validation.
- `profile_breakdown.csv`: coarse profiling decomposition with kernel
  computation time, end-to-end executable time, and non-kernel overhead.
- `tensorcore_complement.csv`: Tensor Core complement measurements with
  `source_benchmark` and `source_case` columns.  Non-applicable cases are
  explicitly marked as `not_applicable`; dense LU Tensor Core rows include
  relative L2/Linf error against a FP64 cuBLAS reference update.
- `tensorcore_ratio_summary.csv`: compact Tensor Core ratio/error table derived
  from `tensorcore_complement.csv`.
- `dense_lu_solution_errors.csv`: direct dense LU solution-vector errors
  against the FP64 CUDA solution, generated when saved solution vectors are
  present.
- `figures/`: direct CUDA time/memory plots, dual-axis direct
  performance/error plots, Backprop output-delta error plots, Dense LU
  solution-error plots, and Hotspot output-error plots, plus Tensor Core
  complement ratio/error plots.

To regenerate the complement plots manually from `papers/complement`, run:

```sh
python3 plot_complement_h100.py results/<job-id> --font-size 12
```

To regenerate the direct dense LU solution-error figure from
`papers/h100_results`, run:

```sh
python3 plot_dense_lu_solution_errors.py <job-id> --font-size 12
python3 plot_h100_accuracy_errors.py --csv ../complement/results/<job-id>/direct_case_ratios.csv --font-size 12
```

Set `ENABLE_NCU=1` to additionally collect Nsight Compute reports for selected
direct CUDA cases when `ncu` is available on the node.

## Profiling scope

The default direct CUDA executables report the timed kernel region
(`kernel_time_ms` or `factorization_time_ms`).  The complement runner also
measures wall-clock time around each executable invocation, so it can separate:

- pure GPU computation time: reported CUDA-event kernel/factorization time;
- non-kernel overhead: wall-clock time minus reported kernel time.

Exact separation of host-device memory transfer time and host/device type
conversion time requires instrumented regions inside each benchmark
implementation, as illustrated by `hotspot/cuda_hotspot_h100_profiled.cuh`.
For full internal decomposition, the same pattern should be added around
`cudaMemcpy` calls, host conversion loops, and kernel launches in Backprop and
Dense LU, or collected with Nsight Systems/Nsight Compute traces on the H100
node.
