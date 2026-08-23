# Backprop WMMA Complement

The direct backprop CUDA port in `papers/backprop` preserves the
PROMISE-transformed benchmark structure.  In that benchmark, the hidden layer is
small and the output layer has size one, so most work is not naturally a large
Tensor-Core GEMM.

This optional complement uses NVIDIA WMMA on tile-aligned matrix
multiplications to measure the performance ceiling for a Tensor-Core-friendly
backprop workload.  It is not run by the default `run_complement_h100.sh`
workflow because the representative PROMISE cases in `digit_case_manifest.csv`
do not expose the same large GEMM structure.  The default workflow therefore
marks Backprop as `not_applicable` for Tensor Cores and relies on the direct
`digit<i>_<j>` rerun for source-case-matched evidence.

Build:

```sh
make CUDA_ARCH=sm_90
```

Run:

```sh
./backprop_wmma 1024 128 50
```

The program prints parseable fields including `kernel_time_ms`,
`forward_time_ms`, `backward_time_ms`, `device_allocation_bytes`, and
`uses_tensor_core_candidate`.
