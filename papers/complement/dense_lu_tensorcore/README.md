# Dense LU Tensor Core Complement

The direct dense-LU CUDA port in `papers/dense_lu` follows the scalar
PROMISE-transformed factorization: pivoting, scaling, and element-wise trailing
updates are preserved as CUDA kernels.  That implementation is the one used for
the paper figures.

This complement isolates the blocked trailing update

```text
C := C - L21 * U12
```

which is the portion of a blocked LU factorization that can be mapped cleanly to
GEMM and therefore to H100 Tensor Cores.  It should be interpreted as a
hardware-ceiling experiment for a Tensor-Core-friendly reformulation of the
dense-LU update, not as the original direct CUDA port.

In the default workflow, this kernel is run only for the Dense LU source cases
listed in `../digit_case_manifest.csv`; the output CSV keeps the originating
`source_case` so the result remains tied to the corresponding `digit<i>_<j>`
precision configuration.

Build:

```sh
make CUDA_ARCH=sm_90
```

Run:

```sh
./dense_lu_panel_update_tc 5000 64 fp64 1 5
./dense_lu_panel_update_tc 5000 64 tf32 1 5
./dense_lu_panel_update_tc 5000 64 fp16 1 5
./dense_lu_panel_update_tc 5000 64 bf16 1 5
```

The program prints parseable fields including `kernel_time_ms`,
`device_allocation_bytes`, `relative_l2_error_vs_fp64`,
`relative_linf_error_vs_fp64`, `gflops`, and `uses_tensor_core_candidate`.
