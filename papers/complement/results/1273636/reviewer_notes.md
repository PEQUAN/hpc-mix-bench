# Reviewer-Facing Notes for H100 Timing Questions

## Main interpretation

- The paper figures use direct CUDA ports of the PROMISE-transformed programs. They preserve the selected precision assignments but do not call cuBLAS, WMMA, or Tensor Core APIs.
- A time ratio greater than one is therefore possible: lower storage precision reduces allocation, but the direct kernels may pay for conversions, scalar rounding, extra casts to preserve the PROMISE arithmetic path, and kernel launch/synchronization overheads.
- Memory ratio and time ratio should be discussed separately.  The former is a direct consequence of the selected storage types; the latter depends on whether the lowered variables dominate the GPU execution.
- The complement job uses `digit_case_manifest.csv` to bind each rerun and Tensor-Core-complement row to an originating `digit<i>_<j>` PROMISE case. The default manifest covers all available first-two-combination digit-sweep cases for Backprop, Dense LU, and Hotspot.  Tensor-Core rows are emitted only when the source case has a suitable dense matrix update; otherwise the CSV marks the case as not applicable.

## Dense LU

- At digit 6 / Combination I, the measured time ratio is 0.979 and the memory ratio is 0.501.  The CPU transformed code has 57 FP64 variables and 13 lower-precision variables.
- At digit 6 / Combination II, the measured time ratio is 0.954 and the memory ratio is 0.501.  The generated CUDA wrapper uses dominant matrix storage `FP32`.
- The generated CUDA wrappers for dense LU digit1_6 and digit2_6 are identical except for the comment header; different timing values should not be interpreted as a precision-layout effect.
- The direct dense-LU CUDA implementation performs pivoting, row swaps, scaling, and element-wise trailing updates as separate kernels.  This is faithful to the PROMISE-transformed scalar code, but it is not a Tensor-Core-friendly blocked LU.  The complement benchmark therefore isolates a blocked trailing update to estimate the performance ceiling of a TC-enabled reformulation.

## Hotspot

- At digit 10 / Combination I, the measured time ratio is 0.891 and the memory ratio is 0.500.  The dominant grid storage in the CUDA wrapper is `FP32`.
- At digit 10 / Combination II, the measured time ratio is 0.973 and the memory ratio is 0.500.  The CPU transformed code has 13 FP64 variables and 18 lower-precision variables.
- The generated CUDA wrappers for hotspot digit1_10 and digit2_10 are identical except for the comment header.  Their timing difference should therefore be reported as measurement variability rather than a real Combination-I/Combination-II precision effect.
- The 13 FP64 variables in Hotspot are scalar physical constants or intermediate parameters in the transformed CPU program; in the CUDA port, the large arrays are `temp`, `power`, and `result`, whose storage is controlled by the generated field precision.  Hotspot is a stencil and is primarily memory-bandwidth/latency sensitive, so it is not a natural Tensor Core target.

## Suggested response wording

The original H100 validation did not use Tensor Cores.  It preserves the PROMISE-derived mixed-precision assignments in direct CUDA kernels to test the performance effect of the transformed program itself.  Time ratios slightly above one are caused by the fact that reduced storage does not remove all FP64 arithmetic and may introduce casts/rounding and synchronization costs. For Dense LU and Hotspot, many low-precision variables are scalar or do not map to Tensor-Core instructions in the direct implementation; consequently the memory savings are much clearer than the speedups.  We added complementary profiling and Tensor-Core-suitable experiments to separate this implementation effect from the hardware performance ceiling.

