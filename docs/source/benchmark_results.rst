=================
Benchmark Results
=================

This section presents the complete empirical results and analysis from our mixed-precision benchmarking study. All experiments use the PROMISE tool to automatically determine optimal precision configurations across four format combinations.

.. contents:: Table of Contents
   :local:
   :depth: 3

Precision Combinations
======================

All benchmarks are evaluated across four precision combinations:

.. list-table:: Floating-Point Format Combinations
   :widths: 15 20 20 20 20
   :header-rows: 1

   * - Combination
     - Format 1
     - Format 2
     - Format 3
     - Format 4
   * - **I**
     - E5M2
     - FP16
     - FP32
     - FP64
   * - **II**
     - E5M2
     - BF16
     - FP32
     - FP64
   * - **III**
     - E4M3
     - FP16
     - FP32
     - FP64
   * - **IV**
     - E4M3
     - BF16
     - FP32
     - FP64

Methodology
===========

PROMISE is evaluated across four precision combinations (I–IV) to examine the number of variables of each type in the transformed codes it provided. In our experiments the number of required digits in the results ranges from 1 to 10.

The following figures also present the computation time of PROMISE in seconds for each required accuracy, including the time to compute the reference result and the time to apply the delta debugging algorithm several times, compiling and executing the code with the tested distribution each time. For each target accuracy level, the total computation time of PROMISE (including compilation and execution) is represented by the red curve in each figure as follows, and the variations indicate the variations in the number of total compilations and executions of the code in the delta debugging process.

Benchmark Code Availability
----------------------------

The benchmark code is publicly available at https://github.com/PEQUAN/hpc-mix-bench.

Rodinia Simulations
===================

We evaluate four Rodinia workloads: Backprop, HotSpot, Particle Filter, and SRAD. These tests are performed across four precision combinations (I–IV) to reflect precision usage for results with varying required accuracies, ranging from 1 to 10 correct digits.

Backprop
--------

The error gradients of the Backprop kernel achieve at most 6 correct significant digits across four precision combinations. Given the required accuracy across the number of correct significant digits, the kernel is governed by two primary numerical sensitivities: long dot-product reductions in the forward and backward passes, and the nonlinear sigmoid activation whose exponential evaluation is highly sensitive to small input perturbations. Precision upgrades therefore concentrate first on the sigmoid pre-activation sums and the activation path, then on the error-delta computations, and finally on the weight-update accumulators when a higher number of digits is demanded.

.. list-table:: Backprop Results
   :widths: 50 50
   :header-rows: 0

   * - .. image:: https://raw.githubusercontent.com/PEQUAN/hpc-mix-bench/main/papers/plots/precision1_backprop_runtime.jpg
          :width: 100%
          :alt: Combination I
     - .. image:: https://raw.githubusercontent.com/PEQUAN/hpc-mix-bench/main/papers/plots/precision2_backprop_runtime.jpg
          :width: 100%
          :alt: Combination II
   * - **Combination I (E5M2, FP16, FP32, FP64)**
     - **Combination II (E5M2, BF16, FP32, FP64)**
   * - .. image:: https://raw.githubusercontent.com/PEQUAN/hpc-mix-bench/main/papers/plots/precision3_backprop_runtime.jpg
          :width: 100%
          :alt: Combination III
     - .. image:: https://raw.githubusercontent.com/PEQUAN/hpc-mix-bench/main/papers/plots/precision4_backprop_runtime.jpg
          :width: 100%
          :alt: Combination IV
   * - **Combination III (E4M3, FP16, FP32, FP64)**
     - **Combination IV (E4M3, BF16, FP32, FP64)**

**Figure**: Precision configurations for Backprop with varying requested accuracies.

Analysis
~~~~~~~~

In combinations I and II, the low-digit range (1 and 2 required digits) sees the first major transition: the sigmoid input path and its pre-activation accumulator are promoted from E5M2 to FP16 (in Combination I) or to BF16 (in Combination II) to stabilize the exponential evaluation near the steep region of the sigmoid. Between digits 3 and 6 both combinations enter a plateau where the reduction accumulators in the layer forward pass and error back-propagation remain at mid-precision levels. Once the sigmoid boundary is protected early, the remaining error budget is dominated by reductions that can be tolerated without further promotions. Combination I (with FP16) stabilizes the sigmoid and delta steps slightly earlier than Combination II because FP16 provides better mantissa support in the nonlinear path than BF16.

When the lowest precision changes to E4M3 in combinations III and IV, the low-digit regime (digits 1 to 3) keeps the sigmoid-related intermediates and initial reductions locked in E4M3 longer, increasing the risk of bias in the activation. The dominant promotion then sharpens between digits 4 and 7, where the pre-sigmoid sums and key accumulators in ``hidden_error`` move from E4M3 to FP16 (Combination III) or BF16 (Combination IV), and the reduction-critical variables are later escalated to FP32. Beyond this band further tightening primarily protects the weight-adjustment path (``new_dw`` in ``adjust_weights``). Combination IV (E4M3+BF16) requires the second promotion to FP32 earlier than Combination III because BF16 offers less protection against accumulation noise in the long reductions.

HotSpot
-------

HotSpot is numerically dominated by an explicit stencil update that repeatedly computes a local temperature increment—a scaled sum of the power term, the discrete Laplacians in the row and column directions, and the ambient coupling. This structure creates two distinct precision sensitivities: the per-cell update path exercised on every timestep (which accumulates rounding errors over many iterations) and the coefficient-preconditioning path that builds the geometry- and material-dependent scalars :math:`Cap`, :math:`R_x`, :math:`R_y`, :math:`R_z`, and their reciprocals.

.. list-table:: HotSpot Results
   :widths: 50 50
   :header-rows: 0

   * - .. image:: https://raw.githubusercontent.com/PEQUAN/hpc-mix-bench/main/papers/plots/precision1_hotspot_runtime.jpg
          :width: 100%
          :alt: Combination I
     - .. image:: https://raw.githubusercontent.com/PEQUAN/hpc-mix-bench/main/papers/plots/precision2_hotspot_runtime.jpg
          :width: 100%
          :alt: Combination II
   * - **Combination I (E5M2, FP16, FP32, FP64)**
     - **Combination II (E5M2, BF16, FP32, FP64)**
   * - .. image:: https://raw.githubusercontent.com/PEQUAN/hpc-mix-bench/main/papers/plots/precision3_hotspot_runtime.jpg
          :width: 100%
          :alt: Combination III
     - .. image:: https://raw.githubusercontent.com/PEQUAN/hpc-mix-bench/main/papers/plots/precision4_hotspot_runtime.jpg
          :width: 100%
          :alt: Combination IV
   * - **Combination III (E4M3, FP16, FP32, FP64)**
     - **Combination IV (E4M3, BF16, FP32, FP64)**

**Figure**: Precision configurations for HotSpot with varying requested accuracies.

Analysis
~~~~~~~~

Across all four combinations, PROMISE consistently retains FP32/FP64 for the loop-carried state arrays while selectively lowering the precision of these scalars and selected intermediates. The observed transitions with increasing required correct digits are therefore best read as a progressive tightening of (a) the scaling factors that control update stability and (b) the cancellation-prone neighbor-difference patterns that define the Laplacians.

In Combinations I and II, a clear shift occurs between one and six correct digits: the "fast" scalars (:math:`step`, :math:`Cap_1`, reciprocal resistances) remain in E5M2 while :math:`Cap` (and sometimes :math:`max\_slope`) moves to FP16. This is meaningful—:math:`Cap` multiplies several physical constants and grid-size ratios, so coarse quantization biases the effective timestep. Between seven and ten correct digits, more variables are promoted to FP32, revealing that the dominant error has moved from local rounding to systematic drift in global scaling.

The E4M3 combinations sharpen the picture further. In both combinations III and IV, the major promotion to higher precision occurs one digit tier earlier, with most coefficient-related variables already upgraded by digits 1–6 because E4M3's narrower dynamic range forces earlier upgrades to the coefficient path. Between seven and ten correct digits, the two then diverge: FP16 improves mantissa accuracy in scaling products, while BF16 better preserves stable reciprocals when the grid is large or :math:`step` is tiny. At the most stringent targets, the conversions increasingly pin the timestep scalars to FP32, showing that correctness is governed less by any single ulp error and more by whether the global amplification factor :math:`Cap_1` remains unbiased across iterations.

Particle Filter
---------------

The Particle Filter kernel exhibits a mixed-precision sensitivity pattern driven primarily by control-flow decisions rather than long reduction chains. Fragility stems from nonlinear operations in the stochastic proposal (log, square root, cosine for Gaussian sampling), a squared-residual observation model, and an exponential reweighting followed by weight normalization and Cumulative Distribution Function (CDF) construction. Because resampling depends on comparing uniform samples against the CDF, small errors can change particle ancestry and redirect trajectories. Precision usage, therefore, concentrates on nonlinear choke points and at the normalization/resampling boundary rather than being applied uniformly.

.. list-table:: Particle Filter Results
   :widths: 50 50
   :header-rows: 0

   * - .. image:: https://raw.githubusercontent.com/PEQUAN/hpc-mix-bench/main/papers/plots/precision1_particle_filter_runtime.jpg
          :width: 100%
          :alt: Combination I
     - .. image:: https://raw.githubusercontent.com/PEQUAN/hpc-mix-bench/main/papers/plots/precision2_particle_filter_runtime.jpg
          :width: 100%
          :alt: Combination II
   * - **Combination I (E5M2, FP16, FP32, FP64)**
     - **Combination II (E5M2, BF16, FP32, FP64)**
   * - .. image:: https://raw.githubusercontent.com/PEQUAN/hpc-mix-bench/main/papers/plots/precision3_particle_filter_runtime.jpg
          :width: 100%
          :alt: Combination III
     - .. image:: https://raw.githubusercontent.com/PEQUAN/hpc-mix-bench/main/papers/plots/precision4_particle_filter_runtime.jpg
          :width: 100%
          :alt: Combination IV
   * - **Combination III (E4M3, FP16, FP32, FP64)**
     - **Combination IV (E4M3, BF16, FP32, FP64)**

**Figure**: Precision configurations for Particle Filter with varying requested accuracies.

Analysis
~~~~~~~~

Under Combinations I and II that use E5M2 as the lowest precision, the low-digit regime (1 and 2 required digits) sees the first major upgrade: the random-number generation and nonlinear transforms (Gaussian proposal and exponential reweighting) shift from E5M2 to FP16 or BF16 to avoid bias in distribution tails and weight collapse. A second transition occurs between digits 2 and 3, where global mass conservation becomes limiting; the weight summation and normalization step requires higher precision (often FP32) because relative errors propagate coherently through the CDF and affect all subsequent resampling decisions.

When the lowest precision changes to E4M3, transitions appear earlier. For 1 and 2 required digits, the exponential reweighting stresses range and scale more quickly, prompting an earlier move from E4M3 to FP16 or BF16 around the likelihood-to-weight mapping. Beyond this, improvements through the mid-digit range (digits 2 to 4) focus on stabilizing the cumulative sum and CDF construction, with the E4M3+BF16 combination often protecting resampling thresholds while E4M3+FP16 more frequently safeguards the exponential neighborhood to prevent premature weight degeneracy. Overall, the pipeline shows a clear staged allocation: stochastic sampling first, then nonlinear reweighting, and finally normalization and resampling, with digit thresholds shifting according to the low-precision format's exponent and significand characteristics.

SRAD
----

.. list-table:: SRAD Results
   :widths: 50 50
   :header-rows: 0

   * - .. image:: https://raw.githubusercontent.com/PEQUAN/hpc-mix-bench/main/papers/plots/precision1_srad_v2_runtime.jpg
          :width: 100%
          :alt: Combination I
     - .. image:: https://raw.githubusercontent.com/PEQUAN/hpc-mix-bench/main/papers/plots/precision2_srad_v2_runtime.jpg
          :width: 100%
          :alt: Combination II
   * - **Combination I (E5M2, FP16, FP32, FP64)**
     - **Combination II (E5M2, BF16, FP32, FP64)**
   * - .. image:: https://raw.githubusercontent.com/PEQUAN/hpc-mix-bench/main/papers/plots/precision3_srad_v2_runtime.jpg
          :width: 100%
          :alt: Combination III
     - .. image:: https://raw.githubusercontent.com/PEQUAN/hpc-mix-bench/main/papers/plots/precision4_srad_v2_runtime.jpg
          :width: 100%
          :alt: Combination IV
   * - **Combination III (E4M3, FP16, FP32, FP64)**
     - **Combination IV (E4M3, BF16, FP32, FP64)**

**Figure**: Precision configurations for SRAD with varying requested accuracies.

The SRAD kernel exhibits a mixed-precision sensitivity pattern that is tightly coupled to its nonlinear diffusion pathway. The algorithm begins by exponentially mapping the input random matrix into the working image :math:`J`, then computes global Region of Interest (ROI) statistics to establish a baseline speckle noise level :math:`q_0^2`. For each pixel it constructs normalized directional measures

.. math::

   G^2 = \frac{d_N^2 + d_S^2 + d_W^2 + d_E^2}{J_c^2}, \qquad
   L = \frac{d_N + d_S + d_W + d_E}{J_c}

followed by the rational chain that yields :math:`q^2` and the diffusion coefficient :math:`c(\cdot)`. Because :math:`c` is an explicit nonlinear function of :math:`(q^2 - q_0^2)` and is immediately clamped to :math:`[0,1]`, even modest rounding errors introduced in the early exponential step, the ROI accumulation, or the local normalizations can push coefficients across saturation boundaries; the subsequent update :math:`J \leftarrow J + 0.25\,\lambda\,D` then propagates these discrete decisions through repeated iterations, making the diffusion-coefficient route the dominant numerical bottleneck.

Analysis
~~~~~~~~

Under Combination I and Combination II, the principal precision pressure concentrates in the coefficient pathway—the variables that participate in the :math:`J_c` normalization, the formation of :math:`G^2` and :math:`L`, and the downstream rational and nonlinear transforms leading to :math:`q^2` and :math:`c`. The dominant transition occurs between digits 3 and 6: for required correct digits below this range (i.e., digits 1–2), the coefficient route is forced to rely on the most aggressive low-precision formats, so divisions by :math:`J_c` and the squaring/accumulation steps suffer from coarse quantization that drives :math:`c` toward saturation after clamping; from digits 3–6 onward the required formats shift toward FP32-like stability, sharply reducing saturation-driven discreteness and restoring diffusion behavior that matches the FP64 reference more closely. In the low-digit regime (digits 1 to 3) both combinations keep the exponential initialization and ROI moments in E5M2, but Combination II (with BF16) moves the normalization-sensitive intermediates off the lowest tier slightly earlier than Combination I.

When the lowest precision changes to E4M3 in combinations III and IV, the same coefficient route remains the bottleneck, but the digit thresholds compress and the saturation-prone regime appears earlier. In the low-digit regime (digits 1 to 3) the normalization and coefficient variables stay locked in the most aggressive formats far more often, increasing the chance that :math:`c` is driven to its clamped extremes and that these extremes reinforce across iterations.

The main qualitative improvement becomes most noticeable between digits 4 and 7, when PROMISE increases the precision of :math:`G^2`, :math:`L`, :math:`q^2`, and the diffusion coefficient :math:`c` from E4M3 to FP16 or BF16. At this point, the diffusion field shifts from being heavily discretized and dominated by saturation, where rounding errors often push :math:`c` to the extreme values of 0 or 1, to a mode where values change smoothly and continuously. When more than 7 correct digits are needed, further increases in precision mostly help reduce leftover numerical bias and rounding errors, but do not change the overall nature of the diffusion process. As before, Combination IV (E4M3+BF16) needs fewer digits than Combination III before normalization-sensitive scalars are treated like FP32, again showing the same range and normalization pressure points.

LU Factorization
================

Our analysis of LU factorization follows two tracks: dense and sparse linear systems. One faces compute-intensive operations with regular memory access, while the other deals with irregular memory access, fill-in control, and pivoting overhead. Both LU implementations leverage pivoting; specifically, sparse LU uses the RCM ordering to reduce fill-in and improve cache locality.

Solving a linear system via LU factorization involves the following steps: (i) factorization of the matrix in :math:`L` and :math:`U`; (ii) forward substitution to solve :math:`Ly = b`; (iii) backward substitution to solve :math:`Ux = y`. In practice, this process is typically preceded by *pivoting*, where the rows (and/or columns) of the matrix are permuted to improve numerical stability and avoid division by zero. This leads to a factorization of the form :math:`PA = LU` where :math:`P` is a permutation matrix.

Dense LU
--------

.. list-table:: Dense LU Results
   :widths: 50 50
   :header-rows: 0

   * - .. image:: https://raw.githubusercontent.com/PEQUAN/hpc-mix-bench/main/papers/plots/precision1_dense_lu_runtime.jpg
          :width: 100%
          :alt: Combination I
     - .. image:: https://raw.githubusercontent.com/PEQUAN/hpc-mix-bench/main/papers/plots/precision2_dense_lu_runtime.jpg
          :width: 100%
          :alt: Combination II
   * - **Combination I (E5M2, FP16, FP32, FP64)**
     - **Combination II (E5M2, BF16, FP32, FP64)**
   * - .. image:: https://raw.githubusercontent.com/PEQUAN/hpc-mix-bench/main/papers/plots/precision3_dense_lu_runtime.jpg
          :width: 100%
          :alt: Combination III
     - .. image:: https://raw.githubusercontent.com/PEQUAN/hpc-mix-bench/main/papers/plots/precision4_dense_lu_runtime.jpg
          :width: 100%
          :alt: Combination IV
   * - **Combination III (E4M3, FP16, FP32, FP64)**
     - **Combination IV (E4M3, BF16, FP32, FP64)**

**Figure**: Precision configurations for dense LU factorization with varying requested accuracies.

The empirical results of the chosen precision configuration for the dense LU linear solve by PROMISE across 10 correct digits are presented in the figure above. The reduced precisions under FP64 are relatively stable. As demonstrated in Combination I and II, the use of the 8-bit format is consistent across digits, achieving more than 20% of the total variables in use. Similarly, Combinations III and IV exhibit a similar profile, the 8-bit format remains largely consistent across digits, with relatively 55∼57 variables in FP64 precision across 10 digits, focusing on calculation of forward and backward substitutions similar to Combinations I and II. For all precision combinations, the 8-bit format is mostly used in pivoting across digits. Compared to E4M3, E5M2 shows an extra two variables in use at digit 3 for finding the pivot row. Besides, similar results have been obtained with matrices with a condition number of 10. The use of FP32 takes 2∼4 variables, focusing on the steps of the accumulator in norm calculation and the upper diagonal matrix formation—the elimination of elements below the diagonal element. When the required number of correct digits increases, the precision used in the upper diagonal matrix formation gradually shifts from FP32 to FP64.

Sparse LU
---------

.. list-table:: Sparse LU Results
   :widths: 50 50
   :header-rows: 0

   * - .. image:: https://raw.githubusercontent.com/PEQUAN/hpc-mix-bench/main/papers/plots/precision1_sparse_lu_runtime.jpg
          :width: 100%
          :alt: Combination I
     - .. image:: https://raw.githubusercontent.com/PEQUAN/hpc-mix-bench/main/papers/plots/precision2_sparse_lu_runtime.jpg
          :width: 100%
          :alt: Combination II
   * - **Combination I (E5M2, FP16, FP32, FP64)**
     - **Combination II (E5M2, BF16, FP32, FP64)**
   * - .. image:: https://raw.githubusercontent.com/PEQUAN/hpc-mix-bench/main/papers/plots/precision3_sparse_lu_runtime.jpg
          :width: 100%
          :alt: Combination III
     - .. image:: https://raw.githubusercontent.com/PEQUAN/hpc-mix-bench/main/papers/plots/precision4_sparse_lu_runtime.jpg
          :width: 100%
          :alt: Combination IV
   * - **Combination III (E4M3, FP16, FP32, FP64)**
     - **Combination IV (E4M3, BF16, FP32, FP64)**

**Figure**: Precision configurations for sparse LU factorization with varying requested accuracies.

Similar to the dense LU implementation, the linear solve with sparse LU exhibits a relatively smooth precision transform across the digits. The 8-bit format (E5M2 and E4M3) exhibits similar patterns across digits, and remains 11 variables except that E4M3, compared to E5M2, shows an extra two variables in use (in terms of Combination III and IV) at digit 1; Here the extra usages of E4M3 (instead of E5M2) are located at two places. One is the temporary swapping value used during the column sorting step inside the matrix permutation procedure; the other is the sum value accumulated for the residual before division inside the backward substitution step, in the sense that a calculation with a smaller unit-roundoff is required. From the transformed code, we found that the 8-bit format is used primarily for pivoting in Combinations I∼IV. There is only one variable in the precision of FP32 from digit 8 to 10, which is used for the accumulator for the calculation of the norm. Before digit 8, back substitution requires residual calculation and division, so it uses higher precision, with FP8 (E5M2, resp. E4M3) or half (FP16, resp. BF16) precision from digit 1 to 3, and FP32 precision from digit 7 (with FP64 precision as accumulator), and FP64 precision from digit 8 to 10. The FP32 use takes 3-4 variables across digits, and the four precision combinations show 5 FP32 variables in use between digits 1 and 3, and 4 between 4 to 5 digits, while remains 3 FP32 variables in the rest of digits. The use of FP32 focuses on accumulators of the norm calculation.

Summary
=======

Key findings from our benchmark results:

**Rodinia Simulations**

* Neural networks (Backprop) achieve at most 6 correct digits with aggressive mixed precision
* Thermal simulations (HotSpot) require careful coefficient preconditioning
* Particle filters show staged precision allocation: sampling → reweighting → normalization
* SRAD exhibits clear transitions from discretized to continuous diffusion behavior

**LU Factorization**

* Pivoting operations safely use 8-bit precision across all accuracy requirements
* Dense and sparse LU show similar three-tier precision strategies
* Forward/backward substitution requires FP64 for high accuracy (8-10 digits)
* Precision requirements relatively stable across condition numbers

**Format Comparisons**

* E5M2 vs E4M3: E4M3 requires earlier precision upgrades due to narrower dynamic range
* FP16 vs BF16: FP16 better for mantissa-sensitive operations; BF16 for range-sensitive operations

**General Trends**

* Low digits (1-3): Aggressive precision reduction, widespread 8-bit usage
* Mid digits (4-7): Progressive upgrades, shift from 8-bit to 16-bit formats
* High digits (8-10): Dominance of FP32/FP64, minimal low-precision usage

Next Steps
==========

* Learn how to run benchmarks: :doc:`quickstart`
* Understand the tool API: :doc:`api_reference`
* Contribute your own benchmarks: :doc:`contributing`