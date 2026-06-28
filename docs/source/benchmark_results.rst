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
