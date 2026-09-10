===============================================
HPC-Mix-Bench: Mixed-Precision Benchmark Suite
===============================================

HPC-Mix-Bench is a benchmark and artifact repository for studying
mixed-precision tuning with PROMISE on C/C++ numerical applications. It
contains benchmark sources, PROMISE run templates, generated mixed-precision
programs, CPU precision-search outputs, H100 CUDA validation scripts, plotting
utilities, and complementary Tensor Core experiments.

The documentation is organized as a practical reference for reproducing the
experiments and extending the suite:

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   installation
   configuration
   example
   applications
   benchmark_results
   api_references
   licenses

What The Repository Provides
============================

* A curated set of C/C++ numerical kernels from linear algebra, optimization,
  numerical integration, Rodinia-style simulations, and machine learning.
* Shared PROMISE configuration templates for the precision combinations used in
  the paper experiments.
* Source-to-source mixed-precision outputs for significant-digit targets,
  stored in ``digit<i>_<j>`` directories where ``i`` is the precision
  combination and ``j`` is the required number of significant digits.
* One-bit custom precision sweeps for isolating the effects of exponent and
  trailing-significand bitwidth.
* Direct CUDA H100 ports of selected PROMISE-derived programs, together with
  performance, memory, and validation scripts.
* Complementary H100 experiments for profiling and Tensor Core comparison.

Precision Formats
=================

The standard experiments use four precision combinations. Each combination is
searched from lower to higher precision together with FP32 and FP64:

.. list-table:: Precision combinations
   :widths: 20 40 40
   :header-rows: 1

   * - Combination
     - Formats
     - Typical use in this repository
   * - I
     - E5M2, FP16, FP32, FP64
     - FP8-like dynamic range plus FP16 intermediate storage
   * - II
     - E5M2, BF16, FP32, FP64
     - FP8-like dynamic range plus BF16 intermediate storage
   * - III
     - E4M3, FP16, FP32, FP64
     - Alternative FP8-like significand/range balance
   * - IV
     - E4M3, BF16, FP32, FP64
     - E4M3 combined with BF16

PROMISE validates candidate configurations against CADNA/DSA numerical
references and searches for reduced-precision assignments that satisfy a
user-specified number of correct significant digits.
