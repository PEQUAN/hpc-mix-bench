=================================
Benchmark Methodology and Value
=================================

HPC-MIX Bench is useful when a precision method must work across more than
one favorable kernel. The suite spans algorithms with different data access,
iteration, conditioning, and output sensitivity. Its artifacts help compare
**how much precision can be reduced**, **what accuracy remains**, **how much
search costs**, and, for selected H100 ports, **what changes on hardware**.

Research questions the suite supports
=====================================

* **Tuning-method evaluation:** Does a search strategy find useful assignments
  across stencils, solvers, factorizations, and machine-learning workloads?
* **Format design:** Does E5M2's extra exponent bit help more than E4M3's extra
  trailing-significand bit for a particular numerical structure?
* **Accuracy budgets:** How do selected variable types change as the requested
  number of correct significant digits increases?
* **Compiler and hardware work:** Which PROMISE-derived assignments are worth
  porting, and where do conversion overhead or unavailable native operations
  erase the expected gain?
* **Reproducibility studies:** Can another machine reproduce the precision
  counts, search behavior, and application-level validation for a fixed input?

Choosing a workload
===================

Select a family based on the numerical behavior you want to probe. The table
offers starting points, not claims that every case in a family behaves alike.

.. list-table:: Workload selection guide
   :widths: 22 31 47
   :header-rows: 1

   * - Family
     - Examples
     - Useful precision question
   * - Factorization and dense algebra
     - ``dense_lu``, ``qr``, ``lud``
     - How do pivots, dependent updates, and matrix conditioning constrain
       low-precision arithmetic?
   * - Iterative linear solvers
     - ``cg``, ``bicgstab``, ``gmres_tol1``, ``ir3``
     - How do residual checks, repeated updates, and stopping criteria respond
       to rounding?
   * - Stencils and physical simulation
     - ``hotspot``, ``hotspot3D``, ``srad_v2``, ``cfd``
     - Does reduced storage help a regular or data-parallel computation after
       conversion costs are included?
   * - Interpolation and integration
     - ``cubic_spline``, ``rk4``, ``simpson``, ``trapezoidal``
     - How does local error propagate across stages or sample points?
   * - Learning and data analysis
     - ``backprop``, ``kmeans``, ``pca``, ``svm``, ``randomforest``
     - Which state and intermediate values tolerate narrowing while retaining
       the checked model output?

For the full inventory, including nested benchmark variants, see
:doc:`applications`. A compact first comparison is ``dense_lu`` versus
``hotspot``: one has dependent factorization updates; the other is a regular
stencil. ``backprop`` adds a learning workload and has a selected H100 port.

A repeatable CPU study
======================

#. **State the question.** Pick benchmark directories, exact inputs, a set of
   significant-digit targets, and the precision combinations to compare.
#. **Record the environment.** Save the repository revision, compiler and
   optimization flags, Python/PROMISE/CADNA versions, CPU model, thread counts,
   and relevant floating-point settings.
#. **Review each ``promise.yml``.** It fixes the source files, compile command,
   executable, and output path. Check that the benchmark validates the output
   you care about. See :doc:`configuration`.
#. **Prepare the templates.** From ``mp_tests/``, ``bash sync_settings.sh``
   copies the shared files into top-level benchmark folders. Its default mode
   deletes existing local ``run_setting_*.py``, ``run_debug_*.sh``, and
   ``fp.json`` files before copying; preserve any custom versions first.
#. **Run a small sequential sweep first.** This checks that the compiler,
   CADNA, input files, and output checks work before a large batch.
#. **Inspect the logs and artifacts.** Confirm every target yielded a valid
   precision assignment. Do not interpret an empty assignment or zero runtime
   as a successful low-cost result.
#. **Repeat and validate independently.** Use additional inputs or an
   application-level residual/error metric where possible. Re-run timing
   measurements instead of relying on one observed wall-clock value.

Example from ``mp_tests/`` after setup:

.. code-block:: bash

   bash sync_settings.sh
   ./run_benchmarks.sh true true false dense_lu hotspot
   python3 calculate_stats.py dense_lu hotspot

The runner supports ``--parallel``, but its setting tasks currently share each
benchmark's working directory. For clean comparisons, run those settings
sequentially or isolate each task's files and generated outputs first. The MPI
runner has a different folder-discovery rule; pass explicit folders and check
its task count before a distributed run.

What each result measures
=========================

.. list-table:: Keep these quantities separate
   :widths: 26 42 32
   :header-rows: 1

   * - Quantity
     - Source in this repository
     - Interpretation
   * - Precision assignment
     - ``prec_setting_<i>.json`` and precision-count plots
     - Number and identities of selected variables by type for a digit target.
   * - PROMISE search wall time
     - ``runtimes<i>.csv`` and CPU figures
     - Includes reference work, compilation, and candidate executions; it is
       the *cost of tuning*.
   * - CUDA execution time
     - H100 ``cuda_h100_ratios.csv`` and raw result files
     - Measured time for a selected direct CUDA port on its test input.
   * - Device allocation
     - ``device_allocation_bytes`` and ratios
     - Tracked device-resident allocation, not source-variable count or total
       memory traffic.
   * - Accuracy or validation error
     - Benchmark-specific residual, relative-error, or output-difference fields
     - Evidence tied to the particular output check and test input.

For a mixed implementation and matching FP64 baseline, the stored H100 ratios
use the following interpretation:

.. math::

   \mathrm{time\ ratio} = \frac{T_{\mathrm{mixed}}}{T_{\mathrm{FP64}}},
   \qquad
   \mathrm{speedup} = \frac{T_{\mathrm{FP64}}}{T_{\mathrm{mixed}}},
   \qquad
   \mathrm{memory\ ratio} =
   \frac{B_{\mathrm{mixed}}}{B_{\mathrm{FP64}}}.

A time ratio below one indicates a faster measured case; a memory ratio below
one indicates fewer tracked device-allocation bytes. Neither alone establishes
that the required numerical accuracy was met. See :doc:`benchmark_results`
for the stored H100 cases and their limitations.

Fair comparisons and practical limits
=====================================

* Use the **same input and output validation** for baseline and mixed cases.
  Report input size, convergence settings, and random seed when applicable.
* Hold compiler flags, thread counts, hardware, and software versions fixed.
  For performance, describe warm-up and measured runs and report variability.
* Compare **like with like**: a PROMISE search-time curve answers a different
  question from a standalone CPU kernel run or CUDA-event time.
* Treat emulated custom-format results as evidence about numerical feasibility.
  Native hardware speedups need their own measurements.
* The selected H100 ports preserve the PROMISE-derived computational structure.
  They are not general cuBLAS/cuDNN or Tensor Core rewrites. The separate
  complement experiments mark Tensor-Core-suitable cases explicitly.
* A passing target applies to the exercised input and checked outputs. More
  inputs, independent references, and error metrics strengthen the conclusion.

Next: :doc:`precision_tuning` explains the formats and search; :doc:`example`
contains runnable commands; :doc:`api_references` documents output schemas.
