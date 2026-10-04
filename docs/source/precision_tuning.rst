===========================
Precision Tuning Explained
===========================

Precision is a resource: it affects the range and spacing of representable
numbers, storage requirements, conversion cost, and the arithmetic available
on a target machine. HPC-MIX Bench helps answer a narrower, testable question:
**for this program, input, and output-accuracy target, which annotated
variables can use each candidate format?**

The goal is an assignment across variables, not a single data type for the
whole application. A long-lived state variable may need FP64 while a temporary
value in the same kernel tolerates FP16 or a custom eight-bit format.

The format trade-off
====================

A binary format is described here by ``(e, t)``: exponent bits ``e`` and
explicit trailing-significand bits ``t``. More exponent bits generally extend
dynamic range; more significand bits reduce rounding between nearby normal
numbers. These are distinct failure modes, so a format with more range is not
automatically more accurate for every computation.

.. list-table:: Formats used in the standard search spaces
   :widths: 18 16 22 44
   :header-rows: 1

   * - Format
     - ``(e, t)``
     - PROMISE alias
     - Useful comparison
   * - E4M3
     - ``(4, 3)``
     - ``c``
     - More trailing-significand bits than E5M2, with less exponent range.
   * - E5M2
     - ``(5, 2)``
     - ``w``
     - Wider exponent range than E4M3, with coarser normal-number spacing.
   * - FP16-like custom format
     - ``(5, 10)``
     - ``p``
     - Tests whether a ten-bit trailing significand is enough.
   * - BF16-like format
     - ``(8, 7)``
     - ``b``
     - Preserves an FP32-like exponent width with fewer significand bits.
   * - FP32
     - ``(8, 23)``
     - ``s``
     - Common intermediate and comparison precision.
   * - FP64
     - ``(11, 52)``
     - ``d``
     - High-precision candidate and reference-format baseline.

The built-in ``h`` alias uses ``half_float::half``. The custom ``p`` alias is
defined in ``run_settings/fp.json`` and uses a FloatX-style ``(5, 10)`` type.
The standard four combinations use ``p``. See :doc:`configuration` for the
exact aliases and files.

.. important::

   Custom FloatX types are useful for **precision sensitivity experiments**.
   Their CPU execution time is not a prediction of native FP8, BF16, or FP16
   speed on a GPU. Hardware performance also depends on vectorization, memory
   layout, conversions, and whether a kernel actually uses specialized units.

How PROMISE searches
====================

#. Annotate tunable declarations with ``__PROMISE__`` and identify outputs to
   check with the PROMISE checking macros. The checked outputs define what an
   accuracy claim covers.
#. Choose an ordered search space, such as ``wpsd``, and a required number of
   correct significant digits through ``--nbDigits``.
#. PROMISE builds a reference using CADNA's Discrete Stochastic Arithmetic
   (DSA), then tests candidate type assignments through source transformation,
   compilation, and execution.
#. Delta debugging narrows the set of variables that can be reduced while the
   checked result still meets the requested target.
#. Inspect the returned per-type variable lists and the generated source. The
   same target can lead to different assignments for different inputs or
   benchmark configurations.

A compact invocation from a configured benchmark directory is:

.. code-block:: bash

   promise --precs=wpsd --nbDigits=5 --conf=promise.yml --fp=fp.json

The repository's ``run_setting_*.py`` scripts sweep targets from one to ten
digits and write ``prec_setting_<i>.json``, ``runtimes<i>.csv``, and figures.
The runtime recorded there is the *complete search wall time*, including
reference work, transformations, compilation, and candidate runs. See
:doc:`benchmark_results` for separate H100 application timing.

What does “five correct digits” mean?
======================================

``--nbDigits=5`` is a requested accuracy threshold for the outputs that a
benchmark checks. It is not a promise that every intermediate value has five
correct digits. It is also not a proof for every possible input: the observed
assignment is tied to the supplied dataset, control flow, build, and checking
logic.

When reading a result, ask:

* Which output variable or array was checked?
* Was the reference computation numerically meaningful for this input?
* Were non-finite values, convergence failures, and invalid outputs detected?
* Does the program use randomness or parallel reductions, and were those
  conditions controlled?
* Does the generated program satisfy an independent application-level error
  or residual check on held-out inputs?

Four standard search spaces
===========================

.. list-table:: Standard combinations
   :widths: 18 17 31 34
   :header-rows: 1

   * - Combination
     - ``--precs``
     - Candidate formats
     - Main comparison
   * - I
     - ``wpsd``
     - E5M2 → FP16-like → FP32 → FP64
     - E5M2 with a narrow intermediate range.
   * - II
     - ``wbsd``
     - E5M2 → BF16-like → FP32 → FP64
     - Effect of BF16-like range at intermediate precision.
   * - III
     - ``cpsd``
     - E4M3 → FP16-like → FP32 → FP64
     - Effect of a different eight-bit range/spacing balance.
   * - IV
     - ``cbsd``
     - E4M3 → BF16-like → FP32 → FP64
     - E4M3 paired with BF16-like intermediate precision.

The letters describe *candidate sets*. A final assignment can contain several
formats at once. Compare the same benchmark, input, and digit target across
combinations to isolate the effect of changing the candidate set.

Diagnosing sensitivity
=======================

Precision counts are an entry point, not a complete numerical explanation.
Patterns worth investigating include:

.. list-table:: Questions suggested by a precision assignment
   :widths: 35 65
   :header-rows: 1

   * - Observation
     - Follow-up check
   * - Low precision fails only for large or tiny inputs
     - Test overflow, underflow, scaling, and the available exponent range.
   * - More digits force accumulators upward
     - Inspect long sums, reductions, iteration counts, and residuals.
   * - Factorizations need high precision in a few variables
     - Inspect pivoting, near-singular inputs, cancellation, and conditioning.
   * - Counts change sharply at one target
     - Compare generated source and checked outputs on both sides of the
       threshold; a search path can also change abruptly.
   * - Lower precision reduces bytes but not time
     - Profile conversions, memory traffic, launch overhead, and use of
       native low-precision arithmetic.

One-bit format exploration
==========================

The ``1-bit-exps`` experiments vary one format dimension at a time. With the
exponent width fixed, varying ``t`` probes rounding sensitivity; with ``t``
fixed, varying ``e`` probes range sensitivity. Each point compares one custom
type with FP64. This makes a useful follow-up when two eight-bit formats behave
differently on the same workload.

The repository provides four plot variants. From ``1-bit-exps/``, for example:

.. code-block:: bash

   python3 onebit_precision_analysis1.py --benchmark hotspot --benchmark dense_lu
   python3 onebit_precision_analysis1.py --run --benchmark hotspot --digits 1-10

The first command reads existing ``onebit_precision_results.csv`` files; the
second reruns PROMISE sweeps and can take substantially longer. See
:doc:`methodology` before comparing outputs across machines.

Next: :doc:`example` for commands, :doc:`applications` for workload selection,
and :doc:`methodology` for a repeatable experimental protocol.
