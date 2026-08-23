=============
API Reference
=============

This section provides reference documentation for the HPC-Mix-Bench tools and configuration files.

Configuration Files
===================

promise.yml
-----------

The ``promise.yml`` file configures the compile/run commands PROMISE uses to build and execute each C/C++ benchmark.

**Example** (from ``mp_tests/dense_lu/promise.yml``):

.. code-block:: yaml

   compile:
   - g++ -O3 lu.cpp -frounding-math -m64 -o lu.out -lcadnaC -L$CADNA_PATH/lib -I$CADNA_PATH/include
   run: lu.out
   files: lu.cpp
   log: lu.log
   output: debug/

**Fields**:

* ``compile``: a YAML list of shell commands used to compile the source code. Compilation must link CADNA (``-lcadnaC -L$CADNA_PATH/lib -I$CADNA_PATH/include``) and use ``g++`` (not ``gcc``).
* ``run``: the command/executable PROMISE runs to produce results.
* ``files``: source file(s) PROMISE instruments and searches for candidate variables (defaults to all ``.cc``/``.cpp`` files when omitted).
* ``log``: optional log file capturing program output.
* ``output``: directory where PROMISE writes the transformed/instrumented source code for each precision combination (commonly ``debug/``).

fp.json
-------

The ``fp.json`` file maps single-letter precision codes to ``[exponent_bits, mantissa_bits]`` pairs. PROMISE's ``--precs`` option selects which letters to search over.

**Example** (``run_settings/fp.json``, broadcast into benchmark folders by ``sync_settings.sh``):

.. code-block:: json

   {
     "c": [4, 3],
     "w": [5, 2],
     "b": [8, 7],
     "p": [5, 10],
     "h": [5, 10],
     "s": [8, 23],
     "d": [11, 52],
     "q": [15, 112],
     "o": [19, 236]
   }

**Fields**:

* Each key is a single letter used as a precision code in ``--precs`` (e.g. ``sd`` searches single/double).
* Each value is ``[exponent, mantissa]`` bit widths for a `FloatX <https://github.com/oprecomp/FloatX>`_-style custom format.
* Built-in codes ``h`` (half/FP16, 5/10), ``s`` (single/FP32, 8/23), and ``d`` (double/FP64, 11/52) map to the standard IEEE formats used throughout this repository's benchmark results; ``c``/``w`` (E4M3/E5M2, 8-bit) and ``b`` (BF16, 8/7) extend the search to the reduced-precision formats used on modern accelerators.

run_setting_*.py
----------------

Configuration scripts for running precision experiments. Each of the four scripts (``run_setting_1.py`` through ``run_setting_4.py``, shared via ``run_settings/`` and broadcast with ``sync_settings.sh``) drives one precision combination (I-IV) by calling the bundled ``cadnaPromise`` package's ``run_promise`` entry point across a range of required significant digits, then plots the resulting precision counts.

**Behavior**:

* Iterates over required accuracy values, typically 1 to 10 correct significant digits.
* For each accuracy value, invokes PROMISE via ``cadnaPromise.run.runPromise(['--precs=<letters>', '--nbDigits=<digit>', ...])`` using the ``fp.json`` precision letters for that combination and the benchmark's ``promise.yml``.
* Writes the resulting variable-to-format assignment to ``prec_setting_<i>.json``.
* Renders a stacked bar chart of precision usage versus required digits to ``precision<i>_with_runtime.jpg``, overlaying total PROMISE computation time.

**Example** (excerpt from ``run_settings/run_setting_1.py``, Combination I: E5M2/FP16/FP32/FP64):

.. code-block:: python

   from cadnaPromise.run import runPromise
   import time

   def run_experiments(method, digits):
       """Run PROMISE once per required-digit value and time each run."""
       prec_setting, runtimes = [], []
       for digit in digits:
           testargs = [f'--precs={method}', f'--nbDigits={digit}',
                       '--conf=promise.yml', '--fp=fp.json']
           start_time = time.time()
           result = runPromise(testargs)
           runtimes.append(time.time() - start_time)
           prec_setting.append(result)
       return prec_setting, runtimes

   if __name__ == "__main__":
       method = 'wpsd'                          # precision letters searched (fp.json codes)
       digits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # required significant digits

**Key variables/functions**:

* ``method``: precision letters passed to PROMISE's ``--precs`` (drawn from ``fp.json``, e.g. ``wpsd`` = E5M2/FP16/FP32/FP64 for Combination I).
* ``digits``: list of required significant digits to sweep (default ``1`` through ``10``).
* ``run_experiments(method, digits)``: runs PROMISE once per digit value via ``runPromise`` and collects one precision-assignment dict per run.
* ``save_prec_setting(prec_setting, filename)`` / ``load_prec_setting(filename)``: persist/reload results to/from ``prec_setting_<i>.json``.
* ``save_runtimes_to_csv`` / ``load_runtimes``: persist/reload per-digit PROMISE runtimes to/from ``runtimes<i>.csv``.
* Plotting code saves the final figure to ``precision<i>_with_runtime.jpg``.

Scripts
=======

run_benchmarks.sh
-----------------

Main automation script for running benchmarks.

**Syntax**:

.. code-block:: bash

   ./run_benchmarks.sh <run_exp> <run_plot> <run_debug> [folders...] [--parallel] [--jobs N]

**Arguments**:

* ``run_exp``: Run experiments (1/true/y or 0/false/n)
* ``run_plot``: Generate plots (1/true/y or 0/false/n)
* ``run_debug``: Run debug scripts (1/true/y or 0/false/n)
* ``folders``: Optional list of specific benchmark folders
* ``--parallel``: Enable parallel execution
* ``--jobs N``: Number of parallel workers

sync_settings.sh
----------------

Synchronizes configuration files across benchmark folders.

**Syntax**:

.. code-block:: bash

   bash sync_settings.sh [options]

**Options**:

* ``--delete`` / ``-d``: Delete existing configuration files
* ``--broadcast`` / ``-b``: Copy files from ``../run_settings/``

organize_plots.sh
-----------------

Collects generated plots into a central directory. Located in ``papers/``.

**Syntax**:

.. code-block:: bash

   cd papers
   bash organize_plots.sh [folder1 folder2 ...]

**Behavior**:

* If no folders are given, scans every immediate subdirectory of the current directory (2 levels deep) for ``precision<i>_with_runtime.jpg`` files.
* If folders are given, only those directories (paths relative to the current directory, e.g. ``../mp_tests/backprop``) are processed.
* Renames and moves matching files to ``precision<i>_<folder_name>_runtime.jpg`` inside a ``plots/`` directory created next to ``organize_plots.sh``.

calculate_stats.py
------------------

Generates summary statistics from ``prec_setting_<i>.json`` files. Located in both ``mp_tests/`` and ``papers/``.

**Syntax**:

.. code-block:: bash

   python3 calculate_stats.py folder1 folder2 ... folderk

**Output** (written to the current directory):

* ``fp_counts_summary.csv``: variable counts per precision type (FP64, FP32, FP16, BF16, E4M3, E5M2), per benchmark folder and precision combination.
* ``fp_ratio_averages.csv``: average share of each precision type, averaged across all rows.

CADNA-PROMISE
=============

``cadnaPromise`` (bundled under `cadnaPromise/ <https://github.com/PEQUAN/hpc-mix-bench/tree/main/cadnaPromise>`_) provides the ``promise`` command-line tool used by every ``run_setting_*.py`` script. See ``cadnaPromise/README.rst`` and ``cadnaPromise/EXAMPLE.rst`` for the full walkthrough.

Marking Code for Instrumentation
---------------------------------

Mark a variable as eligible for reduced precision with the ``__PROMISE__`` type qualifier, and instrument variables/arrays for accuracy checking with:

.. code-block:: cpp

   PROMISE_CHECK_VAR(variable);
   PROMISE_CHECK_ARRAY(array, n_elements);

Command-Line Interface
-----------------------

.. code-block:: bash

   promise --help
   promise --version
   promise --precs=<letters> [options]

**Key options** (see ``promise --help`` for the complete list):

* ``--precs=<strs>``: precision letters to search, drawn from ``fp.json`` (default: ``sd``)
* ``--conf CONF_FILE``: configuration file (default: ``promise.yml``)
* ``--fp FPT_FILE``: floating-point format file (default: ``fp.json``)
* ``--nbDigits DIGITS``: required number of correct significant digits
* ``--output OUTPUT``: output directory for transformed code
* ``--verbosity VERBOSITY``: verbosity level (0-4, default: 1)
* ``--log LOGFILE``: optional log file
* ``--debug``: keep intermediate files and show the execution trace
* ``--noCadna``: disable CADNA and use a double-precision reference instead
* ``--alias ALIAS``: allow command aliases (e.g. ``"g++=g++-14"``)
* ``--CC`` / ``--CXX``: C/C++ compiler override (default: ``g++``)
* ``--plot``: enable plotting of results (default: enabled)

Python Entry Point
-------------------

``run_setting_*.py`` scripts call PROMISE programmatically via:

.. code-block:: python

   from cadnaPromise.run import runPromise

   runPromise(['--precs=sd', '--nbDigits=5'])

Data Structures
===============

Precision Configuration JSON
----------------------------

Each ``prec_setting_<i>.json`` file written by ``run_setting_<i>.py`` is a JSON list with one entry per required-digit level (by default, indices 0-9 correspond to 1-10 correct significant digits). Each entry maps a C++ type name to the list of variable/line indices PROMISE assigned to that type.

**Example** (first entry of ``papers/hotspot/prec_setting_1.json``, for 1 correct digit):

.. code-block:: json

   {
     "double": [0, 1, 2, 9, 10, 11, 24, 25, 26, 27, 28, 29, 30],
     "flx::floatx<5, 10>": [14],
     "flx::floatx<5, 2>": [3, 4, 5, 6, 7, 8, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23]
   }

The full file is a JSON array containing one such object per required-digit value.

**Type name mapping** (used by ``calculate_stats.py``):

* ``double`` -> FP64
* ``float`` -> FP32
* ``flx::floatx<5, 10>`` -> FP16
* ``flx::floatx<8, 7>`` -> BF16
* ``flx::floatx<4, 3>`` -> E4M3
* ``flx::floatx<5, 2>`` -> E5M2

Environment Variables
=====================

CADNA_PATH
----------

Path to the CADNA installation used to compile and link instrumented code (``-lcadnaC -L$CADNA_PATH/lib -I$CADNA_PATH/include``). Set automatically by ``activate-promise``, or manually when using a standalone CADNA install:

.. code-block:: bash

   export CADNA_PATH=/path/to/cadna

JOBS
----

Default number of parallel workers for ``run_benchmarks.sh --parallel`` (falls back to ``nproc``, then ``4``, if unset).

.. code-block:: bash

   export JOBS=8

OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS
--------------------------------------------------------

``run_benchmarks.sh`` sets these to ``1`` by default (unless already set in the environment) so that per-benchmark thread parallelism does not interfere with benchmark-level ``--parallel``/``--jobs`` scheduling.

.. code-block:: bash

   export OMP_NUM_THREADS=4

File Locations
==============

Directory Structure
-------------------

.. code-block:: text

   hpc-mix-bench/
   ├── cadnaPromise/          # CADNA-PROMISE library
   ├── data/                  # Input datasets
   ├── mp_tests/              # Benchmark tests
   ├── papers/
   │   └── plots/             # Result visualizations
   ├── run_settings/          # Global configurations
   └── src/                   # Source code

Standard File Naming
--------------------

* Precision configurations: ``prec_setting_{1-4}.json``
* Plots: ``precision{1-4}_{benchmark}_runtime.jpg``
* Debug scripts: ``run_debug_{1-4}.sh``
* Setting scripts: ``run_setting_{1-4}.py``

Next Steps
==========

* Learn how to use the tools: :doc:`quickstart`
* Understand the results: :doc:`benchmark_results`
* Contribute new benchmarks: :doc:`contributing`