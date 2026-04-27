=============
API Reference
=============

This section provides reference documentation for the HPC-Mix-Bench tools and configuration files.

Configuration Files
===================

promise.yml
-----------

The ``promise.yml`` file configures compilation settings for C/C++ benchmarks.

**Example structure**:

.. code-block:: yaml

   compiler: g++
   flags:
     - -O3
     - -std=c++11
   sources:
     - main.cpp
     - algorithm.cpp
   output: benchmark_executable
   libraries:
     - m
   include_paths:
     - ./include

**Fields**:

* ``compiler``: Compiler command (e.g., ``g++``, ``gcc``, ``clang++``)
* ``flags``: List of compilation flags
* ``sources``: List of source files to compile
* ``output``: Name of output executable
* ``libraries``: Libraries to link (without ``-l`` prefix)
* ``include_paths``: Additional include directories

fp.json
-------

The ``fp.json`` file defines floating-point format search space for PROMISE.

**Example structure**:

.. code-block:: json

   {
     "formats": [
       {"name": "E5M2", "bits": 8, "exponent": 5, "mantissa": 2},
       {"name": "E4M3", "bits": 8, "exponent": 4, "mantissa": 3},
       {"name": "FP16", "bits": 16, "exponent": 5, "mantissa": 10},
       {"name": "BF16", "bits": 16, "exponent": 8, "mantissa": 7},
       {"name": "FP32", "bits": 32, "exponent": 8, "mantissa": 23},
       {"name": "FP64", "bits": 64, "exponent": 11, "mantissa": 52}
     ],
     "search_order": ["E5M2", "FP16", "FP32", "FP64"]
   }

**Fields**:

* ``formats``: List of available floating-point formats
* ``search_order``: Order in which formats are explored (corresponds to Combinations I-IV)

run_setting_*.py
----------------

Configuration scripts for running precision experiments.

**Example structure**:

.. code-block:: python

   # run_setting_1.py - Combination I (E5M2, FP16, FP32, FP64)
   
   # Precision combination
   PRECISION_COMBO = 1
   
   # Required accuracy (correct significant digits)
   REQUIRED_DIGITS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
   
   # PROMISE settings
   DELTA_DEBUG_MODE = "hierarchical"
   TIMEOUT = 3600  # seconds
   
   # Output settings
   OUTPUT_JSON = f"prec_setting_{PRECISION_COMBO}.json"
   OUTPUT_PLOT = f"precision{PRECISION_COMBO}_runtime.jpg"

**Key variables**:

* ``PRECISION_COMBO``: Which precision combination (1-4)
* ``REQUIRED_DIGITS``: List of accuracy requirements to test
* ``DELTA_DEBUG_MODE``: Delta debugging strategy
* ``TIMEOUT``: Maximum time per delta debugging iteration
* ``OUTPUT_JSON``: Output precision configuration file
* ``OUTPUT_PLOT``: Output visualization file

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

Collects all plots into a central directory.

**Syntax**:

.. code-block:: bash

   bash organize_plots.sh

**Behavior**:

* Searches for all ``.png`` and ``.jpg`` files in benchmark subdirectories
* Copies them to a central ``plots/`` folder
* Organizes by benchmark name and precision combination

json_counts_sum.py
------------------

Generates summary statistics from precision configuration JSON files.

**Syntax**:

.. code-block:: bash

   python json_counts_sum.py

**Output**:

* CSV file with variable counts per precision type
* Summary statistics across all benchmarks
* Aggregated by precision combination

CADNA-PROMISE API
=================

For detailed CADNA-PROMISE documentation, see the `official repository <https://github.com/PEQUAN/hpc-mix-bench/tree/main/cadnaPromise>`_.

Key Functions
-------------

``cadna_init()``
~~~~~~~~~~~~~~~~

Initializes CADNA environment for stochastic arithmetic.

.. code-block:: cpp

   void cadna_init();

``set_precision_format()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sets the precision format for a variable.

.. code-block:: cpp

   void set_precision_format(void* var, const char* format);

**Parameters**:

* ``var``: Pointer to variable
* ``format``: Format string (e.g., "E5M2", "FP16", "FP32", "FP64")

``get_significant_digits()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Returns the number of significant correct digits.

.. code-block:: cpp

   int get_significant_digits(double value, double reference);

**Parameters**:

* ``value``: Computed value
* ``reference``: Reference value (typically FP64)

**Returns**: Number of correct significant digits

Data Structures
===============

Precision Configuration JSON
----------------------------

Output format from PROMISE tool:

.. code-block:: json

   {
     "benchmark": "backprop",
     "combination": 1,
     "required_digits": 5,
     "variables": {
       "var_sigmoid_input": "FP16",
       "var_hidden_error": "FP32",
       "var_weight_delta": "FP64",
       "var_pivot_index": "E5M2"
     },
     "statistics": {
       "E5M2": 12,
       "FP16": 8,
       "FP32": 15,
       "FP64": 45
     },
     "computation_time": 234.5
   }

**Fields**:

* ``benchmark``: Benchmark name
* ``combination``: Precision combination (1-4)
* ``required_digits``: Accuracy requirement
* ``variables``: Variable-to-format mapping
* ``statistics``: Count by format
* ``computation_time``: Total PROMISE time (seconds)

Environment Variables
=====================

PROMISE_DEBUG
-------------

Enable debug output from PROMISE tool.

.. code-block:: bash

   export PROMISE_DEBUG=1

PROMISE_CACHE_DIR
-----------------

Set cache directory for intermediate results.

.. code-block:: bash

   export PROMISE_CACHE_DIR=/path/to/cache

JOBS
----

Default number of parallel workers.

.. code-block:: bash

   export JOBS=8

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