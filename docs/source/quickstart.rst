==========
Quickstart
==========

This guide will walk you through running your first benchmark with HPC-Mix-Bench.

Project Structure
=================

The benchmark suite follows this organization:

.. code-block:: text

   hpc-mix-bench/
   ├── mp_tests/                      # Benchmark test directory
   │   ├── setA/                      # Example benchmark
   │   │   ├── run_setting_1.py       # Configuration for combination I
   │   │   ├── run_setting_2.py       # Configuration for combination II
   │   │   ├── run_setting_3.py       # Configuration for combination III
   │   │   ├── run_setting_4.py       # Configuration for combination IV
   │   │   ├── source.cpp             # C/C++ source code
   │   │   ├── fp.json                # Floating-point format definitions
   │   │   └── promise.yml            # Compilation settings
   │   ├── setB/
   │   └── run_benchmarks.sh          # Main automation script
   ├── run_settings/                  # Global configuration templates
   ├── papers/
   │   └── plots/                     # Benchmark result visualizations
   └── data/                          # Input datasets

Configuration Setup
===================

Step 1: Customize Run Settings
-------------------------------

Navigate to the ``run_settings`` folder and customize the configuration files:

.. code-block:: bash

   cd run_settings
   # Edit run_setting_1.py through run_setting_4.py as needed

Each ``run_setting_*.py`` file corresponds to one precision combination.

Step 2: Prepare Your Test Directory
------------------------------------

Copy the ``mp_tests`` folder to ``my_tests`` and select benchmarks:

.. code-block:: bash

   cp -r mp_tests my_tests
   cd my_tests

Step 3: Synchronize Settings
-----------------------------

Broadcast configuration files to all benchmark subfolders:

.. code-block:: bash

   bash sync_settings.sh --broadcast

Available options:

* ``--delete`` or ``-d``: Delete existing configuration files
* ``--broadcast`` or ``-b``: Copy new configuration files from ``../run_settings/``

Running Benchmarks
==================

Basic Usage
-----------

Make the benchmark script executable:

.. code-block:: bash

   chmod +x run_benchmarks.sh

Run experiments and generate plots:

.. code-block:: bash

   ./run_benchmarks.sh 1 1

Script Syntax
-------------

.. code-block:: bash

   ./run_benchmarks.sh <run_exp> <run_plot> <run_debug> [folders...] [--parallel] [--jobs N]

Arguments:

* ``<run_exp>``: Run experiments (1/true/y = yes, 0/false/n = no)
* ``<run_plot>``: Generate plots (1/true/y = yes, 0/false/n = no)
* ``<run_debug>``: Run debug scripts (1/true/y = yes, 0/false/n = no)
* ``[folders...]``: Specific benchmark folders (default: all)
* ``--parallel``: Enable parallel execution
* ``--jobs N``: Number of parallel workers

Common Usage Examples
---------------------

Run only experiments:

.. code-block:: bash

   ./run_benchmarks.sh 1 0

Run only plots (using saved data):

.. code-block:: bash

   ./run_benchmarks.sh 0 1

Run experiments and plots with debug:

.. code-block:: bash

   ./run_benchmarks.sh 1 1 1

Run specific benchmarks:

.. code-block:: bash

   ./run_benchmarks.sh 1 1 backprop hotspot

Parallel execution:

.. code-block:: bash

   ./run_benchmarks.sh 1 1 --parallel --jobs 4

Output Files
============

After running benchmarks, each directory will contain:

* ``prec_setting_1.json`` through ``prec_setting_4.json``: PROMISE-generated precision configurations
* ``*.png``: Visualization plots
* ``plots/``: Organized plot directory

Organizing Plots
----------------

Collect all plots into a central location:

.. code-block:: bash

   cd mp_tests
   bash organize_plots.sh

Generate Summary Statistics
============================

After running all experiments, generate a summary of floating-point type usage:

.. code-block:: bash

   python json_counts_sum.py

This produces statistics showing the distribution of precision types across all benchmarks.

Next Steps
==========

* Learn about benchmark results in :doc:`benchmark_results`
* Explore the API reference: :doc:`api_reference`
* View all benchmarks on GitHub: `mp_tests/ <https://github.com/PEQUAN/hpc-mix-bench/tree/main/mp_tests>`_