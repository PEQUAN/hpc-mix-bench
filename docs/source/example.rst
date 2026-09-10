=======
Example
=======

This page shows common CPU and H100 workflows.

Run A CPU PROMISE Sweep
=======================

Install and activate PROMISE as described in :doc:`installation`, then
synchronize the shared settings:

.. code-block:: bash

   cd hpc-mix-bench/mp_tests
   bash sync_settings.sh

Run selected benchmarks sequentially:

.. code-block:: bash

   ./run_benchmarks.sh true true false backprop dense_lu hotspot

Run the same sweep with six independent workers:

.. code-block:: bash

   JOBS=6 ./run_benchmarks.sh true true false backprop dense_lu hotspot --parallel

The first three boolean arguments control:

* whether to run PROMISE experiments;
* whether to generate plots;
* whether to run the optional debug scripts.

Generate plots only from existing result files:

.. code-block:: bash

   ./run_benchmarks.sh false true false backprop dense_lu hotspot

Output Files
============

For each benchmark and combination, the scripts generate files such as:

* ``prec_setting_1.json`` through ``prec_setting_4.json``.
* ``runtimes1.csv`` through ``runtimes4.csv``.
* ``precision1_with_runtime.jpg`` through ``precision4_with_runtime.jpg``.
* logs under ``mp_tests/logs/<benchmark>/run_<i>.log``.

The ``prec_setting_<i>.json`` files contain the variables selected for each
precision at each requested significant-digit target.

Run The Direct H100 CUDA Validation
===================================

The H100 CUDA validation code is under ``papers/`` and mirrors the
PROMISE-derived ``digit<i>_<j>`` configurations for Backprop, Hotspot, and
Dense LU.

Submit all three benchmark families from ``hpc-mix-bench/papers`` on Jean Zay:

.. code-block:: bash

   sbatch --account=${IDRPROJ}@h100 \
     --export=ALL,REPO_DIR="$(pwd)/..",BENCHMARKS="backprop dense_lu hotspot",COMBINATIONS="1 2",BACKPROP_SIZE=262144,DENSE_LU_SIZE=5000,HOTSPOT_ROWS=1024,HOTSPOT_COLS=1024,HOTSPOT_ITERS=2,FORCE_REBUILD=1,WARMUP_RUNS=1,MEASURED_RUNS=3,RUN_PLOTS=1 \
     submit_hpc_mix_h100.sh

Submit one benchmark at a time:

.. code-block:: bash

   sbatch --account=${IDRPROJ}@h100 \
     --export=ALL,REPO_DIR="$(pwd)/..",BENCHMARKS=backprop,COMBINATIONS="1 2",BACKPROP_SIZE=262144,FORCE_REBUILD=1,WARMUP_RUNS=1,MEASURED_RUNS=3,RUN_PLOTS=1 \
     submit_hpc_mix_h100.sh

   sbatch --account=${IDRPROJ}@h100 \
     --export=ALL,REPO_DIR="$(pwd)/..",BENCHMARKS=hotspot,COMBINATIONS="1 2",HOTSPOT_ROWS=1024,HOTSPOT_COLS=1024,HOTSPOT_ITERS=2,FORCE_REBUILD=1,WARMUP_RUNS=1,MEASURED_RUNS=3,RUN_PLOTS=1 \
     submit_hpc_mix_h100.sh

   sbatch --account=${IDRPROJ}@h100 \
     --export=ALL,REPO_DIR="$(pwd)/..",BENCHMARKS=dense_lu,COMBINATIONS="1 2",DENSE_LU_SIZE=5000,FORCE_REBUILD=1,WARMUP_RUNS=1,MEASURED_RUNS=3,RUN_PLOTS=1 \
     submit_hpc_mix_h100.sh

Check job status:

.. code-block:: bash

   squeue -u $USER
   sacct -u $USER --name=hpcmix-h100 --format=JobID,JobName,State,Elapsed,ExitCode

The job writes raw CSV files, ratio CSV files, generated inputs, benchmark
outputs, and plots under ``papers/h100_results/<slurm-job-id>/``.

Regenerate H100 Plots
=====================

From ``papers/h100_results``:

.. code-block:: bash

   python3 plot_h100_ratios.py <job-id> --font-size 12
   python3 plot_h100_ratios_combined.py <job-id> --font-size 12
   python3 plot_h100_benchmark_panels.py --font-size 10

Run Complementary Tensor Core Experiments
=========================================

The complement experiments compare the direct CUDA validation against
Tensor-Core-suitable kernels where such a mapping is meaningful.

Submit from ``hpc-mix-bench/papers``:

.. code-block:: bash

   sbatch --account=${IDRPROJ}@h100 \
     --export=ALL,REPO_DIR="$(pwd)/..",COMBINATIONS="1 2",BACKPROP_SIZE=262144,DENSE_LU_SIZE=5000,HOTSPOT_ROWS=1024,HOTSPOT_COLS=1024,HOTSPOT_ITERS=2,WARMUP_RUNS=1,MEASURED_RUNS=5,RUN_PLOTS=1 \
     complement/submit_complement_h100.sh

Manual plotting from ``papers/complement``:

.. code-block:: bash

   python3 plot_complement_h100.py results/<job-id> --font-size 12
