.. HPC-Mix-Bench documentation master file

===============================================
HPC-Mix-Bench: Mixed-Precision Benchmark Suite
===============================================


Welcome to **HPC-Mix-Bench**, a comprehensive benchmark suite for evaluating and optimizing mixed-precision configurations in high-performance computing applications using the PROMISE tool.

What is HPC-Mix-Bench?
=======================

HPC-Mix-Bench is an automated framework that helps you:

🎯 **Discover optimal precision configurations** for your numerical algorithms

🔬 **Analyze numerical behavior** across different floating-point formats (E5M2, E4M3, FP16, BF16, FP32, FP64)

⚡ **Reduce computation cost** while maintaining required accuracy

📊 **Visualize precision usage** patterns across accuracy requirements

The tool uses **PROMISE** (Precision Optimization via Mixed-precision Instruction Selection Engine) with delta debugging to automatically determine the minimum precision needed for each variable in your code.

Why Use Mixed Precision?
=========================

Modern hardware accelerators (GPUs, TPUs) support various precision formats:

.. list-table:: Floating-Point Format Comparison
   :widths: 20 15 15 20 30
   :header-rows: 1

   * - Format
     - Bits
     - Exponent
     - Mantissa
     - Best Use Case
   * - **E5M2**
     - 8
     - 5
     - 2
     - Wide dynamic range operations
   * - **E4M3**
     - 8
     - 4
     - 3
     - Higher precision, narrower range
   * - **FP16**
     - 16
     - 5
     - 10
     - Deep learning inference
   * - **BF16**
     - 16
     - 8
     - 7
     - Deep learning training
   * - **FP32**
     - 32
     - 8
     - 23
     - General purpose computing
   * - **FP64**
     - 64
     - 11
     - 52
     - High-precision scientific computing

Using lower precision where possible can:

* **Speed up computation** by 2-8x
* **Reduce memory bandwidth** requirements
* **Lower energy consumption** significantly
* **Maintain numerical accuracy** with proper precision allocation

Quick Example
=============

Let's say you have a simple C++ program:

.. code-block:: cpp

   // my_algorithm.cpp
   #include <iostream>
   #include <cmath>
   
   int main() {
       double x = 1.0;
       for (int i = 0; i < 1000; i++) {
           x = x + 0.001 * sin(x);
       }
       std::cout << x << std::endl;
       return 0;
   }

**Step 1**: Install CADNA-PROMISE

.. code-block:: bash

   pip install cadnaPromise

**Step 2**: Create a benchmark directory

.. code-block:: bash

   cd mp_tests
   mkdir my_algorithm
   cd my_algorithm

**Step 3**: Add your code and configuration files (``promise.yml``, ``fp.json``)

**Step 4**: Run the benchmark

.. code-block:: bash

   cd ..
   ./run_benchmarks.sh 1 1 my_algorithm

**Result**: You'll get:

* ``prec_setting_1.json`` - Precision configuration for each variable
* Visualization plots showing precision usage vs. accuracy
* Performance analysis

Available Benchmarks
====================

HPC-Mix-Bench includes **40+ ready-to-use benchmarks** across multiple domains:

Scientific Computing
--------------------

Linear Algebra Solvers
~~~~~~~~~~~~~~~~~~~~~~~

* **dense_lu** - Dense LU factorization with partial pivoting
* **sparse_lu** - Sparse LU factorization with RCM ordering
* **cg** - Conjugate Gradient iterative solver
* **bicgstab** - BiConjugate Gradient Stabilized method
* **gmres_tol1/tol2** - GMRES with different convergence tolerances
* **jacobi** - Jacobi iterative solver
* **gauss_seidel** - Gauss-Seidel iterative method
* **sor** - Successive Over-Relaxation
* **ir3_tol1/tol2** - Iterative Refinement (3 iterations)
* **multigrid** - Multigrid V-cycle solver
* **qr** - QR factorization
* **lud** - LU decomposition

Numerical Integration & ODEs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **rk4** - Runge-Kutta 4th order ODE solver
* **simpson** - Simpson's rule for numerical integration
* **trapezoidal** - Trapezoidal rule integration
* **cubic_spline** - Cubic spline interpolation
* **lagrange** - Lagrange polynomial interpolation
* **nystrom** - Nyström method for integral equations

Physical Simulations (Rodinia Suite)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **backprop** - Neural network backpropagation
* **hotspot** - 2D thermal simulation
* **hotspot3D** - 3D thermal simulation
* **particle_filter** - Stochastic particle filtering
* **srad_v2** - Speckle Reducing Anisotropic Diffusion
* **cfd** - Computational Fluid Dynamics solver
* **streamcluster** - Stream clustering algorithm

Machine Learning Algorithms
----------------------------

Supervised Learning
~~~~~~~~~~~~~~~~~~~

* **mlp** - Multi-Layer Perceptron (classification)
* **mlp_reg** - Multi-Layer Perceptron (regression)
* **knn** - K-Nearest Neighbors
* **svm** - Support Vector Machine
* **decisiontree** - Decision Tree classifier
* **randomforest** - Random Forest ensemble
* **adaboost** - AdaBoost ensemble learning
* **gassnb** - Gaussian Naive Bayes

Unsupervised Learning
~~~~~~~~~~~~~~~~~~~~~

* **kmeans** - K-Means clustering
* **dbscan** - DBSCAN density-based clustering
* **pca** - Principal Component Analysis
* **rsvd** - Randomized Singular Value Decomposition

All benchmark source code is available at: `mp_tests/ <https://github.com/PEQUAN/hpc-mix-bench/tree/main/mp_tests>`_

Key Features
============

🔧 **Easy Integration**

* Simple YAML-based configuration
* Support for C/C++ codes
* Docker support for reproducible environments

🎨 **Automatic Visualization**

* Precision usage charts
* Computation time analysis
* Variable-level precision breakdown

⚡ **Parallel Execution**

* Run multiple benchmarks simultaneously
* Configurable worker count
* Efficient resource utilization

📈 **Comprehensive Analysis**

* Four precision combinations tested automatically
* 1-10 significant digit requirements
* Statistical summaries

Precision Combinations
======================

HPC-Mix-Bench automatically evaluates four precision combinations:

.. list-table:: Precision Combinations
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

Each combination represents a different search space for the PROMISE tool to explore.

Workflow
========

.. image:: ../../workfloat.png
   :width: 100%
   :alt: HPC-Mix-Bench Workflow

The typical workflow:

1. **Prepare**: Add your C/C++ code and configuration files
2. **Configure**: Customize precision combinations and accuracy requirements
3. **Run**: Execute benchmarks with automation scripts
4. **Analyze**: Review generated precision configurations and plots
5. **Optimize**: Apply discovered precision settings to production code

Getting Started
===============

.. code-block:: bash

   # Install dependencies
   pip install cadnaPromise
   
   # Clone the repository
   git clone https://github.com/PEQUAN/hpc-mix-bench.git
   cd hpc-mix-bench
   
   # Run example benchmarks
   cd mp_tests
   ./run_benchmarks.sh 1 1 backprop hotspot
   
   # Collect the generated plots into a central plots/ directory
   # (organize_plots.sh lives in papers/, so pass paths to the benchmark folders)
   cd ../papers
   bash organize_plots.sh ../mp_tests/backprop ../mp_tests/hotspot

See the :doc:`installation` guide for detailed setup instructions and the :doc:`quickstart` for your first benchmark run.

Use Cases
=========

HPC-Mix-Bench is ideal for:

🔬 **Researchers** evaluating numerical stability of algorithms

🏢 **HPC Centers** optimizing application performance

🤖 **ML Engineers** tuning precision for neural network inference

⚙️ **Software Developers** reducing computation cost in production

📚 **Educators** teaching numerical methods and floating-point arithmetic

Documentation Structure
========================

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Benchmark Results

   benchmark_results

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api_reference
   contributing

Performance Highlights
======================

From our benchmark results:

* **Pivoting operations**: Can safely use 8-bit precision (E5M2/E4M3) with **no accuracy loss**
* **Iterative solvers**: Achieve **50-70%** variables in low/mid precision for moderate accuracy
* **Neural networks**: Maintain **6 correct digits** with aggressive mixed precision
* **Thermal simulations**: Balance coefficient precision and timestep stability automatically

Community & Support
===================

* **GitHub Repository**: https://github.com/PEQUAN/hpc-mix-bench
* **Issue Tracker**: https://github.com/PEQUAN/hpc-mix-bench/issues
* **License**: MIT License for the benchmark suite; the bundled ``cadnaPromise`` package is distributed separately under the GNU LGPLv3 (see ``cadnaPromise/LICENSE``)

References
==========

HPC-Mix-Bench is built to exercise **PROMISE**, a floating-point precision auto-tuning tool that relies on the **CADNA** library (Discrete Stochastic Arithmetic) and a Delta Debugging search strategy. Key references:

* S. Graillat, F. Jézéquel, R. Picot, F. Févotte, B. Lathuilière. *Auto-tuning for floating-point precision with Discrete Stochastic Arithmetic*. Journal of Computational Science, 36, 101017, 2019. `HAL:hal-01331917 <https://hal.archives-ouvertes.fr/hal-01331917>`__
* F. Jézéquel and J.-M. Chesneaux. *CADNA: a library for estimating round-off error propagation*. Computer Physics Communications, 178(12):933-955, 2008.
* P. Eberhart, J. Brajard, P. Fortin, F. Jézéquel. *High Performance Numerical Validation using Stochastic Arithmetic*. Reliable Computing, 21, 35-52, 2015.
* A. Zeller. *Why Programs Fail*, 2nd ed., Morgan Kaufmann, 2009 (the Delta Debugging algorithm used by PROMISE's search).

The full PROMISE/CADNA bibliography is available in ``cadnaPromise/docs/source/index.rst``.

Citation
========

If you use HPC-Mix-Bench in your research, please cite:

.. code-block:: bibtex

   @software{hpc_mix_bench,
     title = {HPC-Mix-Bench: Benchmarks for Mixed-Precision Emulations},
     author = {PEQUAN Team},
     year = {2026},
     url = {https://github.com/PEQUAN/hpc-mix-bench}
   }

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`