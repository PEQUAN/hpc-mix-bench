========
Licenses
========

Repository License
==================

HPC-Mix-Bench is distributed under the MIT License. The full license text is
available in the repository-level ``LICENSE`` file.

Attribution
===========

When using the benchmark suite, please cite the PROMISE/CADNA work and the
repository or paper artifact associated with your experiment. The repository
contains software and examples built around:

* PROMISE and CADNA for validated precision tuning.
* FloatX for emulating user-defined floating-point formats in C++.
* benchmark sources adapted from numerical, machine-learning, and
  Rodinia-style workloads.
* NVIDIA CUDA examples and H100 validation scripts for selected generated
  mixed-precision programs.

Third-Party Components
======================

Third-party projects retain their own licenses. Users should consult the
corresponding upstream projects and local source directories before
redistributing derived artifacts:

* CADNA: https://cadna.lip6.fr/
* FloatX: https://github.com/oprecomp/FloatX
* half.hpp: https://half.sourceforge.net/
* Rodinia benchmark suite: consult the upstream Rodinia distribution.
* SuiteSparse Matrix Collection datasets: consult SuiteSparse Matrix Collection
  terms for each dataset.
* NVIDIA CUDA, cuBLAS, WMMA, and H100-specific headers/libraries: consult the
  NVIDIA CUDA Toolkit documentation and license.

Documentation License
=====================

Unless a file states otherwise, the documentation in ``docs/`` follows the same
MIT License as the repository.
