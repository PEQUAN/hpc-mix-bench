============
Installation
============

This page describes the software needed to run the CPU PROMISE benchmark suite
and the optional H100 CUDA validation scripts.

Prerequisites
=============

For CPU precision-search experiments:

* Python 3.9 or newer.
* ``g++`` with C++11 support.
* CADNA, activated through the bundled ``cadnaPromise`` package or through a
  local CADNA installation.
* Python packages used by the run and plotting scripts: ``numpy``,
  ``matplotlib``, ``colorama``, ``colorlog``, ``tqdm``, ``regex``, ``pyyaml``,
  ``packaging``, and ``docopt-ng`` or ``docopt``.
* GNU Parallel is optional but recommended for ``run_benchmarks.sh
  --parallel``.

For H100 validation experiments:

* NVIDIA CUDA with ``nvcc``.
* An H100-capable CUDA architecture target, normally ``sm_90``.
* On Jean Zay, an H100 allocation such as ``${IDRPROJ}@h100``.

Clone The Repository
====================

.. code-block:: bash

   git clone https://github.com/PEQUAN/hpc-mix-bench.git
   cd hpc-mix-bench

Install CADNA/PROMISE
=====================

The repository bundles the ``cadnaPromise`` Python package. A local editable
install is convenient when running from a cluster account:

.. code-block:: bash

   cd cadnaPromise
   python3 -m pip install --user -e .
   export PATH="$(python3 -m site --user-base)/bin:$PATH"
   activate-promise

Verify that the command-line tools are visible:

.. code-block:: bash

   which promise
   which activate-promise
   python3 -c "import cadnaPromise; print(cadnaPromise.__file__)"

Install Python Dependencies
===========================

On systems with internet access:

.. code-block:: bash

   python3 -m pip install --user numpy matplotlib colorama colorlog tqdm regex pyyaml packaging docopt-ng

On clusters where compute nodes do not have network access, install these
packages on a login node before submitting jobs. The batch scripts add the user
base ``bin`` directory to ``PATH`` and the user ``site-packages`` directory to
``PYTHONPATH`` when needed.

Docker
======

Docker provides a reproducible CPU environment for PROMISE and plotting.

On Linux x86_64 and Windows on Intel/AMD:

.. code-block:: bash

   docker build -t hpc-mix-cadna .
   docker run -it --rm hpc-mix-cadna

On macOS Apple Silicon and Windows on ARM:

.. code-block:: bash

   docker buildx build --platform linux/amd64 -t hpc-mix-cadna .
   docker run --platform linux/amd64 -it --rm hpc-mix-cadna

The repository also provides a Docker Compose service:

.. code-block:: bash

   docker compose run --rm promise-env

ReadTheDocs
===========

The online documentation is built with Sphinx. To reproduce the build locally:

.. code-block:: bash

   python3 -m pip install -r docs/requirements.txt
   python3 -m sphinx -b html docs/source docs/_build/html

ReadTheDocs uses the repository-level ``.readthedocs.yaml`` file and installs
``docs/requirements.txt`` before building ``docs/source``.
