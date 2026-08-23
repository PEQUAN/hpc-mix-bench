============
Installation
============

This guide will help you set up HPC-Mix-Bench on your system.

Prerequisites
=============

* Python 3.10 or higher
* GCC/G++ compiler with C++11 support (PROMISE/CADNA compilation requires ``g++``, not ``gcc``)
* ``build-essential``, ``cmake`` (used by the Docker image; recommended for local builds too)
* Docker (optional, for containerized deployment)

Installing CADNA-PROMISE
=========================

The benchmark tool requires ``cadnaPromise`` to be installed. Install it, along with the plotting/analysis dependencies used by the run-setting scripts, using pip:

.. code-block:: bash

   python3 -m pip install cadnaPromise matplotlib numpy

Then activate CADNA support (enables CADNA and arbitrary-precision customization):

.. code-block:: bash

   activate-promise

To deactivate later:

.. code-block:: bash

   deactivate-promise

Alternatively, install ``CADNA`` (and, for custom formats, ``FloatX``) manually and point PROMISE at it:

.. code-block:: bash

   export CADNA_PATH=/path/to/cadna

Verify the CLI is available:

.. code-block:: bash

   promise --version

For detailed information about CADNA-PROMISE, visit the `cadnaPromise directory <https://github.com/PEQUAN/hpc-mix-bench/tree/main/cadnaPromise>`_.

Clone the Repository
====================

Clone the HPC-Mix-Bench repository:

.. code-block:: bash

   git clone https://github.com/PEQUAN/hpc-mix-bench.git
   cd hpc-mix-bench

Docker Installation
===================

Using Docker provides a consistent environment across different platforms.

macOS (Apple Silicon) and Windows on ARM
-----------------------------------------

Build with platform specification to avoid compilation issues:

.. code-block:: bash

   docker buildx build --platform linux/amd64 -t hpc-mix-cadna .

Run the container:

.. code-block:: bash

   docker run --platform linux/amd64 -it --rm hpc-mix-cadna

Windows (Intel/AMD) and Linux (x86_64)
---------------------------------------

Build normally without platform flags:

.. code-block:: bash

   docker build -t hpc-mix-cadna .

Run the container:

.. code-block:: bash

   docker run -it --rm hpc-mix-cadna

Activate CADNA-PROMISE
-----------------------

After entering the Docker container, activate CADNA-PROMISE (this is done automatically by the container entrypoint, but can be re-run if needed):

.. code-block:: bash

   activate-promise

Docker Compose
--------------

A ``docker-compose.yml`` service is also provided:

.. code-block:: bash

   docker compose run --rm promise-env

Verification
============

Verify your installation by running a simple test:

.. code-block:: bash

   cd mp_tests
   ls

You should see various benchmark directories (e.g., ``backprop``, ``hotspot``, ``dense_lu``).

Confirm PROMISE itself is reachable:

.. code-block:: bash

   promise --version

Next Steps
==========

Continue to the :doc:`quickstart` guide to learn how to run your first benchmark.