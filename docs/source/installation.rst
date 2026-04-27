============
Installation
============

This guide will help you set up HPC-Mix-Bench on your system.

Prerequisites
=============

* Python 3.8 or higher
* GCC/G++ compiler with C++11 support
* Docker (optional, for containerized deployment)

Installing CADNA-PROMISE
=========================

The benchmark tool requires ``cadnaPromise`` to be installed. Install it using pip:

.. code-block:: bash

   pip install cadnaPromise

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

After entering the Docker container, activate CADNA-PROMISE:

.. code-block:: bash

   activate-promise

Verification
============

Verify your installation by running a simple test:

.. code-block:: bash

   cd mp_tests
   ls

You should see various benchmark directories (e.g., ``backprop``, ``hotspot``, ``dense_lu``).

Next Steps
==========

Continue to the :doc:`quickstart` guide to learn how to run your first benchmark.