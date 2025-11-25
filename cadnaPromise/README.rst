cadnaPromise
=============

.. image:: https://img.shields.io/badge/License-GPLv3-yellowgreen.svg
   :target: LICENSE
   :alt: License

.. image:: https://gitlab.lip6.fr/hilaire/promise2/badges/master/pipeline.svg
   :target: pipeline
   :alt: Pipeline

--------------------------------------------------------------------------------------------

``cadnaPromise`` is a software for precision tuning via static analysis and dynamic profiling.  
It is based on the `CADNA library <https://cadna.lip6.fr/>`_, which implements Discrete Stochastic Arithmetic (DSA) to estimate round-off error propagation in numerical codes.

``cadnaPromise`` provides precision auto-tuning through command-line interfaces.

---------------
Install
--------------

To install ``cadnaPromise``:

.. code-block:: bash

   pip install cadnaPromise

After installation, enable CADNA support and arbitrary-precision customization:

.. code-block:: bash

   activate-promise

To deactivate:

.. code-block:: bash

   deactivate-promise

Alternatively, users may install ``floatx`` and ``CADNA`` manually and specify the path:

.. code-block:: bash

   export CADNA_PATH=[YOURPATH]

Check whether ``cadnaPromise`` is installed:

.. code-block:: bash

   promise --version

------------------------------
Usage and configuration
------------------------------

``cadnaPromise`` requires the following Python libraries:
``colorlog``, ``colorama``, ``pyyaml``, ``regex``.

Compilation of instrumented code requires ``g++``.  
Ensure these dependencies are installed for proper operation.

Before running PROMISE, a configuration file must be prepared.

Setting up ``promise.yml``
---------------------------------

Users can customize the configuration file ``promise.yml``.  
A sample configuration:

.. code-block:: yaml

   compile:
     - g++ [SOURCE FILES] -frounding-math -m64 -o [OUTPUT FILE] -lcadnaC -L$CADNA_PATH/lib -I$CADNA_PATH/include
   run: [OUTPUT FILE]
   files: [SOURCE FILES]
   log: [OUTPUT FILE LOG]
   output: debug/

Explanation:

- ``compile`` — command used to compile the code  
- ``run`` — command to execute the compiled program  
- ``files`` — files to instrument (default: all ``.cc`` files)  
- ``log`` — log output file (optional)  
- ``output`` — directory containing transformed code  

Marking the code
-----------------------

Use ``__PROMISE__`` to mark variables eligible for reduced precision.

Use the following macros to mark variables or arrays for checking:

- ``PROMISE_CHECK_VAR(variable)``
- ``PROMISE_CHECK_ARRAY(array, n_elements)``

Custom floating-point formats
------------------------------------

Users can define custom precisions in ``fp.json``.  
Built-in formats: ``h`` (half), ``s`` (single), ``d`` (double).

Example ``fp.json``:

.. code-block:: json

   {
     "c": [5, 2],
     "b": [8, 7],
     "h": [5, 10],
     "s": [8, 23],
     "d": [11, 52]
   }

Here, ``c`` corresponds to E5M2, and ``b`` to bfloat16.

Run the program
----------------------

Basic commands:

.. code-block:: bash

   # help
   promise --help

   # version
   promise --version

   # run program
   promise --precs=<letters> [options]

Options
---------------------

.. code-block:: text

   -h --help                     Show this screen.
   --version                     Show version.
   --precs=<strs>                Precision letters [default: sd]
   --conf CONF_FILE              Configuration file [default: promise.yml]
   --fp FPT_FILE                 FP format file [default: fp.json]
   --output OUTPUT               Output directory
   --verbosity VERBOSITY         Verbosity level (0–4) [default: 1]
   --log LOGFILE                 Log file (optional)
   --verbosityLog VERBOSITY      Log verbosity level
   --debug                       Store intermediate files and show execution trace
   --run RUN                     File/program to run
   --compile COMMAND             Compilation command
   --files FILES                 Files to examine (default: all .cc files)
   --nbDigits DIGITS             Required number of digits
   --path PATH                   Project path (default: current directory)
   --pause                       Pause between steps
   --parsing                     Parse only; no transformation
   --auto                        Enable automatic instrumentation
   --relError THRES              Use relative error threshold
   --noCadna                     Disable CADNA; use double-precision reference
   --alias ALIAS                 Allow command aliases (e.g. "g++=g++-14")
   --CC                          C compiler [default: g++]
   --CXX                         C++ compiler [default: g++]
   --plot                        Enable plotting of results [default: 1]

For detailed examples, see ``EXAMPLE.rst``.

----------------------
Acknowledgements
----------------------

``cadnaPromise`` is based on `Promise2 <https://gitlab.lip6.fr/hilaire/promise2>`_ (Hilaire et al.),  
a full rewrite of the original PROMISE (Picot et al.).

This work was supported by the France 2030 NumPEx Exa-MA (ANR-22-EXNU-0002) project funded by the French National Research Agency (ANR).  
``Promise2`` was also developed with support from the COMET project *Model-Based Condition Monitoring and Process Control Systems*, hosted by Materials Center Leoben Forschung GmbH.
