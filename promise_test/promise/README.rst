cadnaPromise
==============




.. image:: https://img.shields.io/badge/License-GPLv3-yellowgreen.svg
    :target: LICENSE
    :alt: License


.. image:: https://gitlab.lip6.fr/hilaire/promise2/badges/master/pipeline.svg
    :target: pipeline
    :alt: Pipeline
---- 

``cadnaPromise`` is a precision auto-tuning software using command-line interfaces.


--------
Install
--------

To install ``cadnaPromise``, simply use the pip command in terminal:  

.. parsed-literal::

  pip install cadnaPromise


after that, to enable the arbitrary precision customization and cadna installation, simply activate the ``cadnaPromise`` via the command in terminal:

.. parsed-literal::

  activate-promise


To reverse the process, simply do 

.. parsed-literal::

  deactivate-promise


Besides, users can install ``floatx`` and ``CADNA`` outside, and then specifying the path via

.. parsed-literal::

	export CADNA_PATH=[YOURPATH]





Check the if ``cadnaPromise`` is installed:

.. parsed-literal::

  promise --version


-------------
Dependencies
-------------

The installation of ``cadnaPromise`` requires the the following Python libraries: ``colorlog``, ``colorama``, ``pyyaml``, ``regex``.

The compiling of ``cadnaPromise`` requires ``g++``. Please ensure the installation of above libraries for a proper running of cadnaPromise.


-------------
Usage
-------------

Setting up ``promise.yml`` file
--------------------------------

.. parsed-literal::

	compile:
	- g++ [SOURCE FILES] -frounding-math -m64 -o [OURPUT FILE] -lcadnaC -L$CADNA_PATH/lib -I$CADNA_PATH/include
	run: [OURPUT FILE]
	files: [SOURCE FILES]
	log: [OURPUT FILE LOG]
	output: debug/


Mark the code
--------------

Use ``__PROMISE__`` to mark the variables to be used in low precision, and use ``PROMISE_CHECK_VAR([VARIABLE])`` and ``PROMISE_CHECK_ARRAY([ARRAY},[NUM OF ELEMENTS])`` to mark to variabke or array to be checked; 

Run the program in terminal
--------------------------------

In terminal, simply enter the command bellow: 

.. parsed-literal::

	get help:     promise --help | promise
        get version:  promise --version
	run program:  promise --precs=(customized precisions/built precisions) [options]


Options:

.. parsed-literal::

	-h --help                     Show this screen.
	--version                     Show version.
	--precs=<strs>                Set the precision following the built-in or cutomized precision letters [default: sd]
	--conf CONF_FILE              Get the configuration file [default: promise.yml]
	--fp FPT_FILE                 Get the file for floating point number format [default: fp.json]
	--output OUTPUT               Set the path of the output (where the result files are put)
	--verbosity VERBOSITY         Set the verbosity (betwen 0  and 4 for very low level debug) [default: 1]
	--log LOGFILE                 Set the log file (no log file if this is not defined)
	--verbosityLog VERBOSITY      Set the verbosity of the log file
	--debug                       Put intermediate files into `debug/` (and `compileErrors/` for compilation errrors) and display the execution trace when an error comes
	--run RUN                     File to be run
	--compile COMMAND             Command to compile the code
	--files FILES                 List of files to be examined by Promise (by default, all the .cc files)
	--nbDigits DIGITS             General required number of digits
	--path PATH                   Set the path of the project (by default, the current path)
	--pause                       Do pause between steps
	--noParsing                   Do not parse the C file (__PROMISE__ are replaced and that's all)
	--auto                        enable auto-instrumentation of source code
	--relError THRES              use criteria of precision relative error less than THRES instead of number of digits
	--noCadna                     will not use cadna, reference result computed in (non-stochastic) double precision
	--alias ALIAS                 Allow aliases (examples "g++=g++-14") [default:""]
	--CC        				          Set compiler for C program [default: g++]
	--CXX                         Set compiler for C++ program [default: g++]


-------------------
Acknowledgements
-------------------



``cadnaPromise`` is based on `Promise2 <https://gitlab.lip6.fr/hilaire/promise2>`_  (Hilaire et al), a full rewriting of the first PROMISE version (Picot et al).

This work was supported by the France 2030 NumPEx Exa-MA (ANR-22-EXNU-0002) project managed by the French National Research Agency (ANR).
``Promise2`` has been developed with the financial support of the COMET project Model-Based Condition Monitoring and Process Control Systems, hosted by the Materials Center Leoben Forschung GmbH.
