============
Contributing
============

Thank you for your interest in contributing to HPC-Mix-Bench! This guide will help you add new benchmarks and improve the project.

Adding a New Benchmark
======================

Step 1: Create Benchmark Directory
-----------------------------------

Create a new directory in ``mp_tests/`` for your benchmark:

.. code-block:: bash

   cd mp_tests
   mkdir my_new_benchmark
   cd my_new_benchmark

Step 2: Add Source Code
------------------------

Add your C/C++ source files. The code should:

* Be self-contained or have minimal dependencies
* Include a ``main()`` function
* Accept command-line arguments for input parameters
* Output numerical results to stdout or file

**Example structure**:

.. code-block:: cpp

   // my_algorithm.cpp
   #include <iostream>
   #include <vector>
   
   int main(int argc, char** argv) {
       // Parse arguments
       int size = atoi(argv[1]);
       
       // Your algorithm here
       std::vector<double> data(size);
       // ... computation ...
       
       // Output results
       for (double val : data) {
           std::cout << val << std::endl;
       }
       
       return 0;
   }

Step 3: Create promise.yml
---------------------------

Define the compile/run commands PROMISE uses (must link CADNA with ``g++``):

.. code-block:: yaml

   compile:
   - g++ -O3 my_algorithm.cpp -frounding-math -m64 -o my_algorithm.out -lcadnaC -L$CADNA_PATH/lib -I$CADNA_PATH/include
   run: my_algorithm.out
   files: my_algorithm.cpp
   log: my_algorithm.log
   output: debug/

Step 4: Create fp.json
-----------------------

Define floating-point format search space (or copy the shared ``run_settings/fp.json`` template):

.. code-block:: json

    {   
    "c": [4, 3],
    "w": [5, 2],
    "b": [8, 7],
    "p": [5, 10],
    "h": [5, 10],
    "s": [8, 23],
    "d": [11, 52],
    "q": [15, 112],
    "o": [19, 236]
    }

Step 5: Synchronize Settings
-----------------------------

Copy run settings from the global configuration:

.. code-block:: bash

   cd ..
   bash sync_settings.sh --broadcast

This will copy ``run_setting_*.py`` files to your benchmark directory.

Step 6: Test Your Benchmark
----------------------------

Run the benchmark to verify it works:

.. code-block:: bash

   cd my_new_benchmark
   python3 run_setting_1.py

Step 7: Run All Configurations
-------------------------------

Use the automation script:

.. code-block:: bash

   cd ..
   ./run_benchmarks.sh 1 1 my_new_benchmark

Benchmark Guidelines
====================

Code Quality
------------

* Use clear variable names
* Add comments explaining algorithm steps
* Follow C++ best practices
* Avoid platform-specific code

Numerical Stability
-------------------

* Use appropriate numerical algorithms
* Consider conditioning and error propagation
* Document known numerical sensitivities
* Include reference implementations when possible

Performance
-----------

* Avoid unnecessary I/O
* Use efficient data structures
* Consider cache locality
* Profile and optimize hot paths

Documentation
-------------

Add a README.md in your benchmark directory:

.. code-block:: markdown

   # My New Benchmark
   
   ## Description
   Brief description of the algorithm and its purpose.
   
   ## Algorithm
   Mathematical formulation and computational steps.
   
   ## Input Parameters
   - `size`: Problem size
   - `tolerance`: Convergence threshold
   
   ## Expected Output
   Description of output format and expected results.
   
   ## References
   - Paper or algorithm source
   - Related work

Submitting Contributions
=========================

Preparing Your Contribution
----------------------------

1. **Fork the repository** on GitHub
2. **Create a branch** for your benchmark:

   .. code-block:: bash

      git checkout -b add-my-benchmark

3. **Add your files**:

   .. code-block:: bash

      git add mp_tests/my_new_benchmark/
      git commit -m "Add my_new_benchmark"

4. **Run tests**:

   .. code-block:: bash

      ./run_benchmarks.sh 1 1 my_new_benchmark

5. **Push your branch**:

   .. code-block:: bash

      git push origin add-my-benchmark

Creating a Pull Request
-----------------------

1. Go to the `HPC-Mix-Bench repository <https://github.com/PEQUAN/hpc-mix-bench>`_
2. Click "New Pull Request"
3. Select your branch
4. Fill in the template:

   * **Title**: Concise description (e.g., "Add FFT benchmark")
   * **Description**: 
     
     * What does your benchmark do?
     * What numerical properties does it test?
     * Have you run all four precision combinations?
     * Any known issues or limitations?

5. Submit the pull request

Pull Request Checklist
----------------------

Before submitting, ensure:

- [ ] Code compiles without errors
- [ ] All four precision combinations run successfully
- [ ] README.md is included with documentation
- [ ] promise.yml and fp.json are properly configured
- [ ] Plots are generated correctly
- [ ] Code follows project style guidelines
- [ ] No unnecessary files are included (build artifacts, temporary files)

Code Review Process
===================

After submission:

1. **Automated checks** will run (compilation, basic tests)
2. **Maintainers review** your code
3. **Feedback** may be provided for improvements
4. **Iterate** based on feedback
5. **Merge** once approved

Typical review time: 1-2 weeks

Improving Documentation
=======================

Documentation contributions are highly valued!

Types of Documentation
----------------------

* **Tutorials**: Step-by-step guides for specific use cases
* **Explanations**: Conceptual documentation about mixed precision
* **Reference**: API documentation and configuration options
* **Examples**: Sample benchmarks and use cases

How to Contribute Documentation
--------------------------------

1. Edit ``.rst`` files in ``docs/source/``
2. Build locally to preview:

   .. code-block:: bash

      cd docs
      make html
      # Open _build/html/index.html in browser

3. Submit a pull request with your changes

Reporting Issues
================

Found a bug or have a suggestion?

Bug Reports
-----------

Include:

* **Description**: What went wrong?
* **Steps to reproduce**: How can we trigger the bug?
* **Expected behavior**: What should happen?
* **Actual behavior**: What actually happened?
* **Environment**: OS, compiler version, Python version
* **Logs**: Relevant error messages or output

Feature Requests
----------------

Include:

* **Use case**: Why is this feature needed?
* **Proposed solution**: How should it work?
* **Alternatives**: What other approaches did you consider?
* **Impact**: Who would benefit from this feature?

Communication Channels
======================

* **GitHub Issues**: Bug reports, feature requests
* **Pull Requests**: Code contributions
* **Discussions**: General questions, ideas

Community Guidelines
====================

* Be respectful and inclusive
* Provide constructive feedback
* Help others learn
* Follow best practices
* Document your contributions

License
=======

By contributing to HPC-Mix-Bench, you agree that your contributions will be licensed under the MIT License.

Recognition
===========

Contributors will be acknowledged in:

* Project README
* Release notes
* Documentation credits

Thank you for making HPC-Mix-Bench better!

Next Steps
==========

* Review the :doc:`api_reference` for technical details
* Explore existing benchmarks in :doc:`benchmark_results`
* Join the discussion on GitHub