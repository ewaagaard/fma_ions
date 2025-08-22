Installation Guide
==================

Requirements
------------

* Python 3.8 or higher
* Git (for cloning CERN accelerator models)
* CUDA toolkit (optional, for GPU acceleration)

Basic Installation
------------------

Install directly from PyPI:

.. code-block:: bash

   pip install fma_ions

Development Installation
------------------------

For development or to access the latest features:

.. code-block:: bash

   git clone https://github.com/ewaagaard/fma_ions.git
   cd fma_ions
   pip install -e .

Virtual Environment Setup
--------------------------

Using conda (recommended):

.. code-block:: bash

   conda create --name fma_env python=3.11
   conda activate fma_env
   pip install fma_ions

Using venv:

.. code-block:: bash

   python -m venv fma_env
   source fma_env/bin/activate  # On Windows: fma_env\Scripts\activate
   pip install fma_ions

CERN Accelerator Models
-----------------------

To use the full functionality with CERN accelerator sequences:

.. code-block:: bash

   cd fma_ions/data/
   git clone https://gitlab.cern.ch/acc-models/acc-models-sps.git
   git clone https://gitlab.cern.ch/acc-models/acc-models-ps.git

These repositories contain the MADX sequence files needed for SPS and PS simulations.

GPU Support
-----------

For faster tracking with GPU acceleration:

**CUDA 11.x:**

.. code-block:: bash

   pip install cupy-cuda11x

**CUDA 12.x:**

.. code-block:: bash

   pip install cupy-cuda12x

**Additional setup (if needed):**

.. code-block:: bash

   conda install mamba -n base -c conda-forge
   mamba install cudatoolkit=11.8.0

Verification
------------

Test your installation:

.. code-block:: python

   import fma_ions
   print(fma_ions.__version__)
   
   # Test basic functionality
   from fma_ions import FMA
   fma = FMA()
   print("Installation successful!")

Troubleshooting
---------------

**Common Issues:**

1. **Missing CERN models**: Ensure acc-models repositories are cloned in the correct location
2. **GPU errors**: Verify CUDA installation and cupy compatibility
3. **Import errors**: Check that all dependencies are installed correctly

**Get help:**

- Check the `examples/` directory for usage patterns
- Review the API documentation
- Open an issue on GitHub for bugs or feature requests
