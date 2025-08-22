fma_ions Documentation
====================

.. image:: https://img.shields.io/pypi/v/fma_ions.svg
   :target: https://pypi.org/project/fma_ions/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/fma_ions.svg
   :target: https://pypi.org/project/fma_ions/
   :alt: Python versions

A Python package for **Frequency Map Analysis (FMA)** of ion beams in particle accelerators, with a focus on the CERN accelerator complex.

Frequency Map Analysis is a method to probe tune diffusion from particle tracking simulations and identify resonances that enhance chaotic motion in particle accelerators.

**Key Features:**

- **FMA Analysis**: Track particles and analyze tune diffusion to detect chaotic behavior
- **Multi-Accelerator Support**: SPS, PS, and LEIR sequence generation and tracking
- **Advanced Physics**: Space charge, intra-beam scattering, and tune ripple effects
- **High Performance**: GPU acceleration support for large-scale simulations
- **HTCondor Integration**: Built-in support for cluster computing at CERN

Quick Start
-----------

Install the package:

.. code-block:: bash

   pip install fma_ions

Basic usage:

.. code-block:: python

   from fma_ions import FMA, SPS_Flat_Bottom_Tracker
   
   # Create FMA analysis
   fma = FMA()
   
   # Set up SPS tracking
   tracker = SPS_Flat_Bottom_Tracker()
   
   # Track particles and analyze tune diffusion
   fma.track_particles(tracker.particles)

Documentation
-------------
.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   installation
   quickstart
   examples
   physics

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   autoapi/index

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   contributing
   changelog

Package Components
------------------

Core Modules
~~~~~~~~~~~~

* **FMA**: Main class for Frequency Map Analysis with tune diffusion calculations
* **SPS_Flat_Bottom_Tracker**: Comprehensive particle tracking for SPS flat bottom operations
* **Submitter**: HTCondor job submission interface for CERN cluster computing
* **Tune_Ripple_SPS**: Power converter ripple simulation from 50 Hz grid oscillations

Accelerator Support
~~~~~~~~~~~~~~~~~~~

* **SPS**: Super Proton Synchrotron sequence generation and beam parameters
* **PS**: Proton Synchrotron lattice and ion beam configurations  
* **LEIR**: Low Energy Ion Ring sequence makers and beam setups

Physics Modules
~~~~~~~~~~~~~~~

* **Space Charge**: Frozen and adaptive space charge models
* **Longitudinal**: Parabolic and binomial distribution generators
* **Resonance Lines**: Tune diagram resonance identification
* **Plotting**: Specialized visualization tools for accelerator physics

Installation & Setup
--------------------

**Development Installation:** (not yet available on PyPI)

.. code-block:: bash

   git clone https://github.com/ewaagaard/fma_ions.git
   cd fma_ions
   pip install -e .

**GPU Support (Optional):**

.. code-block:: bash

   pip install cupy-cuda11x  # For CUDA 11.x
   pip install cupy-cuda12x  # For CUDA 12.x

**CERN Dependencies:**

For full functionality with CERN accelerator models:

.. code-block:: bash

   cd fma_ions/data/
   git clone https://gitlab.cern.ch/acc-models/acc-models-sps.git
   git clone https://gitlab.cern.ch/acc-models/acc-models-ps.git

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`