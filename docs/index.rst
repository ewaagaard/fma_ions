.. fma_ions documentation master file, created by
   sphinx-quickstart on Tue Jun  3 17:02:37 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _fma_ions:

fma_ions
========

.. image:: https://img.shields.io/pypi/v/fma_ions.svg
   :target: https://pypi.org/project/fma_ions/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/fma_ions.svg
   :target: https://pypi.org/project/fma_ions/
   :alt: Python versions

.. image:: https://github.com/ewaagaard/fma_ions/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/ewaagaard/fma_ions/actions
   :alt: Build Status

.. image:: https://readthedocs.org/projects/fma-ions/badge/?version=latest
   :target: https://fma-ions.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

A Python package for Frequency Map Analysis (FMA) of ion beams in particle accelerators, with a focus on the CERN accelerator complex.

Features
--------

- Track particles through accelerator lattices
- Analyze tune diffusion and identify resonances
- Study effects of space charge, intra-beam scattering, and tune ripple
- Support for various CERN accelerators (SPS, PS, LEIR)
- GPU acceleration support for faster tracking

Installation
------------

.. code-block:: bash

   pip install fma_ions

For GPU support, install the appropriate version of `cupy` for your CUDA version:

.. code-block:: bash

   pip install cupy-cuda11x  # For CUDA 11.x

Documentation
-------------
For full documentation, including installation, usage, and API reference, see:

https://fma-ions.readthedocs.io/

Quick Start
-----------

.. code-block:: python

   from fma_ions import FMA, SPS_sequence_maker, BeamParameters_SPS
   import numpy as np
   
   # Create SPS sequence
   sps = SPS_sequence_maker()
   line, twiss = sps.load_xsuite_line_and_twiss()
   
   # Set up beam parameters
   beam_params = BeamParameters_SPS()
   
   # Initialize FMA
   fma = FMA(line, beam_params)
   
   # Generate particles and track
   particles = fma.generate_particles()
   x, y = fma.track_particles(particles)
   
   # Run FMA analysis
   d, Qx, Qy = fma.run_FMA(x, y)
   
   # Plot results
   fma.plot_tune_diagram(Qx, Qy, d)

API Reference
============

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   autoapi/index

   If the API reference doesn't appear, make sure you have run:
   
   .. code-block:: bash
   
      sphinx-apidoc -o docs/autoapi/ ../fma_ions/
      make html

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

License
-------
This project is licensed under the MIT License - see the `LICENSE <https://github.com/ewaagaard/fma_ions/blob/main/LICENSE>`_ file for details.

Citation
--------
If you use this software in your research, please consider citing it.

.. code-block:: bibtex

   @software{waagaard2024fma_ions,
     author = {Elias Waagaard},
     title = {fma_ions: Frequency Map Analysis for ion beams at CERN},
     year = {2024},
     publisher = {GitHub},
     journal = {GitHub repository},
     howpublished = {\\url{https://github.com/ewaagaard/fma_ions}},
   }
