Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

Added
~~~~~
- Comprehensive documentation structure
- Physics background documentation
- Examples and tutorials
- Contributing guidelines

Changed
~~~~~~~
- Improved package structure documentation
- Enhanced docstrings throughout codebase

Removed
~~~~~~~
- GitHub workflows (not needed for current development)

[0.1.0] - 2024-XX-XX
--------------------

Initial release of fma_ions.

Added
~~~~~
- **FMA Module**: Core Frequency Map Analysis functionality
  
  - Tune diffusion calculation
  - Particle tracking with xsuite integration
  - Resonance line plotting
  - GPU acceleration support

- **SPS Flat Bottom Tracking**: Comprehensive SPS simulation
  
  - Space charge effects (frozen and adaptive)
  - Intra-beam scattering (IBS)
  - RF capture modeling
  - Turn-by-turn data collection

- **Accelerator Support**:
  
  - SPS: Super Proton Synchrotron sequences and beam parameters
  - PS: Proton Synchrotron lattice generation
  - LEIR: Low Energy Ion Ring configurations

- **HTCondor Integration**:
  
  - Job submission to CERN computing cluster
  - Batch processing for parameter scans
  - Automated output management

- **Tune Ripple Simulation**:
  
  - 50 Hz power converter ripple modeling
  - Time-varying tune modulation
  - Grid oscillation effects

- **Longitudinal Beam Dynamics**:
  
  - Parabolic and binomial distribution generators
  - RF bucket modeling
  - Bunch length evolution

- **Visualization Tools**:
  
  - Tune diagrams with resonance lines
  - FMA diffusion plots
  - Emittance evolution tracking
  - Beam profile monitoring

- **Physics Models**:
  
  - Multiple ion species support (Pb, O, etc.)
  - Space charge tune shifts
  - IBS growth rates
  - Dynamic aperture studies

- **Performance Features**:
  
  - GPU acceleration with cupy
  - Vectorized calculations
  - Memory-efficient particle tracking
  - Batch processing capabilities

Technical Details
~~~~~~~~~~~~~~~~~

- **Dependencies**: xsuite, numpy, matplotlib, scipy, pandas
- **Python Support**: 3.8, 3.9, 3.10, 3.11
- **Accelerator Models**: Integration with CERN acc-models
- **Documentation**: Sphinx-based with autodoc
- **Testing**: pytest framework with CI/CD

Known Issues
~~~~~~~~~~~~

- GPU memory management for very large particle numbers
- Some legacy code patterns need modernization
- Documentation coverage could be improved for helper functions

Future Plans
~~~~~~~~~~~~

- Enhanced IBS models
- More accelerator configurations
- Better integration with CERN tools
- Performance optimizations
- Extended physics models
