# FMA Ions

[![PyPI version](https://badge.fury.io/py/fma-ions.svg)](https://badge.fury.io/py/fma-ions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A Python package for Frequency Map Analysis of ion beams in particle accelerators**

Frequency Map Analysis (FMA) is a powerful technique for studying nonlinear dynamics and detecting chaotic motion in particle accelerators. This package provides comprehensive tools for FMA studies of ion beams at CERN, with support for space charge effects, intra-beam scattering, and tune ripple.

## Key Features

- **🔬 Frequency Map Analysis**: Tune diffusion calculation and resonance identification
- **⚡ High Performance**: GPU acceleration support for large-scale tracking
- **🏗️ CERN Integration**: Built-in support for SPS, PS, and LEIR accelerators  
- **☁️ Cluster Computing**: HTCondor job submission for batch processing
- **📊 Visualization**: Specialized plotting tools for accelerator physics
- **🧮 Advanced Physics**: Space charge, IBS, and tune ripple modeling

## Installation

```bash
pip install fma_ions
```

For GPU support:
```bash
pip install cupy-cuda11x  # CUDA 11.x
pip install cupy-cuda12x  # CUDA 12.x
```

For CERN accelerator models:
```bash
cd fma_ions/data/
git clone https://gitlab.cern.ch/acc-models/acc-models-sps.git
git clone https://gitlab.cern.ch/acc-models/acc-models-ps.git
```

## Quick Start

```python
from fma_ions import FMA, SPS_Flat_Bottom_Tracker

# Basic FMA analysis
fma = FMA()
d, Qx, Qy = fma.run_SPS()  # Run default SPS analysis
fma.plot_tune_diagram(Qx, Qy, d)

# Comprehensive SPS tracking with space charge
tracker = SPS_Flat_Bottom_Tracker()
tracker.track_SPS(n_turns=1000)
tracker.plot_tracking_data()
```

## Physics Background

FMA detects chaotic motion by analyzing tune diffusion:

$$d = \log_{10} \sqrt{(Q_{x,2} - Q_{x,1})^2 + (Q_{y,2} - Q_{y,1})^2}$$

Where $Q_{x,y,1}$ and $Q_{x,y,2}$ are tunes calculated from different turn blocks.

**Interpretation:**
- $d < -5$: Regular motion (stable)
- $d > -3$: Chaotic motion (unstable)

## Package Structure

- **`FMA`**: Core analysis class with tune diffusion calculations
- **`SPS_Flat_Bottom_Tracker`**: Comprehensive SPS particle tracking
- **`Submitter`**: HTCondor job submission for CERN cluster
- **`Tune_Ripple_SPS`**: Power converter ripple simulation
- **Sequence makers**: PS, SPS, and LEIR lattice generation
- **Beam parameters**: Ion-specific configurations for different species

## Documentation

Comprehensive documentation is available at: https://fma-ions.readthedocs.io/

- [Installation Guide](https://fma-ions.readthedocs.io/en/latest/installation.html)
- [Quick Start Tutorial](https://fma-ions.readthedocs.io/en/latest/quickstart.html)
- [Physics Background](https://fma-ions.readthedocs.io/en/latest/physics.html)
- [API Reference](https://fma-ions.readthedocs.io/en/latest/autoapi/index.html)

## Examples

See the [`examples/`](examples/) directory and [documentation examples](https://fma-ions.readthedocs.io/en/latest/examples.html) for:

- Dynamic aperture studies
- Space charge effect analysis  
- Tune ripple impact assessment
- Multi-species ion comparisons
- HTCondor batch processing

## Contributing

We welcome contributions! Please see our [Contributing Guide](https://fma-ions.readthedocs.io/en/latest/contributing.html) for details on:

- Development setup
- Code style guidelines  
- Testing procedures
- Documentation standards

## Citation

If you use this software in your research, please cite:

```bibtex
@software{waagaard2024fma_ions,
  author = {Elias Waagaard},
  title = {fma_ions: Frequency Map Analysis for ion beams at CERN},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/ewaagaard/fma_ions}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- CERN for accelerator models and computing resources
- Xsuite team for tracking infrastructure
- NAFF library for frequency analysis algorithms



