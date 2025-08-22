"""fma_ions - A Python package for Frequency Map Analysis of ion beams at CERN.

This package provides tools for tracking particles and studying the behavior of ion beams
with effects such as space charge, intra-beam scattering, and tune ripple.

Main Components:
    FMA: Core Frequency Map Analysis functionality
    SPS_Flat_Bottom_Tracker: Comprehensive SPS particle tracking
    Submitter: HTCondor job submission for CERN cluster computing
    Tune_Ripple_SPS: Power converter ripple simulation

Accelerator Support:
    - SPS: Super Proton Synchrotron
    - PS: Proton Synchrotron  
    - LEIR: Low Energy Ion Ring
"""

__version__ = "0.1.0"
__author__ = "Elias Waagaard"
__email__ = "elias.walter.waagaard@cern.ch"

# Core FMA functionality
from .fma_ions import FMA, FMA_plotter

# Particle tracking
from .sps_flat_bottom_tracking import SPS_Flat_Bottom_Tracker

# Accelerator sequences and beam parameters
from .sequences import (
    SPS_sequence_maker,
    PS_sequence_maker, 
    LEIR_sequence_maker
)

from .beam_parameters import (
    BeamParameters_PS,
    BeamParameters_LEIR,
    BeamParameters_SPS,
    BeamParameters_SPS_2024_2b,
    BeamParameters_SPS_Binomial_2016,
    BeamParameters_SPS_Binomial_2016_before_RF_capture,
    BeamParameters_SPS_Oxygen,
    BeamParameters_SPS_Proton
)

# Physics modules
from .resonance_lines import resonance_lines
from .tune_ripple import Tune_Ripple_SPS

# Cluster computing
from .submitter import Submitter

# Longitudinal beam dynamics
from .longitudinal import (
    generate_parabolic_distribution,
    generate_binomial_distribution,
    generate_binomial_distribution_from_PS_extr,
    generate_particles_transverse_gaussian,
    build_particles_linear_in_zeta,
    return_separatrix_coordinates
)

# Visualization
from .plotting import SPS_Plotting, SPS_Kick_Plotter

# Helper functions and data structures
from .helpers_and_functions import (
    Fit_Functions,
    FMA_keeper,
    Records,
    Records_Growth_Rates,
    Full_Records,
    Zeta_Container,
    Longitudinal_Monitor
)

# Define what gets imported with "from fma_ions import *"
__all__ = [
    # Core classes
    'FMA',
    'FMA_plotter',
    'SPS_Flat_Bottom_Tracker',
    'Submitter',
    'Tune_Ripple_SPS',
    
    # Sequence makers
    'SPS_sequence_maker',
    'PS_sequence_maker', 
    'LEIR_sequence_maker',
    
    # Beam parameters
    'BeamParameters_PS',
    'BeamParameters_LEIR', 
    'BeamParameters_SPS',
    'BeamParameters_SPS_2024_2b',
    'BeamParameters_SPS_Binomial_2016',
    'BeamParameters_SPS_Binomial_2016_before_RF_capture',
    'BeamParameters_SPS_Oxygen',
    'BeamParameters_SPS_Proton',
    
    # Physics
    'resonance_lines',
    
    # Longitudinal
    'generate_parabolic_distribution',
    'generate_binomial_distribution',
    'generate_binomial_distribution_from_PS_extr',
    'generate_particles_transverse_gaussian',
    'build_particles_linear_in_zeta',
    'return_separatrix_coordinates',
    
    # Plotting
    'SPS_Plotting',
    'SPS_Kick_Plotter',
    
    # Helpers
    'Fit_Functions',
    'FMA_keeper',
    'Records',
    'Records_Growth_Rates',
    'Full_Records',
    'Zeta_Container',
    'Longitudinal_Monitor'
]
