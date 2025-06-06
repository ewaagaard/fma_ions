"""fma_ions - A Python package for Frequency Map Analysis of ion beams at CERN.

This package provides tools for tracking particles and studying the behavior of ion beams
with effects such as space charge, intra-beam scattering, and tune ripple.
"""

__version__ = "0.1.0"

from .fma_ions import FMA, FMA_plotter

from .sequences import SPS_sequence_maker
from .sequences import PS_sequence_maker, BeamParameters_PS
from .sequences import LEIR_sequence_maker, BeamParameters_LEIR

from .beam_parameters import BeamParameters_SPS, BeamParameters_SPS_2024_2b, BeamParameters_SPS_Binomial_2016, BeamParameters_SPS_Binomial_2016_before_RF_capture, BeamParameters_SPS_Oxygen, BeamParameters_SPS_Proton


from .resonance_lines import resonance_lines
from .tune_ripple import Tune_Ripple_SPS
from .sps_flat_bottom_tracking import SPS_Flat_Bottom_Tracker
from .submitter import Submitter

from .longitudinal import generate_parabolic_distribution
from .longitudinal import generate_binomial_distribution
from .longitudinal import generate_binomial_distribution_from_PS_extr
from .longitudinal import generate_particles_transverse_gaussian, build_particles_linear_in_zeta, return_separatrix_coordinates

from .plotting import SPS_Plotting, SPS_Kick_Plotter

from .helpers_and_functions import Fit_Functions, FMA_keeper, Records, Records_Growth_Rates, Full_Records, Zeta_Container, Longitudinal_Monitor
