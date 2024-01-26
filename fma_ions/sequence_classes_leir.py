"""
Main module for sequence generator container classes for LEIR

An overview of LEIR nominal optics can be found here: https://acc-models.web.cern.ch/acc-models/leir/2021/scenarios/nominal/
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
import os 
from scipy.optimize import minimize

import xobjects as xo
import xtrack as xt
import xpart as xp

from scipy import constants
from cpymad.madx import Madx
import json

optics =  Path(__file__).resolve().parent.joinpath('../data/acc-models-leir').absolute()
sequence_path = Path(__file__).resolve().parent.joinpath('../data/leir_sequences').absolute()
#error_file_path = Path(__file__).resolve().parent.joinpath('../data/sps_sequences/magnet_errors').absolute()

@dataclass
class BeamParameters_LEIR:
    """Data Container for LEIR Pb default beam parameters"""
    Nb:  float = 10.0e8 #3.5e8
    sigma_z: float = 8.0
    exn: float = 0.4e-6
    eyn: float = 0.4e-6
    Qx_int: float = 1.
    Qy_int: float = 2.
    
    
@dataclass
class LEIR_sequence_maker:
    """ 
    Data class to generate Xsuite line from SPS optics repo, selecting
    - qx0, qy0: horizontal and vertical tunes
    - dq1, dq2: X and Y chroma values 
    - m_ion: ion mass in atomic units
    - optics: absolute path to optics repository -> cloned from https://gitlab.cern.ch/acc-models
    """
    qx0: float = 1.82
    qy0: float = 2.72
    dq1: float = -3.460734474533172e-09 
    dq2: float = -3.14426538905229e-09
    # Default SPS PB ION CHROMA VALUES: not displayed on acc-model, extracted from PTC Twiss 
    
    # Define beam type - default is Pb
    ion_type: str = 'Pb'
    seq_name: str = 'nominal'
    seq_folder: str = 'sps'
    B_PS_extr: float = 1.2368 # [T] - magnetic field in PS for Pb ions, from Heiko Damerau
    Q_PS: float = 54.
    Q_SPS: float = 82.
    m_ion: float = 207.98