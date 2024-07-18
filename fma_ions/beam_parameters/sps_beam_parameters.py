"""
Main container for SPS beam parameters
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class BeamParameters_SPS:
    """Data Container for SPS Pb default beam parameters (Gaussian)"""
    Nb:  float = 2.46e8 # measured 2.46e8 ions per bunch on 2023-10-16
    sigma_z: float = 0.213 # 0.225 in m, is the old value (close to Isabelle's and  Hannes'), but then bucket is too full if Gaussian longitudinal. 0.19 also used
    exn: float = 1.1e-6
    eyn: float = 0.9e-6
    q : float = 0.59 # q-Gaussian parameter after RF spill (third profile)

@dataclass
class BeamParameters_SPS_Binomial_2016:
    """
    Data Container for SPS Pb longitudinally binomial/qgaussian beam parameters, from 2016 measurements,
    after RF capture at SPS injection
    """
    Nb: float = 3.536e8 # injected intensity, after initial spill out of RF bucket
    sigma_z: float = 0.213 # RMS bunch length of binomial, after initial spill out of RF bucket #0.213 measured, but takes ~30 turns to stabilze
    m : float = 2.98 # binomial parameter to determine tail of parabolic distribution, after initial spill out of RF bucket
    q : float = 0.59 # q-Gaussian parameter after RF spill (third profile)
    exn: float = 1.3e-6
    eyn: float = 0.9e-6

@dataclass
class BeamParameters_SPS_Binomial_2016_before_RF_capture:
    """
    Data Container for SPS Pb longitudinally binomial beam parameters, from 2016 measurements, 
    before RF capture, matched to PS extraction
    """
    Nb:  float = 3.722e8  # injected bunch intensity measured with Wall Current Monitor (WCM)
    sigma_z: float = 0.286 # RMS bunch length of binomial, measured before RF spill
    m : float = 6.124 # binomial parameter to determine tail of parabolic distribution
    q : float = 0.82 # q-Gaussian parameter
    exn: float = 1.3e-6
    eyn: float = 0.9e-6

@dataclass
class BeamParameters_SPS_Oxygen:
    """Data Container for SPS oxygen beam parameters"""
    Nb:  float = 25e8 # half of (John, Bartosik 2021) for oxygen, assuming bunch splitting
    sigma_z: float = 0.213 # assume same as Pb
    q : float = 0.59 # q-Gaussian parameter after RF spill --> assume same as Pb
    exn: float = 1.3e-6
    eyn: float = 0.9e-6

@dataclass
class BeamParameters_SPS_Proton:
    """Data Container for SPS proton default beam parameters"""
    Nb:  float = 1e11 # 
    sigma_z: float = 0.22 #
    sigma_z_binomial: float = 0.285 # RMS bunch length of binomial, default value to match data
    exn: float = 0.65e-6 # to get same tune spread as Pb, 2.5e-6 is old test values for round proton beams
    eyn: float = 0.65e-6

