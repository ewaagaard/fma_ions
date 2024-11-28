"""
Main container for SPS beam parameters
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class BeamParameters_SPS:
    """Data Container for SPS Pb default beam parameters 2024"""
    Nb:  float = 3.4e8 # measured ions per bunch on 2024-11-13, at 16:23 (lowest amount of noise)
    sigma_z: float = 0.213 # measured with WCM on 2024-11-13
    exn: float = 2.31e-6 # measured 2.58 after 25 ms on 2024-11-19, but 2.31 on 2024-11-25
    eyn: float = 1.34e-6 # measured 1.34 after 25 ms on 2024-11-19, but 1.246 on 2024-11-25
    q : float = 0.72 # q-Gaussian parameter after RF spill
    m : float = 4.0 # binomial parameter to determine tail of parabolic distribution (approximate value)

@dataclass
class BeamParameters_SPS_2024_2b:
    """
    Data Container for SPS Pb 2b, with no PS splitting
    Use https://be-op-logbook.web.cern.ch/elogbook-server/GET/showEventInLogbook/4174679 as reference event
    """
    Nb:  float = 6.46e8 # measured with FBCT on 2024-10-30 with FBCT 
    sigma_z: float = 0.215 # assumed to be the same as with PS bunch splitting
    exn: float = 2.38e-6 # measured on 2024-10-30, at 25 ms after injection
    eyn: float = 1.39e-6
    q : float = 0.7 # assumed q-Gaussian parameter to be identical
    m : float = 4.0 # binomial parameter to determine tail of parabolic distribution (approximate value)

@dataclass
class BeamParameters_SPS_2023:
    """Data Container for SPS Pb default beam parameters (2023), assuming similar longitudinal to 2016"""
    Nb:  float = 2.46e8 # measured 2.46e8 ions per bunch on 2023-10-16
    sigma_z: float = 0.213 # 0.225 in m, is the old value (close to Isabelle's and  Hannes'), but then bucket is too full if Gaussian longitudinal. 0.19 also used
    exn: float = 1.1e-6
    eyn: float = 0.9e-6
    q : float = 0.59 # q-Gaussian parameter after RF spill (third profile)
    m : float = 2.98 # binomial parameter to determine tail of parabolic distribution, after initial spill out of RF bucket

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
    """
    Data Container for SPS proton default beam parameters
    
    Emittances and bunch intensity easured values in SPS with Q26 high brightness beam on 2024-07-19 
    """
    Nb:  float = 1.2e11
    sigma_z: float = 0.22
    sigma_z_binomial: float = 0.285 # RMS bunch length of binomial, default value to match data
    exn: float = 0.7e-6
    eyn: float = 0.7e-6

