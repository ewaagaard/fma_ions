"""
Data containers for beam parameters and sequences 
"""
from dataclasses import dataclass
import xtrack as xt
from pathlib import Path

# Different Pb ion sequences from Xsuite
sps_fname = Path(__file__).resolve().parent.joinpath('../sequences/sps/SPS_2021_Pb_ions_matched_with_RF.json').absolute()
ps_fname = Path(__file__).resolve().parent.joinpath('../sequences/ps/PS_2022_Pb_ions_matched_with_RF.json').absolute()

@dataclass
class BeamParameters_SPS :
    """Data Container for SPS Pb default beam parameters"""
    Nb:  float = 2.2e8 #3.5e8
    sigma_z: float = 0.225
    exn: float = 1.3e-6
    eyn: float = 0.9e-6
    Q_int: float = 26.
    
@dataclass
class BeamParameters_PS :
    """Data Container for PS Pb default beam parameters"""
    Nb: float = 8.1e8
    sigma_z: float = 4.74
    exn: float = 0.8e-6
    eyn: float = 0.5e-6    
    Q_int: float = 6.

@dataclass    
class PS_Sequence: 
    """Data container for Xsuite PS sequence"""
    print('\nLoading PS sequence...')
    ps_line = xt.Line.from_json(ps_fname)
    ps_line.build_tracker()
    twiss_ps = ps_line.twiss() 
    
@dataclass    
class SPS_Sequence: 
    """Data container for Xsuite SPS sequence"""
    print('\nLoading SPS sequence...')
    sps_line = xt.Line.from_json(sps_fname)
    sps_line.build_tracker()
    twiss_sps = sps_line.twiss() 
