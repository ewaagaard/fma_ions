"""
Data containers for beam parameters and sequences 
"""
from dataclasses import dataclass
import xtrack as xt
from pathlib import Path

# Different Pb ion sequences from Xsuite
sps_fname = Path(__file__).resolve().parent.joinpath('../sequences/sps/SPS_2021_Pb_ions_matched_with_RF.json').absolute()
ps_fname = Path(__file__).resolve().parent.joinpath('../sequences/ps/PS_2022_Pb_ions_matched_with_RF.json').absolute()

'''  TO BE DEPRECATED - PARAMETERS ARE ALREADY IN SEQUENCE CLASSES
@dataclass
class BeamParameters_SPS :
    """Data Container for SPS Pb default beam parameters"""
    Nb:  float = 2.2e8 #3.5e8
    sigma_z: float = 0.225
    sigma_z_binomial: float = 0.225
    exn: float = 1.3e-6
    eyn: float = 0.9e-6
    Qx_int: float = 26.
    Qy_int: float = 26.
    
@dataclass
class BeamParameters_PS :
    """
    Data Container for PS Pb default beam parameters
    -> updated to 2023 values observed during PS MDs
    """
    Nb: float = 6.0e8  # 6.5e10 charges over two bunches
    sigma_z: float = 6.0  # injection bunch length in PS, but fluctuates if bunch splitting happens  
    exn: float = 0.75e-6
    eyn: float = 0.5e-6    
    Qx_int: float = 6.
    Qy_int: float = 6.
'''
  
class Sequences: 
    """Container of Xsuite default Pb sequences"""
    
    @staticmethod 
    def get_PS_line_and_twiss():
        """Extract Xsuite PS sequence""" 
        print('\nLoading PS sequence...')
        ps_line = xt.Line.from_json(ps_fname)
        ps_line.build_tracker()
        twiss_ps = ps_line.twiss() 
    
        return ps_line, twiss_ps

    @staticmethod 
    def get_SPS_line_and_twiss():
        """Extract Xsuite SPS sequence""" 
        print('\nLoading SPS sequence...')
        sps_line = xt.Line.from_json(sps_fname)
        sps_line.build_tracker()
        twiss_sps = sps_line.twiss() 
        
        return sps_line, twiss_sps


