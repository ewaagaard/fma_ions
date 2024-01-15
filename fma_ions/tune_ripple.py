"""
Main class containing the tune ripple simulator for SPS
- inspired by xtrack example: https://xsuite.readthedocs.io/en/latest/fast_lattice_changes.html
"""
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

import xtrack as xt
import xpart as xp
import xfields as xf

from .sequence_classes_ps import PS_sequence_maker, BeamParameters_PS
from .sequence_classes_sps import SPS_sequence_maker, BeamParameters_SPS
from .resonance_lines import resonance_lines
from .fma_ions import FMA


@dataclass
class Tune_Ripple_SPS:
    """
    Class to simulate tune ripple by varying quadrupole strengths
    
    Parameters:
    -----------
    Qy_fractional - fractional vertical tune. "19"" means fractional tune Qy = 0.19
    beta_beat - relative beta beat, i.e. relative difference between max beta function and max original beta function
    use_symmetric_lattice - flag to use symmetric lattice without QFA and QDA
    """
    Qy_frac: float = 25
    beta_beat: float = None
    use_symmetric_lattice = True
    
    
    def find_k_from_q_setvalue(self, dq, plane='X'):
        """
        For desired tune amplitude modulation dQx or dQy, find corresponding change in quadrupole strengths
        
        Parameters:
        -----------
        dq - change in tune amplitude, e.g. 0.05
        plane - default is 'X'
        """
        # Load MADX line of SPS
        sps_seq = SPS_sequence_maker()
        madx = sps_seq.load_madx_SPS()
        
    
    def run_simple_ripple(self, period, Qx_amplitude):
        """
        Run SPS standard tune ripple:
            
        Parameters:
        -----------
        period - oscillation period in number of turns
        Qx_amplitude - dQx to vary, amplitude of oscillations
        """
        
        # Load Xtrack line of SPS
        sps_seq = SPS_sequence_maker()
        line, twiss = sps_seq.load_xsuite_line_and_twiss(Qy_frac=self.Qy_frac, 
                                                         beta_beat=self.beta_beat, 
                                                         use_symmetric_lattice=self.use_symmetric_lattice)
        
        # Extract list of elements to trim (all focusing quads)
        elements_to_trim = [nn for nn in line.element_names if nn.startswith('qf.')]
    
        # Build a custom setter
        qf_setter = xt.MultiSetter(line, elements_to_trim,
                                    field='knl', index=1 # we want to change knl[1]
                                    )
        
        # Get the initial values of the quad strength
        k1l_0 = qf_setter.get_values()