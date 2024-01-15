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
    
    
    def find_k_from_q_setvalue(self, dq=0.05, ripple_period=2000, total_turns=10000, plane='X'):
        """
        For desired tune amplitude modulation dQx or dQy, find corresponding change in quadrupole strengths
        'qh_setvalue' and 'qv_setvalue' are linear knobs that regulate QD and QF strength
        - https://gitlab.cern.ch/acc-models/acc-models-sps/-/blame/2021/toolkit/macro.madx?ref_type=heads#L81
        
        Parameters:
        -----------
        dq - change in tune amplitude, e.g. 0.05
        period - oscillation period of tune in number of turns
        plane - 'X' or 'Y' (default is 'X')
        """
        # Load MADX line of SPS and define quad knobs
        sps_seq = SPS_sequence_maker()
        madx = sps_seq.load_madx_SPS()
        madx.exec('sps_define_quad_knobs')
        
        # Find old magnet stregths
        kqf_0 = madx.globals['kqf']
        kqd_0 = madx.globals['kqd']
        print('\nOld strengths: kqf = {:.5f}, kqd = {:.5f}'.format(kqf_0, kqd_0))
        
        # Adjust the qh_setvalue or qv_setvalue
        if plane == 'X':
            madx.input('qh_setvalue = {};'.format(sps_seq.qx0 + dq))
        elif plane == 'Y':
            madx.input('qv_setvalue = {};'.format(sps_seq.qy0 + dq))
        else:
            raise ValueError('Undefined plane!')
        twiss = madx.twiss().summary
        print('New tunes: Qx = {:.6f} and Qy = {:.6f}'.format(twiss['q1'], twiss['q2']))
        
        # Find new quadrupole strength
        kqf_1 = madx.globals['kqf']
        kqd_1 = madx.globals['kqd']
        print('New strengths: kqf = {:.5f}, kqd = {:.5f}\n'.format(kqf_1, kqd_1))
        
        # Reset the qh_setvalue or qv_setvalue
        if plane == 'X':
            madx.input('qh_setvalue = {};'.format(sps_seq.qx0))
        elif plane == 'Y':
            madx.input('qv_setvalue = {};'.format(sps_seq.qy0))
        print('Q setvalue reset!')
        
        # Find amplitudes
        amp_kqf = np.abs(kqf_1 - kqf_0)
        amp_kqd = np.abs(kqd_1 - kqd_0)
        
        # Create arrays of quadrupole strengths to iterate over
        turns = np.arange(1, total_turns+1)
        kqf_vals = kqf_0 + amp_kqf * np.sin(2 * np.pi * turns / ripple_period)
        kqd_vals = kqd_0 + amp_kqd * np.sin(2 * np.pi * turns / ripple_period)
        
        return kqf_vals, kqd_vals
        
    
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
        
        # Extract list of elements to trim (all focusing quads) - only take the multipoles
        elements_to_trim = []
        for i, ele in enumerate(line.element_names):
            if ele.startswith('qf') and line.elements[i].__class__.__name__ == 'Multipole':
                elements_to_trim.append(ele)
                
        #elements_to_trim = [nn for nn in line.element_names if nn.startswith('qf.')]
    
        # Build a custom setter
        qf_setter = xt.MultiSetter(line, elements_to_trim,
                                    field='knl', index=1 # we want to change knl[1]
                                    )
        
        # Get the initial values of the quad strength
        k1l_0 = qf_setter.get_values()