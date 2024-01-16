"""
Main class containing the tune ripple simulator for SPS
- inspired by xtrack example: https://xsuite.readthedocs.io/en/latest/fast_lattice_changes.html
"""
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pathlib import Path
import json

import xtrack as xt
import xpart as xp
import xfields as xf
import xobjects as xo

from .sequence_classes_ps import PS_sequence_maker, BeamParameters_PS
from .sequence_classes_sps import SPS_sequence_maker, BeamParameters_SPS
from .resonance_lines import resonance_lines
from .fma_ions import FMA

sequence_path = Path(__file__).resolve().parent.joinpath('../data/sps_sequences').absolute()

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
    ripple_period: int = 2000 
    num_turns: int = 10000
    
    
    def load_SPS_line_with_deferred_madx_expressions(self, use_symmetric_lattice=True, Qy_frac=25):
        """
        Loads xtrack Pb sequence file with deferred expressions to regulate QD and QF strengths
        or generate from MADX if does not exist
        
        Parameters:
        -----------
        use_symmetric_lattice - flag to use symmetric lattice without QFA and QDA
        Qy_fractional - fractional vertical tune. "19"" means fractional tune Qy = 0.19
        
        Returns:
        -------
        xtrack line
        """
        # Try loading existing json file, otherwise create new from MADX
        if use_symmetric_lattice:
            fname = '{}/qy_dot{}/SPS_2021_Pb_symmetric_deferred_exp.json'.format(sequence_path, Qy_frac)
        else:
            fname = '{}/qy_dot{}/SPS_2021_Pb_nominal_deferred_exp.json'.format(sequence_path, Qy_frac)
        
        try: 
            line = xt.Line.from_json(fname)
        except FileNotFoundError:
            print('\nSPS sequence file {} not found - generating new!\n'.format(fname))
            sps = SPS_sequence_maker()
            madx = sps.load_simple_madx_seq(use_symmetric_lattice, Qy_frac=25)
            madx.use(sequence="sps")
    
            # Convert to line
            line = xt.Line.from_madx_sequence(madx.sequence['sps'], deferred_expressions=True)
            m_in_eV, p_inj_SPS = sps.generate_SPS_beam()
            
            line.particle_ref = xp.Particles(
                    p0c = p_inj_SPS,
                    q0 = sps.Q_SPS,
                    mass0 = m_in_eV)
            line.build_tracker()
            
            with open(fname, 'w') as fid:
                json.dump(line.to_dict(), fid, cls=xo.JEncoder)
            
        twiss = line.twiss()
        
        print('\nGenerated SPS Pb beam with gamma = {:.3f}, Qx = {:.3f}, Qy = {:.3f}\n'.format(line.particle_ref.gamma0[0],
                                                                                              twiss['qx'],
                                                                                              twiss['qy']))
        return line
    
    
    def find_k_from_q_setvalue(self, dq=0.05, plane='X'):
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
        turns = np.arange(1, self.num_turns+1)
        kqf_vals = kqf_0 + amp_kqf * np.sin(2 * np.pi * turns / self.ripple_period)
        kqd_vals = kqd_0 + amp_kqd * np.sin(2 * np.pi * turns / self.ripple_period)
        
        return kqf_vals, kqd_vals, turns
        
      
    def find_k_from_xtrack_matching(self, dq=0.05, nr_matches=10, plane='X'):
        """
        Find desired tune amplitude modulation dQx or dQy by matching the global
        variable kqf and kqd
        
        Parameters:
        -----------
        period - oscillation period in number of turns
        nr_matches - interpolation resolution, number of times to match Qx and Qy with propoer kqf and kqd
        Qx_amplitude - dQx to vary, amplitude of oscillations
        """
        # Load Xsuite line with deferred expressions from MADx
        line = self.load_SPS_line_with_deferred_madx_expressions()
        twiss = line.twiss()
        
        # Empty arrays of quadrupolar strenghts:
        kqfs = np.zeros(nr_matches)
        kqds = np.zeros(nr_matches)
        Qx_vals = np.zeros(nr_matches)
        Qy_vals = np.zeros(nr_matches)
    
        # Investigate linear dependence
        if plane == 'X':
            Qx_target = np.linspace(np.round(twiss['qx'], 2), np.round(twiss['qx'], 2) + dq, num=nr_matches)
            
            for i, Q in enumerate(Qx_target): 
            
                print('\nMatching Qx = {}'.format(Q))
                
                # Match tunes to assigned values
                line.match(
                    vary=[
                        xt.Vary('kqf', step=1e-8),
                        xt.Vary('kqd', step=1e-8),
                    ],
                    targets = [
                        xt.Target('qx', Q, tol=1e-5),
                        xt.Target('qy', twiss['qy'], tol=1e-5)])
                
                twiss = line.twiss()
                
                # Add first values 
                kqfs[i] = line.vars['kqf']._value
                kqds[i] = line.vars['kqd']._value
                Qx_vals[i] = twiss['qx']
                Qy_vals[i] = twiss['qy']
                
                
            # Plot tune evolution over time
            fig, ax = plt.subplots(1, 2, figsize=(12,6))
            ax[0].plot(Qx_vals, kqfs, '-', color='blue')
            ax[1].plot(Qx_vals, kqds, '-', color='red')
            ax[0].set_xlabel('$Q_{x}$')
            ax[0].set_ylabel('kqf')
            ax[1].set_xlabel('$Q_{x}$')
            ax[1].set_ylabel('kqd')
            fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.show()
            
        elif plane == 'Y':
            Qy_target = np.linspace(np.round(twiss['qy'], 2), np.round(twiss['qy'], 2) + dq, num=nr_matches)
            
            for i, Q in enumerate(Qy_target): 
            
                print('\nMatching Qx = {}'.format(Q))
                
                # Match tunes to assigned values
                line.match(
                    vary=[
                        xt.Vary('kqf', step=1e-8),
                        xt.Vary('kqd', step=1e-8),
                    ],
                    targets = [
                        xt.Target('qx', twiss['qx'], tol=1e-5),
                        xt.Target('qy', Q, tol=1e-5)])
                
                twiss = line.twiss()
                    
                # Add first values 
                kqfs[i] = line.vars['kqf']._value
                kqds[i] = line.vars['kqd']._value
                Qx_vals[i] = twiss['qx']
                Qy_vals[i] = twiss['qy']
                
                # Plot tune evolution over time
                fig, ax = plt.subplots(1, 2, figsize=(12,6))
                ax[0].plot(Qy_vals, kqfs, '-', color='blue')
                ax[1].plot(Qy_vals, kqds, '-', color='red')
                ax[0].set_xlabel('$Q_{y}$')
                ax[0].set_ylabel('kqf')
                ax[1].set_xlabel('$Q_{y}$')
                ax[1].set_ylabel('kqd')
                fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
                plt.show()
        else:
            raise ValueError('Plane not valid - X or Y!')
        
        # Find amplitudes
        amp_kqf = kqfs[-1] - kqfs[0]
        amp_kqd = kqds[-1] - kqds[0]
        
        # Create arrays of quadrupole strengths to iterate over
        turns = np.arange(1, self.num_turns+1)
        kqf_vals = kqfs[0] + amp_kqf * np.sin(2 * np.pi * turns / self.ripple_period)
        kqd_vals = kqds[0] + amp_kqd * np.sin(2 * np.pi * turns / self.ripple_period)
        
        return kqf_vals, kqd_vals, turns
        
        
    
    
    def run_simple_ripple(self, dq=0.05, plane='X', use_xtrack_matching=True):
        """
        Run SPS standard tune ripple:
            
        Parameters:
        -----------
        period - oscillation period in number of turns
        Qx_amplitude - dQx to vary, amplitude of oscillations
        use_xtrack_matching - flag to use xtrack or MADX qh_setvalue for the matching
        """
        
        # Get SPS Pb line with deferred expressions
        line = self.load_SPS_line_with_deferred_madx_expressions()
        if use_xtrack_matching:
            kqf_vals, kqd_vals, turns = self.find_k_from_xtrack_matching(dq=dq, plane=plane)
        else:   
            kqf_vals, kqd_vals, turns = self.find_k_from_q_setvalue(dq=dq, plane=plane)
        
        # Generate particles
        fma_sps = FMA(n_linear=20)
        particles = fma_sps.generate_particles(line, BeamParameters_SPS, make_single_Jy_trace=True)
        
        # Empty array for turns
        Qx = np.zeros(len(kqf_vals))
        Qy = np.zeros(len(kqf_vals))
        
        for ii in range(self.num_turns):
            if ii % 20 == 0: print(f'Turn {ii} of {self.num_turns}')
        
            # Change the strength of the quads
            line.vars['kqf'] = kqf_vals[ii]
            line.vars['kqd'] = kqd_vals[ii]
        
            # Track one turn
            line.track(particles)
        
            # Check tunes
            twiss = line.twiss()
            Qx[ii] = twiss['qx']
            Qy[ii] = twiss['qy']

        # Plot tune evolution over time
        fig, ax = plt.subplots(1, 2, figsize=(12,6))
        fig.suptitle('Xtrack matching') if use_xtrack_matching else fig.suptitle('MADX qh_setvalue')
        ax[0].plot(turns, Qx, '-', color='blue', label='$Q_{x}$')
        ax[1].plot(turns, Qy, '-', color='red', label='$Q_{y}$')
        ax[0].set_ylabel('Tune')
        ax[0].set_xlabel('Turns')
        ax[1].set_xlabel('Turns')
        
        plt.show()
        
        return turns, Qx, Qy
    
    
    def print_quadrupolar_elements_in_line(self, line):
        """Print all quadrupolar elements"""
        
        # Print all quadrupolar components present
        my_dict = line.to_dict()
        d =  my_dict["elements"]
        for key, value in d.items():
            if value['__class__'] == 'Multipole' and value['_order'] == 1:
                print('{}: knl = {}'.format(key, value['knl']))