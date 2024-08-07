"""
Main module for sequence generator container classes for PS 
"""
from dataclasses import dataclass

import numpy as np
import os 
from pathlib import Path

import xobjects as xo
import xtrack as xt
import xpart as xp

from scipy.optimize import minimize
from scipy import constants
from cpymad.madx import Madx
import json

sequence_path = Path(__file__).resolve().parent.joinpath('../../data/ps_sequences').absolute()
optics =  Path(__file__).resolve().parent.joinpath('../../data/acc-models-ps').absolute()

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

@dataclass
class PS_sequence_maker:
    """ 
    Data class to generate Xsuite line from PS optics repo, selecting
    - qx0, qy0: horizontal and vertical tunes
    - dq1, dq2: X and Y chroma values
    - Q_LEIR: ion charge state in LEIR
    - Q_PS: ion charge state in PS
    - m_ion: ion mass in atomic units
    - Brho: magnetic rigidity in T*m at injection
    - optics: absolute path to optics repository -> cloned from  https://acc-models.web.cern.ch/acc-models/ps/2022/scenarios/lhc_ion/
    Default chromaticity values originate from here. 
    """
    qx0: float = 6.15
    qy0: float = 6.245
    dq1: float =  -5.26716824
    dq2: float = -7.199251093
    # Default chroma values from optics repo
    
    # Define beam type - default is Pb
    ion_type: str = 'Pb'
    seq_name: str = 'nominal'
    seq_folder: str = 'ps'
    LEIR_Brho  = 4.8 # [Tm] -> Brho at PS injection - same as at LEIR extraction
    PS_Brho = 70.079 * 1.2368 # PS rho times PS_max_B_field for ions
    Q_LEIR: float = 54.
    Q_PS: float = 54.
    m_ion: float = 207.98

    def load_xsuite_line_and_twiss(self, beta_beat=None, at_injection_energy=True):
        """
        Method to load pre-generated PS lattice files for Xsuite
        
        Parameters:
        -----------
        beta_beat : float
            relative beta beat, i.e. relative difference between max beta function and max original beta function
        at_injection_energy : bool
            whether beam is at flat bottom (injection energy), or at extraction
        
        Returns:
        -------
        xsuite line
        twiss - twiss table from xtrack 
        """
        # Update fractional vertical tune 
        print('\nTrying to load sequence with Qx, Qy = ({}, {}) and beta-beat = {}!\n'.format(self.qx0, self.qy0, beta_beat))
        extraction_str = '' if at_injection_energy else '_extraction'

        # Check if pre-generated sequence exists 
        if beta_beat is None or beta_beat == 0.0:
            ps_fname = '{}/PS_2022_{}_nominal{}.json'.format(sequence_path , self.ion_type, extraction_str)
        else:                                                  
            ps_fname = '{}/PS_2022_{}_{}_percent_beta_beat.json'.format(sequence_path, self.ion_type, int(beta_beat*100))
            
        try:
            ps_line = xt.Line.from_json(ps_fname)
        except FileNotFoundError:
            print('\nPre-made PS sequence does not exists, generating new sequence with Qx, Qy = ({}, {}) and beta-beat = {}!\n'.format(self.qx0, 
                                                                                                                                         self.qy0, 
                                                                                                                                         beta_beat))
            ps_line = self.generate_xsuite_seq(at_injection_energy=at_injection_energy) if beta_beat is None else self.generate_xsuite_seq_with_beta_beat(beta_beat)
            
        # Build tracker and Twiss
        ps_line.build_tracker()
        twiss_ps = ps_line.twiss() 
        
        return ps_line, twiss_ps
    
    
    @staticmethod
    def generate_PS_gaussian_beam(line, n_part):
        """ 
        Class method to generate matched Gaussian beam for PS
        
        Parameters:
        -----------
        line - xtrack line object
        n_part - number of macroparticles
        
        Returns:
        --------
        particles - xpart particles object 
        
        """
        particles = xp.generate_matched_gaussian_bunch(
                                 num_particles=n_part, total_intensity_particles = BeamParameters_PS.Nb,
                                 nemitt_x = BeamParameters_PS.exn, nemitt_y = BeamParameters_PS.eyn, 
                                 sigma_z = BeamParameters_PS.sigma_z,
                                 particle_ref = line.particle_ref, line=line)
        print('\nGenerated Gaussian beam') 
        print('with Nb = {:.3e}, exn = {:.4e}, \neyn = {:.4e}, sigma_z = {:.3e} m \n{} macroparticles\n'.format(BeamParameters_PS.Nb,
                                                                                                                BeamParameters_PS.exn,
                                                                                                                BeamParameters_PS.eyn,
                                                                                                                BeamParameters_PS.sigma_z,
                                                                                                                n_part))
        return particles    

    def generate_PS_beam(self, at_injection_energy=True):
        """
        Generate correct injection parameters for PS beam

        Parameters:
        -----------
        at_injection_energy : bool
            whether beam is at flat bottom (injection energy), or at extraction

        Returns:
        -------
        ion rest mass in eV, beam momentum in eV/c at PS injection 
        """
        #print('\nCreating MADX-beam of {}\n'.format(self.ion_type))
        m_in_eV = self.m_ion * constants.physical_constants['atomic mass unit-electron volt relationship'][0]   # 1 Dalton in eV/c^2 -- atomic mass unit
        if at_injection_energy:
            p_beam = self.LEIR_Brho * self.Q_PS * constants.c # in  [eV/c], if q is number of elementary charges
        else: 
            p_beam = self.PS_Brho * self.Q_PS * constants.c # in  [eV/c], if q is number of elementary charges

        return m_in_eV, p_beam


    def generate_xsuite_seq(self, save_madx_seq=True, save_xsuite_seq=True, return_xsuite_line=True, at_injection_energy=True):
        """
        Load MADX line, match tunes and chroma, add RF and generate Xsuite line
        
        Parameters:
        -----------
        flags to save / return xsuite or madx lines
        at_injection_energy : bool
            whether beam is at injection energy (flat bottom) or extraction energy
        """
        os.makedirs(sequence_path, exist_ok=True)
        print('\nGenerating sequence for {} with qx = {}, qy = {}\n'.format(self.ion_type, self.qx0, self.qy0))
        
        #### Initiate MADX sequence and call the sequence and optics file ####
        madx = Madx()
        madx.call("{}/_scripts/macros.madx".format(optics))
        madx.call("{}/ps.seq".format(optics))
        if at_injection_energy:
            madx.call("{}/scenarios/lhc_ion/1_flat_bottom/ps_fb_ion.str".format(optics))
        else: 
            madx.call("{}/scenarios/lhc_ion/3_extraction/ps_ext_ion.str".format(optics))
        
        
        # Generate PS beam - use default Pb or make custom beam
        m_in_eV, p_PS = self.generate_PS_beam(at_injection_energy=at_injection_energy) # at injection or extraction
        
        madx.input(" \
                   Beam, particle=ion, mass={}, charge={}, pc = {}, sequence='ps'; \
                   DPP:=BEAM->SIGE*(BEAM->ENERGY/BEAM->PC)^2;  \
                   ".format(m_in_eV/1e9, self.Q_PS, p_PS/1e9))   # convert mass to GeV/c^2
           
        
        # When we perform matching, recall the MADX convention that all chromatic functions are multiplied by relativistic beta factor
        # Thus, when we match for a given chromaticity, need to account for this factor to get correct value in Xsuite and PTC
        self.particle_sample = xp.Particles(
                p0c = p_PS,
                q0 = self.Q_PS,
                mass0 = m_in_eV)
        beta0 = self.particle_sample.beta0[0]
        
        
        madx.input("Qx := {}".format(self.qx0 % 1))  # fractional part of tune
        madx.input("Qy := {}".format(self.qy0 % 1))
        madx.input(f"dqx := -5.26716824/{beta0}")
        madx.input(f"dqy := -7.199251093/{beta0}")
        
        # Flatten line
        madx.use("ps")
        madx.input("seqedit, sequence=PS;")
        madx.input("flatten;")
        madx.input("endedit;")
        madx.use("ps")
        madx.input("select, flag=makethin, slice=5, thick=false;")
        madx.input("makethin, sequence=ps, style=teapot, makedipedge=True;")
        madx.use('ps')
        
        # Match tunes with PTC, not with custom macro
        madx.use("ps")
        madx.input('exec, match_tunes(Qx, Qy, dqx, dqy);')
        madx.input('exec, tunes_leq_knob_factors(Qx, Qy);')

        # madx.input("exec, match_tunes_and_chroma(qx, qy, qpx, qpy);")
        
        # Create Xsuite line, check that Twiss command works 
        madx.use(sequence='ps')
        twiss_thin = madx.twiss()  
        
        line = xt.Line.from_madx_sequence(madx.sequence['ps'])
        line.build_tracker()
        
        print('\nGenerated PS {} beam with gamma = {:.3f}, Qx = {:.3f}, Qy = {:.3f}\ndq1 = {:.4f}, dq2 = {:.4f}\n'.format(self.ion_type, 
                                                                                              self.particle_sample.gamma0[0],
                                                                                              twiss_thin.summary['q1'],
                                                                                              twiss_thin.summary['q2'],
                                                                                              twiss_thin.summary['dq1']*beta0,
                                                                                              twiss_thin.summary['dq2']*beta0))
        
        line.particle_ref = self.particle_sample
        
        #### SET CAVITY VOLTAGE - with info from Alexandre Lasheen
        # Ions: 10 MHz cavities: 1.7 MV, h=16 at injection
        # at extraction: 300 kV, h=269 for ions
        # In the ps_ss.seq, the 10 MHz cavities are PR_ACC10 - there seems to be 12 of them in the straight sections
        harmonic_nb = 16 if at_injection_energy else 169
        nn = 'pa.c10.11'  # for now test the first of the RF cavities 
        # RF voltage: checked for IonLifeTime cycles at LSA for injection, Alexandre provided extraction value
        V_RF = 38.0958 if at_injection_energy else 300.0 # kV
        lag = 0 if at_injection_energy else 180
        extraction_str = '' if at_injection_energy else '_extraction'

        # MADX sequence 
        madx.sequence.ps.elements[nn].lag = lag
        madx.sequence.ps.elements[nn].volt = V_RF*1e-3*self.particle_sample.q0 # different convention between madx and xsuite
        madx.sequence.ps.elements[nn].freq = madx.sequence['ps'].beam.freq0*harmonic_nb

        # Xsuite sequence 
        line[nn].lag = lag  # 0 if below transition
        line[nn].voltage =  V_RF*1e3 # In Xsuite for ions, do not multiply by charge as in MADX
        line[nn].frequency = madx.sequence['ps'].beam.freq0*1e6*harmonic_nb
        
        # Save MADX sequence
        if save_madx_seq:
            madx.command.save(sequence='ps', file='{}/PS_2022_{}_{}{}.seq'.format(sequence_path, 
                                                                                  self.ion_type, 
                                                                                  self.seq_name,
                                                                                  extraction_str), beam=True)  
        # Save Xsuite sequence
        if save_xsuite_seq:
            with open('{}/PS_2022_{}_{}{}.json'.format(sequence_path, self.ion_type, 
                                                       self.seq_name, extraction_str), 'w') as fid:
                json.dump(line.to_dict(), fid, cls=xo.JEncoder)
                
        if return_xsuite_line:
            print('\nReturned Xsuite line\n')
            return line
        
        
    def generate_xsuite_seq_with_beta_beat(self, beta_beat=0.02,
                                           save_xsuite_seq=False, line=None):
        """
        Generate Xsuite line with desired beta beat, optimizer quadrupole errors finds
        quadrupole error in first slice of last PS quadrupole to emulate desired beta_beat
        
        Parameters:
        -----------
        beta_beat - desired beta beat, i.e. relative difference between max beta function and max original beta function
        save_xsuite_seq - flag to save xsuite sequence in desired location
        line - can provide generated line, otherwise generate new
        
        Returns:
        -------
        line - xsuite line for tracking
        """
        self._line = self.generate_xsuite_seq(return_xsuite_line=True) if line is None else line
        self._line0 = self._line.copy()
        self._twiss0 = self._line0.twiss()
        
        # Introduce beta to arbitrary QD 
        print('\nFinding optimal QD error for chosen beta-beat {}...\n'.format(beta_beat))
        
        # Initial guess: small perturbation to initial value, then minimize loss function
        dqd0 = self._line0['pr.qdn96..1'].knl[1] + 1e-5
        result = minimize(self._loss_function_beta_beat, dqd0, args=(beta_beat), 
                          method='nelder-mead', tol=1e-5, options={'maxiter':100})
        print(result)
        self._line['pr.qdn96..1'].knl[1] = result.x[0] 
        twiss2 = self._line.twiss()
        
        # Compare difference in Twiss
        print('\nOptimization terminated:')
        print('Twiss max betx difference: {:.3f} vs {:.3f} with QD error'.format(np.max(self._twiss0['betx']),
                                                                                        np.max(twiss2['betx'])))
        print('Twiss max bety difference: {:.3f} vs {:.3f} with QD error'.format(np.max(self._twiss0['bety']),
                                                                                        np.max(twiss2['bety'])))

        # Show beta-beat 
        print('\nX beta-beat: {:.5f}'.format( (np.max(twiss2['betx']) - np.max(self._twiss0['betx']))/np.max(self._twiss0['betx']) ))
        print('Y beta-beat: {:.5f}'.format( (np.max(twiss2['bety']) - np.max(self._twiss0['bety']))/np.max(self._twiss0['bety']) ))
        print('Generated sequence with Qx, Qy = ({}, {}) and beta-beat = {}!\n'.format(self.qx0, self.qy0, beta_beat))
        
        # Save Xsuite sequence
        if save_xsuite_seq:
            with open('{}/PS_2022_{}_{}_percent_beta_beat.json'.format(self.seq_folder, 
                                                                           self.ion_type,
                                                                           int(beta_beat*100)), 'w') as fid:
                json.dump(self._line.to_dict(), fid, cls=xo.JEncoder)
                
        return self._line              


    def _loss_function_beta_beat(self, dqd, beta_beat):
        """
        Loss function to optimize to find correct beta-beat - adapt for horizontal plane
        
        Parameters:
        ----------
        dqd - quadrupolar strength for first slice of last PS QD quadrupole
        beta_beat - beta beat, i.e. relative difference between max beta function and max original beta function
        
        Returns:
        --------
        loss - loss function value, np.abs(beta_beat - desired beta beat)
        """
        
        
        # Try with new quadrupole error, otherwise return high value (square of error)
        try:
            # Copy the line, adjust QD strength of last sliced quadrupole by dqd 
            self._line['pr.qdn96..1'].knl[1] = dqd
            
            twiss2 = self._line.twiss()
        
            # Horizontal plane beta-beat
            X_beat = (np.max(twiss2['betx']) - np.max(self._twiss0['betx']))/np.max(self._twiss0['betx'])
            print('Setting QD error to {:.3e} with Xbeat {:.3e}'.format(dqd[0], X_beat))
            
            loss = np.abs(beta_beat - X_beat)
        except ValueError:
            loss = dqd**2 
            self._line = self._line0.copy()
            print('Resetting line...')
        
        return loss         
