"""
Main module for sequence generator container classes for SPS 
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

optics =  Path(__file__).resolve().parent.joinpath('../data/acc-models-sps').absolute()
sequence_path = Path(__file__).resolve().parent.joinpath('../data/sps_sequences').absolute()
error_file_path = Path(__file__).resolve().parent.joinpath('../data/sps_sequences/magnet_errors').absolute()

@dataclass
class BeamParameters_SPS:
    """Data Container for SPS Pb default beam parameters"""
    Nb:  float = 2.2e8 #3.5e8
    sigma_z: float = 0.225
    exn: float = 1.3e-6
    eyn: float = 0.9e-6
    Qx_int: float = 26.
    Qy_int: float = 26.

@dataclass
class SPS_sequence_maker:
    """ 
    Data class to generate Xsuite line from SPS optics repo, selecting
    - qx0, qy0: horizontal and vertical tunes
    - dq1, dq2: X and Y chroma values 
    - Q_PS: ion charge state in PS
    - Q_SPS: ion charge state in SPS 
    - m_ion: ion mass in atomic units
    - Brho: magnetic rigidity in T*m at injection
    - optics: absolute path to optics repository -> cloned from https://gitlab.cern.ch/acc-models
    """
    qx0: float = 26.30
    qy0: float = 26.19
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
    
    def load_xsuite_line_and_twiss(self, Qy_frac=19, beta_beat=None):
        """
        Method to load pre-generated SPS lattice files for Xsuite
        
        Parameters:
        -----------
        Qy_fractional - fractional vertical tune. "19"" means fractional tune Qy = 0.19
        beta_beat - relative beta beat, i.e. relative difference between max beta function and max original beta function
        
        Returns:
        -------
        xsuite line
        twiss - twiss table from xtrack 
        """
        # Update fractional vertical tune 
        self.qy0 = int(self.qy0) + Qy_frac / 100 
        print('\nTrying to load sequence with Qx, Qy = ({}, {}) and beta-beat = {}!\n'.format(self.qx0, self.qy0, beta_beat))
        
        # Check if pre-generated sequence exists 
        if beta_beat is None or beta_beat == 0.0:
            sps_fname = '{}/qy_dot{}/SPS_2021_{}_nominal.json'.format(sequence_path, Qy_frac, self.ion_type)
        else:                                                  
            sps_fname = '{}/qy_dot{}/SPS_2021_{}_{}_percent_beta_beat.json'.format(sequence_path, Qy_frac, self.ion_type, int(beta_beat*100))
            
        try:
            sps_line = xt.Line.from_json(sps_fname)
        except FileNotFoundError:
            print('\nPre-made SPS sequence does not exists, generating new sequence with Qx, Qy = ({}, {}) and beta-beat = {}!\n'.format(self.qx0, 
                                                                                                                                         self.qy0, 
                                                                                                                                         beta_beat))
            sps_line = self.generate_xsuite_seq() if beta_beat is None else self.generate_xsuite_seq_with_beta_beat(beta_beat)
            
        # Build tracker and Twiss
        #sps_line.build_tracker()  # tracker is already built when loading
        twiss_sps = sps_line.twiss() 
        
        return sps_line, twiss_sps


    @staticmethod
    def generate_SPS_gaussian_beam(line, n_part, exn=None, eyn=None, Nb=None, sigma_z=None):
        """ 
        Class method to generate matched Gaussian beam for SPS. Can provide custom parameters, otherwise default values
        
        Parameters:
        -----------
        line - xtrack line object
        n_part - number of macroparticles
        exn, eyn: horizontal and vertical normalized emittances
        Nb: bunch intensity, number of ions per bunch
        sigma_z: bunch length in meters 
        
        Returns:
        --------
        particles - xpart particles object 
        
        """
        # If no values are provided, use default values
        n_emitt_x = BeamParameters_SPS.exn if exn is None else exn
        n_emitt_y = BeamParameters_SPS.exn if eyn is None else eyn
        Nb_SPS = BeamParameters_SPS.Nb if Nb is None else Nb
        sigma_z_SPS = BeamParameters_SPS.sigma_z if sigma_z is None else sigma_z
        
        particles = xp.generate_matched_gaussian_bunch(
                                 num_particles=n_part, total_intensity_particles = Nb_SPS,
                                 nemitt_x = n_emitt_x, nemitt_y = n_emitt_y, 
                                 sigma_z = sigma_z_SPS,
                                 particle_ref = line.particle_ref, line=line)
        print('\nGenerated Gaussian beam') 
        print('with Nb = {:.3e}, exn = {:.4e}, \neyn = {:.4e}, sigma_z = {:.3e} m \n{} macroparticles\n'.format(BeamParameters_SPS.Nb,
                                                                                                                BeamParameters_SPS.exn,
                                                                                                                BeamParameters_SPS.eyn,
                                                                                                                BeamParameters_SPS.sigma_z,
                                                                                                                n_part))
        return particles


    def generate_SPS_beam(self):
        """
        Generate correct injection parameters for SPS beam
        
        Returns:
        -------
        ion rest mass in eV, beam momentum in eV/c at SPS injection 
        """
        # Calculate Brho at SPS injection
        rho_PS = 70.1206 # [m] - PS bending radius 
        self._Brho_PS_extr = self.B_PS_extr * rho_PS
        
        #print('\nCreating MADX-beam of {}\n'.format(self.ion_type))
        m_in_eV = self.m_ion * constants.physical_constants['atomic mass unit-electron volt relationship'][0]   # 1 Dalton in eV/c^2 -- atomic mass unit
        p_inj_SPS = 1e9 * (self._Brho_PS_extr * self.Q_PS) / 3.3356 # in  [eV/c], if q is number of elementary charges

        return m_in_eV, p_inj_SPS


    def generate_xsuite_seq(self, save_madx_seq=False, 
                            save_xsuite_seq=False, 
                            return_xsuite_line=True, voltage=3.0e6):
        """
        Load MADX line, match tunes and chroma, add RF and generate Xsuite line
        
        Parameters:
        -----------
        save_madx_seq - save madx sequence to directory 
        save_xsuite_seq - save xtrack sequence to directory  
        return_xsuite_line - return generated xtrack line
        voltage - RF voltage
        
        Returns:
        --------
        None
        """
        os.makedirs(self.seq_folder, exist_ok=True)
        print('\nGenerating sequence for {} with qx = {}, qy = {}\n'.format(self.ion_type, self.qx0, self.qy0))
        
        #### Initiate MADX sequence and call the sequence and optics file ####
        madx = Madx()
        madx.call("{}/sps.seq".format(optics))
        madx.call("{}/strengths/lhc_ion.str".format(optics))
        madx.call("{}/beams/beam_lhc_ion_injection.madx".format(optics))  # attach beam just in case, otherwise error table will be empty
        
        # Generate SPS beam - use default Pb or make custom beam
        m_in_eV, p_inj_SPS = self.generate_SPS_beam()
        
        #madx.call("{}/beams/beam_lhc_ion_injection.madx".format(optics))
        madx.input(" \
                   Beam, particle=ion, mass={}, charge={}, pc = {}, sequence='sps'; \
                   DPP:=BEAM->SIGE*(BEAM->ENERGY/BEAM->PC)^2;  \
                   ".format(m_in_eV/1e9, self.Q_SPS, p_inj_SPS/1e9))   # convert mass to GeV/c^2

        # Flatten line
        madx.use(sequence='sps')
        madx.input("seqedit, sequence=SPS;")
        madx.input("flatten;")
        madx.input("endedit;")
        madx.use("sps")
        madx.input("select, flag=makethin, slice=5, thick=false;")
        madx.input("makethin, sequence=sps, style=teapot, makedipedge=True;")

        # Use correct tune and chromaticity matching macros
        madx.call("{}/toolkit/macro.madx".format(optics))
        madx.use('sps')
        madx.exec(f"sps_match_tunes({self.qx0}, {self.qy0});")
        madx.exec("sps_define_sext_knobs();")
        madx.exec("sps_set_chroma_weights_q26();")
        madx.input(f"""match;
        global, dq1={self.dq1};
        global, dq2={self.dq2};
        vary, name=qph_setvalue;
        vary, name=qpv_setvalue;
        jacobian, calls=10, tolerance=1e-25;
        endmatch;""")
        
        # Create Xsuite line, check that Twiss command works 
        madx.use(sequence='sps')
        twiss_thin = madx.twiss()  
        
        line = xt.Line.from_madx_sequence(madx.sequence['sps'])
        line.build_tracker()
        #madx_beam = madx.sequence['sps'].beam
        
        self.particle_sample = xp.Particles(
                p0c = p_inj_SPS,
                q0 = self.Q_SPS,
                mass0 = m_in_eV)
        print('\nGenerated SPS {} beam with gamma = {:.3f}, Qx = {:.3f}, Qy = {:.3f}\n'.format(self.ion_type, 
                                                                                              self.particle_sample.gamma0[0],
                                                                                              self.qx0,
                                                                                              self.qy0))
        
        line.particle_ref = self.particle_sample
        
        #### SET CAVITY VOLTAGE - with info from Hannes
        # 6x200 MHz cavities: actcse, actcsf, actcsh, actcsi (3 modules), actcsg, actcsj (4 modules)
        # acl 800 MHz cavities
        # acfca crab cavities
        # Ions: all 200 MHz cavities: 1.7 MV, h=4653
        harmonic_nb = 4653
        nn = 'actcse.31632'
        
        # MADX sequence 
        madx.sequence.sps.elements[nn].lag = 0
        madx.sequence.sps.elements[nn].volt = 3.0*self.particle_sample.q0 # different convention between madx and xsuite
        madx.sequence.sps.elements[nn].freq = madx.sequence['sps'].beam.freq0*harmonic_nb
        
        # Xsuite sequence 
        line[nn].lag = 0  # 0 if below transition
        line[nn].voltage = voltage # In Xsuite for ions, do not multiply by charge as in MADX
        line[nn].frequency = madx.sequence['sps'].beam.freq0*1e6*harmonic_nb
        
        print('\nSet RF voltage to {:.3e} V\n'.format(voltage))
        
        # Save MADX sequence
        if save_madx_seq:
            madx.command.save(sequence='sps', file='{}/SPS_2021_{}_{}.seq'.format(self.seq_folder, 
                                                                                  self.ion_type, 
                                                                                  self.seq_name), beam=True)  
        # Save Xsuite sequence
        if save_xsuite_seq:
            with open('{}/SPS_2021_{}_{}.json'.format(self.seq_folder, self.ion_type, self.seq_name), 'w') as fid:
                json.dump(line.to_dict(), fid, cls=xo.JEncoder)
                
        if return_xsuite_line:
            return line


    def generate_xsuite_seq_with_magnet_errors(self, save_madx_seq=False, 
                            save_xsuite_seq=False, 
                            return_xsuite_line=True, 
                            add_non_linear_magnet_errors=False):
        """
        Loads default SPS Pb sequences, adds non-linear magnet errors (from Xavier buffat)
        
        Parameters:
        -----------
        save_madx_seq - save madx sequence to directory 
        save_xsuite_seq - save xtrack sequence to directory  
        return_xsuite_line - return generated xtrack line
        
        Returns:
        --------
        None
        """
        madx = Madx()
        madx.call("{}/madx/SPS_2021_Pb_nominal.seq".format(sequence_path))
        
        # Add the extra magnet errors
        madx.use(sequence='sps')
        madx.call('{}/sps_setMultipoles_upto7.cmd'.format(error_file_path))
        madx.input('exec, set_Multipoles_26GeV;')
        madx.call('{}/sps_assignMultipoles_upto7.cmd'.format(error_file_path))
        madx.input('exec, AssignMultipoles;')
        errtab_ions = madx.table.errtab
        
        self.seq_name += '_with_magnet_errors'
        
        # Generate SPS beam - use default Pb or make custom beam
        m_in_eV, p_inj_SPS = self.generate_SPS_beam()
        
        self.particle_sample = xp.Particles(
                p0c = p_inj_SPS,
                q0 = self.Q_SPS,
                mass0 = m_in_eV)
        print('\nGenerated SPS {} beam with gamma = {:.3f}, Qx = {:.3f}, Qy = {:.3f}\n'.format(self.ion_type, 
                                                                                              self.particle_sample.gamma0[0],
                                                                                              self.qx0,
                                                                                              self.qy0))
        madx.input(" \
                   Beam, particle=ion, mass={}, charge={}, pc = {}, sequence='sps'; \
                   DPP:=BEAM->SIGE*(BEAM->ENERGY/BEAM->PC)^2;  \
                   ".format(m_in_eV/1e9, self.Q_SPS, p_inj_SPS/1e9))   # convert mass to GeV/c^2

        # Flatten line
        madx.use(sequence='sps')
        madx.input("seqedit, sequence=SPS;")
        madx.input("flatten;")
        madx.input("endedit;")
        madx.use("sps")
        madx.input("select, flag=makethin, slice=5, thick=false;")
        madx.input("makethin, sequence=sps, style=teapot, makedipedge=True;")

        # Use correct tune and chromaticity matching macros
        madx.call("{}/toolkit/macro.madx".format(optics))
        madx.use('sps')
        madx.exec(f"sps_match_tunes({self.qx0}, {self.qy0});")
        madx.exec("sps_define_sext_knobs();")
        madx.exec("sps_set_chroma_weights_q26();")
        madx.input(f"""match;
        global, dq1={self.dq1};
        global, dq2={self.dq2};
        vary, name=qph_setvalue;
        vary, name=qpv_setvalue;
        jacobian, calls=10, tolerance=1e-25;
        endmatch;""")
        
        # Create Xsuite line, check that Twiss command works 
        madx.use(sequence='sps')
        twiss_thin = madx.twiss()  
        
        line = xt.Line.from_madx_sequence(madx.sequence['sps'])
        line.particle_ref = self.particle_sample
        line.build_tracker()
        
        # Save MADX sequence
        if save_madx_seq:
            madx.command.save(sequence='sps', file='{}/SPS_2021_{}_{}.seq'.format(self.seq_folder, 
                                                                                  self.ion_type, 
                                                                                  self.seq_name), beam=True)  
        # Save Xsuite sequence
        if save_xsuite_seq:
            with open('{}/SPS_2021_{}_{}.json'.format(self.seq_folder, self.ion_type, self.seq_name), 'w') as fid:
                json.dump(line.to_dict(), fid, cls=xo.JEncoder)
                
        if return_xsuite_line:
            return line


    def generate_xsuite_seq_with_beta_beat(self, beta_beat=0.05,
                                           save_xsuite_seq=False, line=None,
                                           add_non_linear_magnet_errors=False
                                           ):
        """
        Generate Xsuite line with desired beta beat, optimizer quadrupole errors finds
        quadrupole error in first slice of last SPS quadrupole to emulate desired beta_beat
        
        Parameters:
        -----------
        beta_beat - desired beta beat, i.e. relative difference between max beta function and max original beta function
        save_xsuite_seq - flag to save xsuite sequence in desired location
        line - can provide generated line, otherwise generate new
        add_non_linear_magnet_errors - add errors from non-linear chromaticity if desired 
        
        Returns:
        -------
        line - xsuite line for tracking
        """
        # If non-linear errors are used, use twiss0 from reference line without errors  
        if add_non_linear_magnet_errors:
            self._line = self.generate_xsuite_seq_with_magnet_errors() if line is None else line
            self._line0 = self.generate_xsuite_seq() if line is None else line
        else:
            self._line = self.generate_xsuite_seq() if line is None else line
            self._line0 = self._line.copy()
        self._twiss0 = self._line0.twiss()
        
        # Introduce beta to arbitrary QD 
        print('\nFinding optimal QD error for chosen beta-beat {}...\n'.format(beta_beat))
        
        # Initial guess: small perturbation to initial value, then minimize loss function
        dqd0 = self._line0['qd.63510..1'].knl[1] + 1e-5
        result = minimize(self._loss_function_beta_beat, dqd0, args=(beta_beat), 
                          method='nelder-mead', tol=1e-5, options={'maxiter':100})
        print(result)
        self._line['qd.63510..1'].knl[1] = result.x[0] 
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
            with open('{}/SPS_2021_{}_{}_percent_beta_beat.json'.format(self.seq_folder, 
                                                                           self.ion_type,
                                                                           int(beta_beat*100)), 'w') as fid:
                json.dump(self._line.to_dict(), fid, cls=xo.JEncoder)
                
        return self._line              


    def _loss_function_beta_beat(self, dqd, beta_beat):
        """
        Loss function to optimize to find correct beta-beat
        
        Parameters:
        ----------
        dqd - quadrupolar strength for first slice of last SPS quadrupoles
        beta_beat - beta beat, i.e. relative difference between max beta function and max original beta function
        
        Returns:
        --------
        loss - loss function value, np.abs(beta_beat - desired beta beat)
        """
        
        
        # Try with new quadrupole error, otherwise return high value (square of error)
        try:
            # Copy the line, adjust QD strength of last sliced quadrupole by dqd 
            self._line['qd.63510..1'].knl[1] = dqd
            
            twiss2 = self._line.twiss()
        
            # Vertical plane beta-beat
            Y_beat = (np.max(twiss2['bety']) - np.max(self._twiss0['bety']))/np.max(self._twiss0['bety'])
            print('Setting QD error to {:.3e} with Ybeat {:.3e}'.format(dqd[0], Y_beat))
            
            loss = np.abs(beta_beat - Y_beat)
        except ValueError:
            loss = dqd**2 
            self._line = self._line0.copy()
            print('Resetting line...')
        
        return loss 