"""
Main module for sequence generator container classes for SPS 
"""
import numpy as np
import pandas as pd
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

from ..beam_parameters import BeamParameters_SPS, BeamParameters_SPS_Oxygen, BeamParameters_SPS_Proton


optics =  Path(__file__).resolve().parent.joinpath('../../data/acc-models-sps').absolute()
sequence_path = Path(__file__).resolve().parent.joinpath('../../data/sps_sequences').absolute()
error_file_path = Path(__file__).resolve().parent.joinpath('../../data/sps_sequences/magnet_errors').absolute()
aperture_fixed_path = Path(__file__).resolve().parent.joinpath('../../data/aperture_fixed').absolute()

@dataclass
class SPS_sequence_maker:
    """ 
    Data class to generate Xsuite line from SPS optics repo, selecting
    
    Parameters:
    -----------
    qx0, qy0: float
        horizontal and vertical tunes
    dq1, dq2: float
        X and Y chroma values 
    Q_PS: int
        ion charge state in PS
    Q_SPS: int
        ion charge state in SPS 
    m_ion: float
        ion mass in atomic units
    Brho: float
        magnetic rigidity in T*m at injection
    optics: str 
        absolute path to optics repository -> cloned from https://gitlab.cern.ch/acc-models
    """
    ion_type: str = 'Pb' # Define beam type - default is Pb
    qx0: float = 26.30
    qy0: float = 26.19
    dq1: float = -0.367 if ion_type == 'Pb' else 0.174 # ion values measured 2024-11-19, proton vals: knobs + Ingrid
    dq2: float = -0.449 if ion_type == 'Pb' else 0.086 # ion values measured 2024-11-19, proton vals: knobs + Ingrid 
    seq_name: str = 'nominal'
    B_PS_extr: float = 1.2368 # [T] - magnetic field in PS for Pb ions, from Heiko Damerau
    rho_PS: float = 70.1206 # [m] - PS bending radius 
    Q_PS: float = 54.
    Q_SPS: float = 82.
    m_ion: float = 207.98 # atomic units
    proton_optics : str = 'q26'

    def __post_init__(self):
        
        self.seq_folder = '{}/{}_qy_dot_{}'.format(sequence_path, self.proton_optics, int(self.qy0 % 1 * 100))
        # Check that proton charge is be correct
        if self.ion_type == 'proton':
            self.Q_PS, self.Q_SPS = 1., 1.
            

    def load_default_twiss_table(self, cycled_to_minimum_dx=True, add_beta_beat=True):
        """
        Return pandas dataframe with twiss table of default SPS sequence. Create json if does not exist already
        """
        string = '_min_dx' if cycled_to_minimum_dx else ''
        bb_string = '_with_beta_beat' if add_beta_beat else ''
        try:
            df_twiss = pd.read_json('{}/twiss_sps_pandas{}{}.json'.format(sequence_path, string, bb_string))
            print('\nLoaded twiss table - cycled_to_minimum_dx = {}, beta_beat = {}\n'.format(cycled_to_minimum_dx, add_beta_beat)) 
        except FileNotFoundError:
            line = self.generate_xsuite_seq(add_aperture=True)
            
            if add_beta_beat:
                line.element_refs['qd.63510..1'].knl[1] = -1.07328640311457e-02
                line.element_refs['qf.63410..1'].knl[1] = 1.08678014669101e-02
                print('Beta-beat added: kk_QD = {:.6e}, kk_QF = {:.6e}'.format(line.element_refs['qd.63510..1'].knl[1]._value,
                                                                               line.element_refs['qf.63410..1'].knl[1]._value))
            
            df_twiss = line.twiss().to_pandas()
            df_twiss.to_json('{}/twiss_sps_pandas{}{}.json'.format(sequence_path, string, bb_string))
            print('\nFailed to load Twiss dataframe, generating new and saved to {}/twiss_sps_pandas{}{}.json'.format(sequence_path, string, bb_string))
        return df_twiss

    def load_xsuite_line_and_twiss(self,
                                   beta_beat=None, 
                                   use_symmetric_lattice=False,
                                   add_non_linear_magnet_errors=False, 
                                   save_new_xtrack_line=False,
                                   deferred_expressions=True, 
                                   add_aperture=False, 
                                   plane='both',
                                   voltage=3.0e6):
        """
        Method to load pre-generated SPS lattice files for Xsuite, or generate new if does not exist
        
        Parameters:
        -----------
        beta_beat : float
            relative beta beat, i.e. relative difference between max beta function and max original beta function
        use_symmetric_lattice : bool
            flag to use symmetric lattice without QFA and QDA
        save_new_xtrack_line : bool
            if new sequence is created, save it for future use
        deferred_expressions : bool
            whether to use deferred expressions while importing madx sequence into xsuite
        add_aperture : bool
            whether to include aperture for SPS
        plane : str
            if loading line with beta-beat, specify in which plane beat is taking place: 'X', 'Y' or 'both'
        voltage : float
            RF voltage in V
        
        Returns:
        -------
        xsuite line
        twiss - twiss table from xtrack 
        """        
        # Check if proton or ion
        if self.ion_type=='proton':
            use_Pb_ions = False
        else:
            use_Pb_ions = True

        # Substrings to identify line
        symmetric_string = '_symmetric' if use_symmetric_lattice else '_nominal'
        err_str = '_with_non_linear_chrom_error' if add_non_linear_magnet_errors else ''
        def_exp_str = '_deferred_exp' if deferred_expressions else ''
        aperture_str = '_with_aperture' if add_aperture else ''
        if self.proton_optics == 'q20':
            proton_optics_str = '_q20_optics' 
        elif self.proton_optics == 'q26':
            proton_optics_str = ''
        else:
            raise ValueError('Invalid optics: select Q20 or Q26')
        
        # Update sequence folder location
        self.seq_folder = '{}/{}_qy_dot_{}'.format(sequence_path, self.proton_optics, int(self.qy0 % 1 * 100))
        os.makedirs(self.seq_folder, exist_ok=True)
        print('\nTrying to load sequence with Qx, Qy = ({}, {}) and beta-beat = {}!\n'.format(self.qx0, self.qy0, beta_beat))
        
        if use_symmetric_lattice:
            print('\nLoading symmetric SPS lattice\n')
        if add_non_linear_magnet_errors:
            print('\nLoading lattic with non-linear magnet errors\n')
        if add_aperture:
            print('\nLoading lattice with aperture\n')
                
        # Check if pre-generated sequence exists 
        if beta_beat is None or beta_beat == 0.0:
            sps_fname = '{}/SPS_2021_{}{}{}{}{}{}.json'.format(self.seq_folder, self.ion_type, symmetric_string,
                                                                          def_exp_str, err_str, aperture_str, proton_optics_str)
        else:                                                  
            sps_fname = '{}/SPS_2021_{}{}{}_{}plane_{}_percent_beta_beat{}{}{}.json'.format(self.seq_folder, self.ion_type, 
                                                                                     symmetric_string, def_exp_str, plane, 
                                                                                     int(beta_beat*100), err_str, aperture_str, 
                                                                                     proton_optics_str)
            
        # Try to load pre-generated sequence if exists
        try:
            sps_line = xt.Line.from_json(sps_fname)
            print('\nSuccessfully loaded {}\n'.format(sps_fname))

            # If loaded, set RF voltage to correct value
            nn = 'actcse.31632' if use_Pb_ions else 'actcse.31637'
            sps_line[nn].voltage = voltage # In Xsuite for ions, do not multiply by charge as in MADX    
            print('RF voltage set to {:.3e} V\n'.format(voltage))

        except FileNotFoundError:
            print('\nPre-made SPS sequence does not exists, generating new sequence with Qx, Qy = ({}, {}), beta-beat = {} \
                  and non-linear error={} and aperture= {} and optics = {}\n{}'.format(self.qx0, 
                                                                        self.qy0, 
                                                                        beta_beat,
                                                                        add_non_linear_magnet_errors,
                                                                        add_aperture,
                                                                        self.proton_optics,
                                                                        sps_fname))
            # Make new line with beta-beat and/or non-linear chromatic errors
            if beta_beat is None:
                sps_line = self.generate_xsuite_seq(use_symmetric_lattice=use_symmetric_lattice, 
                                                    deferred_expressions=deferred_expressions,
                                                    add_non_linear_magnet_errors=add_non_linear_magnet_errors, 
                                                    add_aperture=add_aperture, voltage=voltage) 
            else:
                sps_line = self.generate_xsuite_seq_with_beta_beat(beta_beat=beta_beat, use_symmetric_lattice=use_symmetric_lattice, 
                                                                   add_non_linear_magnet_errors=add_non_linear_magnet_errors, add_aperture=add_aperture,
                                                                   plane=plane, voltage=voltage)
                
            # Save new Xsuite sequence if desired
            if save_new_xtrack_line:
                with open(sps_fname, 'w') as fid:
                    json.dump(sps_line.to_dict(), fid, cls=xo.JEncoder)
                print('\nSaved new xtrack line {}\n'.format(sps_fname))
            
        # Build tracker and Twiss
        #sps_line.build_tracker()  # tracker is already built when loading
        twiss_sps = sps_line.twiss() 
        
        return sps_line, twiss_sps


    def generate_xsuite_seq(self,
                            save_madx_seq=False, 
                            save_xsuite_seq=False, 
                            return_xsuite_line=True, 
                            voltage=3.0e6,
                            use_symmetric_lattice=False,
                            add_non_linear_magnet_errors=False,
                            deferred_expressions=True,
                            add_aperture=False,
                            nr_slices=5):
        """
        Load MADX line, match tunes and chroma, add RF and generate Xsuite line
        
        Parameters:
        -----------
        save_madx_seq : bool
            whether save madx sequence to directory 
        save_xsuite_seq : bool
            whether to save xtrack sequence to directory  
        return_xsuite_line : bool
            wether to return generated xtrack line
        voltage : float
            RF voltage in V
        add_non_linear_magnet_errors : bool
            whether to add line with non-linear chromatic errors
        deferred_expressions : bool
            whether to use deferred expressions while importing madx sequence into xsuite    
        add_aperture : bool
            whether to include aperture for SPS
        nr_slices : int
            number of slices when slicing MADX sequence from thick to thin
        
        Returns:
        --------
        None
        """
        # Check if proton or ion
        if self.ion_type=='proton':
            use_Pb_ions = False
        else:
            use_Pb_ions = True

        # Define optics
        if self.proton_optics == 'q20':
            proton_optics_str = '_q20_optics' 
        elif self.proton_optics == 'q26':
            proton_optics_str = ''
        else:
            raise ValueError('Invalid optics: select Q20 or Q26')

        # Update sequence folder location
        os.makedirs(self.seq_folder, exist_ok=True)
        print('\nGenerating sequence for {} with qx = {}, qy = {}\n'.format(self.ion_type, self.qx0, self.qy0))
        
        # Substrings to identify line
        symmetric_string = '_symmetric' if use_symmetric_lattice else '_nominal'
        err_str = '_with_non_linear_chrom_error' if add_non_linear_magnet_errors else ''
        def_exp_str = '_deferred_exp' if deferred_expressions else ''
        aperture_str = '_with_aperture' if add_aperture else ''
        
        # Load madx instance with SPS sequence
        madx = self.load_simple_madx_seq(add_non_linear_magnet_errors=add_non_linear_magnet_errors, 
                                         add_aperture=add_aperture,
                                         nr_slices=nr_slices)
                
        line = xt.Line.from_madx_sequence(madx.sequence['sps'], deferred_expressions=deferred_expressions,
                                          install_apertures=add_aperture, apply_madx_errors=add_non_linear_magnet_errors)
        line.build_tracker()
        #madx_beam = madx.sequence['sps'].beam
        
        # Generate SPS beam - use default Pb or make custom beam
        self.m_in_eV, self.p_inj_SPS = self.generate_SPS_beam()
        
        self.particle_sample = xp.Particles(
                p0c = self.p_inj_SPS,
                q0 = self.Q_SPS,
                mass0 = self.m_in_eV)
        print('\nGenerated SPS {} beam p = {:.4f}, gamma = {:.3f}, Qx = {:.3f}, Qy = {:.3f}\n'.format(self.ion_type,
                                                                                                      self.p_inj_SPS * 1e-9, 
                                                                                                      self.particle_sample.gamma0[0],
                                                                                                      self.qx0,
                                                                                                      self.qy0))
        
        line.particle_ref = self.particle_sample
        
        ############## ADD RF VOLTAGE FOR LONGITUDINAL - DIFFERENT FOR MADX AND XSUITE ##############
        
        #### SET CAVITY VOLTAGE - with info from Hannes
        # 6x200 MHz cavities: actcse, actcsf, actcsh, actcsi (3 modules), actcsg, actcsj (4 modules)
        # acl 800 MHz cavities
        # acfca crab cavities
        # Ions: all 200 MHz cavities: 1.7 MV, h=4653
        harmonic_nb = 4653 if use_Pb_ions else 4620
        nn = 'actcse.31632' if use_Pb_ions else 'actcse.31637'
        
        # MADX sequence 
        # different convention between madx and xsuite - MADX uses MV, and requires multiplication by charge
        madx.sequence.sps.elements[nn].lag = 0 if use_Pb_ions else 180 # above transition if protons
        madx.sequence.sps.elements[nn].volt = (voltage/1e6)*self.particle_sample.q0 
        
        # Xsuite sequence 
        line[nn].lag = 0  if use_Pb_ions else 180 # above transition if protons
        line[nn].voltage = voltage # In Xsuite for ions, do not multiply by charge as in MADX
        line[nn].frequency = madx.sequence['sps'].beam.freq0*1e6*harmonic_nb
        
        print('Set RF voltage to {:.3e} V\n'.format(voltage))
        
        # Save MADX sequence
        if save_madx_seq:
            madx.command.save(sequence='sps', file='{}/SPS_2021_{}_{}{}.seq'.format(self.seq_folder, 
                                                                                  self.ion_type, 
                                                                                  self.seq_name,
                                                                                  err_str), beam=True)  
        # Save Xsuite sequence
        if save_xsuite_seq:
            sps_fname = '{}/SPS_2021_{}{}{}{}{}{}.json'.format(self.seq_folder, self.ion_type, symmetric_string,
                                                                      def_exp_str, err_str, proton_optics_str, 
                                                                      aperture_str)
            with open(sps_fname, 'w') as fid:
                json.dump(line.to_dict(), fid, cls=xo.JEncoder)
                
        if return_xsuite_line:
            return line
    

    def add_beta_beat_to_line(self, line : xt.Line)->xt.Line:
        """Add measured Q26 RMS beta-beat to line with kqd and kqf knobs"""

        line.element_refs['qd.63510..1'].knl[1] = -1.07328640311457e-02
        line.element_refs['qf.63410..1'].knl[1] = 1.08678014669101e-02
        print('Beta-beat added: kk_QD = {:.6e}, kk_QF = {:.6e}'.format(line.element_refs['qd.63510..1'].knl[1]._value,
                                                                        line.element_refs['qf.63410..1'].knl[1]._value))
        return line
    
    
    def change_synchrotron_tune_by_factor(self, A, line, sigma_z=None, Nb=None):
        """
        Scale synchrotron tune Qs while keeping bucket half-height delta constant, also adjusting
        bunch length and bunch intensity accordingly for identical bunch filling factor and space charge effects

        Parameters
        ----------
        line : xtrack.Line
            line used in tracking
        A : float
            factor by which to scale the synchrotron tune
        sigma_z : float
            original bunch length
        Nb : float
            original bunch intensity

        Returns
        -------
        line_new : xtrack.Line
            line with updated RF voltage and harmonic
        sigma_z_new : float
            updated new bunch length
        Nb_new : float
            updated new bunch intensity
        """
        # Check if proton or ion
        if self.ion_type=='proton':
            use_Pb_ions = False
        else:
            use_Pb_ions = True
            
        # Provide default bunch length values if nothing given
        beamParams = BeamParameters_SPS_Proton() if self.ion_type=='proton' else BeamParameters_SPS()
        if sigma_z is None:
            sigma_z = beamParams.sigma_z
        if Nb is None:
            Nb = beamParams.Nb
        
        # Find RF cavity number 
        nn = 'actcse.31632' if use_Pb_ions else 'actcse.31637'
        
        line[nn].voltage *= A # scale voltage by desired factor
        line[nn].frequency *= A # in reality scale harmonic number, but translates directly to frequency
        sigma_z_new = sigma_z / A
        Nb_new = Nb / A
        
        print('\nScaling RF voltage to {:.3e} V with factor {:.3f}'.format(line[nn].voltage, A))
        
        return line, sigma_z_new, Nb_new
    

    def load_SPS_line_with_deferred_madx_expressions(self, use_symmetric_lattice=False, Qy_frac=25,
                                                     add_non_linear_magnet_errors=False, add_aperture=False,
                                                     voltage=3.0e6):
        """
        Loads xtrack Pb sequence file with deferred expressions to regulate QD and QF strengths
        or generate from MADX if does not exist
        
        Parameters:
        -----------
        use_symmetric_lattice : bool
            flag to use symmetric lattice without QFA and QDA
        Qy_frac : int
            fractional vertical tune. "19"" means fractional tune Qy = 0.19
        add_non_linear_magnet_errors : bool
            whether to add line with non-linear chromatic errors
        add_aperture : bool
            whether to include aperture for SPS        
        voltage : float
            RF voltage in V

        Returns:
        -------
        xtrack line
        """
        # Check if proton or ion
        if self.ion_type=='proton':
            use_Pb_ions = False
        else:
            use_Pb_ions = True

        # Check optics
        if self.proton_optics == 'q20':
            proton_optics_str = '_q20_optics' 
        elif self.proton_optics == 'q26':
            proton_optics_str = ''
        else:
            raise ValueError('Invalid optics: select Q20 or Q26')

        err_str = '_with_non_linear_chrom_error' if add_non_linear_magnet_errors else ''
        aperture_str = '_with_aperture' if add_aperture else ''
        
        # Create directory if not exists already
        os.makedirs('{}/qy_dot{}'.format(sequence_path, Qy_frac), exist_ok=True)
        
        # Try loading existing json file, otherwise create new from MADX
        if use_symmetric_lattice:
            fname = '{}/qy_dot{}/SPS_2021_{}_symmetric_deferred_exp{}{}{}.json'.format(sequence_path, Qy_frac, self.ion_type, err_str, 
                                                                                       aperture_str, proton_optics_str)
        else:
            fname = '{}/qy_dot{}/SPS_2021_{}_nominal_deferred_exp{}{}{}.json'.format(sequence_path, Qy_frac, self.ion_type, err_str, 
                                                                                     aperture_str, proton_optics_str)
        
        #sps = SPS_sequence_maker()
        madx = self.load_simple_madx_seq(add_non_linear_magnet_errors=add_non_linear_magnet_errors,
                                         add_aperture=add_aperture)

        # Convert to line
        line = xt.Line.from_madx_sequence(madx.sequence['sps'], deferred_expressions=True, install_apertures=add_aperture,
                                          apply_madx_errors=add_non_linear_magnet_errors, 
                                          enable_field_errors=add_non_linear_magnet_errors, enable_align_errors=add_non_linear_magnet_errors)
        m_in_eV, p_inj_SPS = self.generate_SPS_beam()
        
        line.particle_ref = xp.Particles(
                p0c = p_inj_SPS,
                q0 = self.Q_SPS,
                mass0 = m_in_eV)
        line.build_tracker()
        
        ############## ADD RF VOLTAGE FOR LONGITUDINAL - DIFFERENT FOR MADX AND XSUITE ##############
        
        #### SET CAVITY VOLTAGE - with info from Hannes
        # 6x200 MHz cavities: actcse, actcsf, actcsh, actcsi (3 modules), actcsg, actcsj (4 modules)
        # acl 800 MHz cavities
        # acfca crab cavities
        # Ions: all 200 MHz cavities: 1.7 MV, h=4653
        harmonic_nb = 4653 if use_Pb_ions else 4620
        nn = 'actcse.31632' if use_Pb_ions else 'actcse.31637'
        
        # MADX sequence 
        # different convention between madx and xsuite - MADX uses MV, and requires multiplication by charge
        madx.sequence.sps.elements[nn].lag = 0 if use_Pb_ions else 180 # above transition if protons
        madx.sequence.sps.elements[nn].volt = (voltage/1e6)*line.particle_ref.q0 
        
        # Xsuite sequence 
        line[nn].lag = 0  if use_Pb_ions else 180 # above transition if protons
        line[nn].voltage = voltage # In Xsuite for ions, do not multiply by charge as in MADX
        line[nn].frequency = madx.sequence['sps'].beam.freq0*1e6*harmonic_nb
        
        with open(fname, 'w') as fid:
            json.dump(line.to_dict(), fid, cls=xo.JEncoder)
            
        twiss = line.twiss(method='6d')
        
        print('\nGenerated SPS Pb beam with gamma = {:.3f}, Qx = {:.3f}, Qy = {:.3f}\n'.format(line.particle_ref.gamma0[0],
                                                                                              twiss['qx'],
                                                                                              twiss['qy']))
        return line, twiss



    def generate_xsuite_seq_with_beta_beat(self, 
                                           find_beta_beat_from_rms=True,
                                           beta_beat_x=0.05,
                                           beta_beat_y=0.05,
                                           save_xsuite_seq=False, 
                                           line=None,
                                           use_symmetric_lattice=False,
                                           add_non_linear_magnet_errors=False,
                                           add_aperture=False,
                                           voltage=3.0e6,
                                           mask_with_BPMs=False
                                           ):
        """
        Generate Xsuite line with desired beta beat, optimizer finds
        quadrupole error in first slice of last SPS quadrupole to emulate desired beta_beat
        
        Parameters:
        -----------
        find_beta_beat_from_rms : bool
            whether to optimize quadrupolar strengths to find RMS values around the ring. If False, will search for maximum
        beta_beat_x : float
            desired beta beat in x, i.e. relative difference between max beta function and max original beta function.
        beta_beat_y : float
            desired beta beat in x, i.e. relative difference between max beta function and max original beta function. 
        save_xsuite_seq : bool
            flag to save xsuite sequence in desired location
        line : xtrack.Line
            can provide generated line, otherwise generate new
        add_non_linear_magnet_errors : bool
            add errors from non-linear chromaticity if desired 
        add_non_linear_magnet_errors : bool
            whether to add line with non-linear chromatic errors
        add_aperture : bool
            whether to include aperture for SPS
        voltage : float
            RF voltage in V
        mask_with_BPMs: bool
            whether to only measure beta functions at BPM locations, like in the measurements. May cause undesired modulation
        
        Returns:
        -------
        line - xsuite line for tracking
        """
        # Check optics
        if self.proton_optics == 'q20':
            proton_optics_str = '_q20_optics' 
        elif self.proton_optics == 'q26':
            proton_optics_str = ''
        else:
            raise ValueError('Invalid optics: select Q20 or Q26')

        Qy_frac = int(100*(np.round(self.qy0 % 1, 2)))
        self._line, _ = self.load_SPS_line_with_deferred_madx_expressions(use_symmetric_lattice=use_symmetric_lattice, Qy_frac=Qy_frac,
                                                         add_non_linear_magnet_errors=add_non_linear_magnet_errors, add_aperture=add_aperture,
                                                         voltage=voltage)
        
        self._line0 = self._line.copy()
        self._twiss0 = self._line0.twiss()
        
        # Load BPM data and find BPM locations for beta functions
        if mask_with_BPMs:
            bpm_x_names = []
            bpm_y_names = []
            self.bpm_x_ind = []
            self.bpm_y_ind = []
            bpms_h = np.loadtxt('{}/BPMs/common_bpms_H.txt'.format(sequence_path), dtype=str)
            bpms_v = np.loadtxt('{}/BPMs/common_bpms_V.txt'.format(sequence_path), dtype=str)
            
            for i, key in enumerate(self._line.element_names):
            
                # X plane --> check BPM elements
                for bpm_h in bpms_h:
                    if bpm_h[:-2].lower() in key:
                        self.bpm_x_ind.append(i)
                        bpm_x_names.append(key)
    
                # Y plane --> check BPM elements
                for bpm_v in bpms_v:
                    if bpm_v[:-2].lower() in key:
                        self.bpm_y_ind.append(i)
                        bpm_y_names.append(key)
            print('Only use Twiss at BPMs locations: found {} in H and {} in V'.format(len(self.bpm_x_ind), len(self.bpm_y_ind)))
        else:
            # Select all elements
            self.bpm_x_ind = np.arange(len(self._twiss0))
            self.bpm_y_ind = np.arange(len(self._twiss0))

        
        # Initial guess: small perturbation to initial value, then minimize loss function
        print('\nBeta-beat search for X: {} and Y: {}\n'.format(beta_beat_x, beta_beat_y))
        dqd0 = [self._line0['qd.63510..1'].knl[1] + 1e-6, self._line0['qf.63410..1'].knl[1] + 1e-6]
        
        # Search for RMS beta-beat, if desired
        if find_beta_beat_from_rms:
            result = minimize(self._loss_function_beta_beat_RMS, dqd0, args=(beta_beat_x, beta_beat_y), 
                              method='nelder-mead', tol=1e-7, options={'maxiter':100})
        else:
            result = minimize(self._loss_function_beta_beat, dqd0, args=(beta_beat_x, beta_beat_y), 
                              method='nelder-mead', tol=1e-7, options={'maxiter':100})
        
        
        print(result)
        
        # Update QD value, and also additional QF value if both planes are used
        self._line['qd.63510..1'].knl[1] = result.x[0]
        self._line['qf.63410..1'].knl[1] = result.x[1]
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
        
        # Find where this maximum beta function occurs
        betx_max_loc = twiss2.rows[np.argmax(twiss2.betx)].name[0]
        betx_max = twiss2.rows[np.argmax(twiss2.betx)].betx[0]
        bety_max_loc = twiss2.rows[np.argmax(twiss2.bety)].name[0]
        bety_max = twiss2.rows[np.argmax(twiss2.bety)].bety[0]
        
        # Add extra knob for the quadrupole
        self._line.vars['kk_QD'] = 0
        self._line.vars['kk_QF'] = 0
        self._line.element_refs['qd.63510..1'].knl[1] = self._line.vars['kk_QD']
        self._line.element_refs['qf.63410..1'].knl[1] = self._line.vars['kk_QF']
        
        # Try to rematch the tunes with the knobs
        try:
            
            self._line.match(
                vary=[
                    xt.Vary('kqf', step=1e-8),
                    xt.Vary('kqd', step=1e-8),
                    xt.Vary('kk_QD', step=1e-8),  #vary knobs and quadrupole simulatenously 
                    xt.Vary('kk_QF', step=1e-8),  #vary knobs and quadrupole simulatenously 
                    xt.Vary('qph_setvalue', step=1e-7),
                    xt.Vary('qpv_setvalue', step=1e-7)
                ],
                targets = [
                    xt.Target('qx', self.qx0, tol=1e-7),
                    xt.Target('qy', self.qy0, tol=1e-7),
                    xt.Target('betx', value=betx_max, at=betx_max_loc, tol=1e-7),
                    xt.Target('bety', value=bety_max, at=bety_max_loc, tol=1e-7),
                    xt.Target('dqx', self.dq1, tol=1e-7),
                    xt.Target('dqy', self.dq2, tol=1e-7),
                ])
            twiss3 = self._line.twiss()
        
            print('After matching: Qx = {:.4f}, Qy = {:.4f}, dQx = {:.4f}, dQy = {:.4f}\n'.format(twiss3['qx'], twiss3['qy'], twiss3['dqx'], twiss3['dqy']))
            print('New Y beat={:.5f}, X-beat={:.5f}'.format( (np.max(twiss3['bety']) - np.max(self._twiss0['bety']))/np.max(self._twiss0['bety']),
                                                            (np.max(twiss3['betx']) - np.max(self._twiss0['betx']))/np.max(self._twiss0['betx']) ))
            print('\nAchieved with new single quadrupolar knobs:')
            print('kk_QD = {:.6e}, kk_QF = {:.6e}'.format(self._line.vars['kk_QD']._value, self._line.vars['kk_QF']._value))
        except ValueError:
            print('Twiss unstable for these tunes, could not rematch again exactly')
        

        return self._line              


    def _loss_function_beta_beat(self, dqd, beta_beat_rms_X, beta_beat_rms_Y):
        """
        Loss function to optimize to find correct maximum beta-beat in 'X' or 'Y' or 'both'
        
        Parameters:
        ----------
        dqd : np.ndarray
            array containing: quadrupolar strength for first slice of last SPS quadrupoles, value of kqf and value of kqd knob
        beta_beat : float
            beta beat, i.e. relative difference between max beta function and max original beta function
        plane : str
            'X' or 'Y'
        
        Returns:
        --------
        loss - loss function value, np.abs(beta_beat - desired beta beat)
        """
        
        # Define beta-beat vector
        beta_beats = [beta_beat_rms_X, beta_beat_rms_Y]
            
        # Try with new quadrupole error, otherwise return high value (square of error)
        try:
            # Vary first slice of last quadrupole
            self._line['qd.63510..1'].knl[1] = dqd[0]
            self._line['qf.63410..1'].knl[1] = dqd[1]
            twiss2 = self._line.twiss()
        
            # Vertical plane beta-beat
            Y_beat = (np.max(twiss2['bety']) - np.max(self._twiss0['bety']))/np.max(self._twiss0['bety'])
            X_beat = (np.max(twiss2['betx']) - np.max(self._twiss0['betx']))/np.max(self._twiss0['betx'])
            print('Setting QD error to {:.3e}, QF error to {:.3e},  with Ybeat={:.4e}, Xbeat={:.4f}, qx = {:.4f}, qy = {:.4f}'.format(dqd[0],
                                                                                                                                      dqd[1],
                                                                                                                                      Y_beat, 
                                                                                                                                      X_beat,
                                                                                                                                      twiss2['qx'], 
                                                                                                                                      twiss2['qy']))
            # Define objective function as squared difference
            loss = (beta_beats[0] - X_beat)**2 + (beta_beats[1] - Y_beat)**2

        except ValueError:
            loss = dqd[0]**2 
            self._line = self._line0.copy()
            print('Resetting line...')
        
        return loss 
    
    
    def _loss_function_beta_beat_RMS(self, dqd, beta_beat_rms_X, beta_beat_rms_Y):
        """
        Loss function to optimize to find correct RMS beta-beat in 'X' or 'Y' or 'both'
        
        Parameters:
        ----------
        dqd : np.ndarray
            array containing: quadrupolar strength for first slice of last SPS quadrupoles, value of kqf and value of kqd knob
        beta_beat_rms_X : float
            RMS X beta beat around the ring, i.e. relative difference between max beta function and max original beta function
        beta_beat_rms_Y : float
            RMS X beta beat around the ring, i.e. relative difference between max beta function and max original beta function
        
        Returns:
        --------
        loss - loss function value, np.abs(beta_beat - desired beta beat)
        """
        
        # Define beta-beat vector
        beta_beats = [beta_beat_rms_X, beta_beat_rms_Y]
                
        # Try with new quadrupole error, otherwise return high value (square of error)
        try:
            # Vary first slice of last quadrupole
            self._line['qd.63510..1'].knl[1] = dqd[0]
            self._line['qf.63410..1'].knl[1] = dqd[1]
            twiss2 = self._line.twiss()
        
            # Calculate beta-beat
            beat_x = (self._twiss0['betx'][self.bpm_x_ind] - twiss2['betx'][self.bpm_x_ind]) / self._twiss0['betx'][self.bpm_x_ind] 
            beat_y = (self._twiss0['bety'][self.bpm_y_ind] - twiss2['bety'][self.bpm_y_ind]) / self._twiss0['bety'][self.bpm_y_ind] 
            rms_x = np.sqrt(np.sum(beat_x**2)/len(beat_x))
            rms_y = np.sqrt(np.sum(beat_y**2)/len(beat_y))
            
            print('Setting QD error to {:.3e}, QF error to {:.3e},  with Ybeat_RMS={:.4e}, Xbeat_RMS={:.4f}, qx = {:.4f}, qy = {:.4f}'.format(dqd[0],
                                                                                                                                              dqd[1], 
                                                                                                                                              rms_x,
                                                                                                                                              rms_y,
                                                                                                                                              twiss2['qx'], 
                                                                                                                                              twiss2['qy']))
            # Define objective function as squared difference
            loss = (beta_beats[0] - rms_x)**2 + (beta_beats[1] - rms_y)**2

        except ValueError:
            loss = dqd[0]**2 
            self._line = self._line0.copy()
            print('Resetting line...')
        
        return loss     


    @staticmethod
    def generate_SPS_gaussian_beam(line, n_part, exn=None, eyn=None, Nb=None, sigma_z=None):
        """ 
        Class method to generate matched Gaussian beam for SPS. Can provide custom parameters, otherwise default values
        
        Parameters:
        -----------
        line : xtrack.Line 
            line object for beam
        n_part : int
            number of macroparticles
        exn, eyn: float 
            horizontal and vertical normalized emittances
        Nb: float
            bunch intensity, number of ions per bunch
        sigma_z: float 
            bunch length in meters 
        
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
        self._Brho_PS_extr = self.B_PS_extr * self.rho_PS
        
        if self.ion_type == 'proton':
            m_in_eV = xt.PROTON_MASS_EV
        else:
            m_in_eV = self.m_ion * constants.physical_constants['atomic mass unit-electron volt relationship'][0]   # 1 Dalton in eV/c^2 -- atomic mass unit
        
        p_inj_SPS = 1e9 * (self._Brho_PS_extr * self.Q_PS) / 3.3356 # in  [eV/c], if q is number of elementary charges

        return m_in_eV, p_inj_SPS


    def load_simple_madx_seq(self,
                             add_non_linear_magnet_errors=False, 
                             make_thin=True, 
                             add_aperture=False,
                             nr_slices=5):
        """
        Loads default SPS Pb sequence at flat bottom. 
        
        Parameters:
        -----------
        add_non_linear_magnet_errors : bool
            whether to add line with non-linear chromatic errors
        make_thin : bool
            whether to slice the sequence or not
        add_aperture : bool
            whether to include aperture for SPS
        nr_slices : int
            number of slices when slicing MADX sequence from thick to thin
        
        Returns: 
        --------    
        madx - madx instance with SPS sequence    
        """
        # Check if proton or ion
        if self.ion_type=='proton':
            use_Pb_ions = False
        else:
            use_Pb_ions = True

        # Load madx instance
        madx = self.load_madx_SPS_from_job(use_Pb_ions=use_Pb_ions)

        # Flatten and slice line --> start at zero-dispersion location
        if make_thin:
            madx.use(sequence='sps')
            madx.input("seqedit, sequence=SPS;")
            madx.input("flatten;")
            madx.input("endedit;")
            madx.use("sps")
            madx.input("select, flag=makethin, slice={}, thick=false;".format(nr_slices))
            madx.input("makethin, sequence=sps, style=teapot, makedipedge=True;")

        # Add aperture classes
        if add_aperture:
            print('\nAdded aperture!\n')
            madx.use(sequence='sps')
            madx.call('{}/APERTURE_SPS_LS2_30-SEP-2020.seq'.format(aperture_fixed_path))
            
        # Cycle line to lowest dispersion location
        if make_thin:
            madx.use(sequence='sps')
            madx.input("seqedit, sequence=SPS;")
            madx.input("flatten;")
            madx.input("cycle, start=tacw.51998..5;")
            madx.input("endedit;")

        madx.call("{}/toolkit/macro.madx".format(optics))

        # Use correct tune and chromaticity matching macros
        madx.command.use(sequence='sps')       
        
        # Assign magnet errors - disappears if 'use' command is put in
        if add_non_linear_magnet_errors:
            madx.call('{}/sps_setMultipoles_upto7.cmd'.format(error_file_path))
            madx.input('exec, set_Multipoles_26GeV;')
            madx.call('{}/sps_assignMultipoles_upto7.cmd'.format(error_file_path))
            madx.input('exec, AssignMultipoles;')
            print('\nReassigned magnet errors!\n')
        print('Sliced sequences in {}'.format(nr_slices))

        madx.exec(f"sps_match_tunes({self.qx0}, {self.qy0});")
        madx.exec("sps_define_sext_knobs();")
        if self.proton_optics == 'q26':
            madx.exec("sps_set_chroma_weights_q26();")
        elif self.proton_optics == 'q20':
            madx.exec("sps_set_chroma_weights_q20();")
        madx.input(f"""match;
        global, dq1={self.dq1};
        global, dq2={self.dq2};
        vary, name=qph_setvalue;
        vary, name=qpv_setvalue;
        jacobian, calls=10, tolerance=1e-25;
        endmatch;""")
        
        return madx


    def load_madx_SPS_from_job(self, use_Pb_ions=True):
        """
        Loads default SPS Pb sequence at flat bottom for ions as in 
        https://gitlab.cern.ch/acc-models/acc-models-sps/-/tree/2021/scenarios/lhc/lhc_ion?ref_type=heads 
        and matches the tunes. 
        
        Parameters:
        -----------
        use_Pb_ions : bool
            whether to use Pb ion scenario (True), or protons (False)
        
        Returns: 
        --------    
        madx - madx instance with SPS sequence    
        """
        
        #### Initiate MADX sequence and call the sequence and optics file ####
        madx = Madx()
        madx.call("{}/sps.seq".format(optics))
        
        # Call the right magnet strengths for Q26: if not Pb ions, then protons
        if use_Pb_ions:
            if self.proton_optics == 'q26':
                madx.call("{}/strengths/lhc_ion.str".format(optics))
            elif self.proton_optics == 'q20':
                madx.call("{}/strengths/lhc_q20.str".format(optics))
        else:
            if self.proton_optics == 'q26':
                madx.call("{}/strengths/lhc_q26.str".format(optics))
            elif self.proton_optics == 'q20':
                madx.call("{}/strengths/lhc_q20.str".format(optics))

        # Generate SPS beam - use default Pb or make custom beam
        self.m_in_eV, self.p_inj_SPS = self.generate_SPS_beam()
        
        madx.input(" \
               Beam, particle=ion, mass={}, charge={}, pc = {}, sequence='sps'; \
               ".format(self.m_in_eV/1e9, self.Q_SPS, self.p_inj_SPS/1e9))   # convert mass to GeV/c^2
               
        ''' 
        Originally also had   
        DPP:=BEAM->SIGE*(BEAM->ENERGY/BEAM->PC)^2;
        but problematic when importing deferred expressions!
        ''' 
        madx.command.use(sequence='sps')

        return madx

    
    def generate_symmetric_SPS_lattice(self, 
                                       save_madx_seq=True, save_xsuite_seq=True, 
                                       return_xsuite_line=True,
                                       make_thin=True, voltage=3.0e6):
        """
        Replace all QFA magnets in SPS with QF, to make it fully symmetric
        
        Parameters:
        -----------
        save_madx_seq : bool 
            whether to save madx sequence to directory 
        save_xsuite_seq : bool 
            whether to save xtrack sequence to directory  
        return_xsuite_line : bool
            whether to return generated xtrack line
        make_thin : bool 
            flag to slice sequence or not
        voltage : float 
            RF voltage in V        
        
        Returns:
        --------
        line_symmetric - SPS xsuite line whose QFAs are replaced with QF 
        """
        
        # Update sequence folder location
        self.seq_folder = '{}/qy_dot{}'.format(sequence_path, int((self.qy0 % 1) * 100))
        os.makedirs(self.seq_folder, exist_ok=True)
        print('\nGenerating symmetric SPS in {}\n'.format(self.seq_folder))
        
        # Load MADX instance
        madx = self.load_simple_madx_seq(make_thin=False)
        
        #Print all QFA and QDA elements
        dash = '-' * 65
        header = '\n{:<27} {:>12} {:>15} {:>8}\n{}'.format("Element", "Location", "Type", "Length", dash)
        print(header)
        for ele in madx.sequence['sps'].elements:
            if ele.name[:3] == 'qfa' or ele.name[:3] == 'qda':   
                print('{:<27} {:>12.6f} {:>15} {:>8.3}'.format(ele.name, ele.at, ele.base_type.name, ele.length))
        print(dash)
        print('Printed all QFA and QDA magnets\n')

        # Reference quadrupoles
        ref_qf = madx.sequence['sps'].elements['qf.11010']
        ref_qd = madx.sequence['sps'].elements['qd.10110']

        # Initiate seqedit and replace all QFA and QDA magnets with a given QF
        madx.command.seqedit(sequence='sps')
        madx.command.flatten()
        for ele in madx.sequence['sps'].elements:
            if ele.name[:3] == 'qfa': 
                madx.command.replace(element=ele.name, by=ref_qf.name)
                print('Replacing {} by {}'.format(ele, ref_qf))
            elif ele.name[:3] == 'qda':   
                madx.command.replace(element=ele.name, by=ref_qd.name)
                print('Replacing {} by {}'.format(ele, ref_qd))
        madx.command.endedit()

        print('\nNew sequence quadrupoles:')
        print(header)
        for ele in madx.sequence['sps'].elements:
            if ele.name[:2] == 'qf' or ele.name[:2] == 'qd': 
                print('{:<27} {:>12.6f} {:>15} {:>8.3}'.format(ele.name, ele.at, ele.base_type.name, ele.length))
        print(dash)

        # Check if remaining QFAs or QDAs
        print("\nRemaining QFAs or QDAs")
        print(header)
        for ele in madx.sequence['sps'].elements:
            if ele.name[:3] == 'qfa' or ele.name[:3] == 'qda':   
                print('{:<27} {:>12.6f} {:>15} {:>8.3}'.format(ele.name, ele.at, ele.base_type.name, ele.length))
        print(dash)

        # Flatten and slice line
        if make_thin:
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
        
        # Generate SPS beam - use default Pb or make custom beam
        self.m_in_eV, self.p_inj_SPS = self.generate_SPS_beam()
        
        self.particle_sample = xp.Particles(
                p0c = self.p_inj_SPS,
                q0 = self.Q_SPS,
                mass0 = self.m_in_eV)
        print('\nGenerated SPS {} beam p = {:.4f}, gamma = {:.3f}, Qx = {:.3f}, Qy = {:.3f}\n'.format(self.ion_type,
                                                                                                      self.p_inj_SPS * 1e-9, 
                                                                                                      self.particle_sample.gamma0[0],
                                                                                                      self.qx0,
                                                                                                      self.qy0))
        
        line.particle_ref = self.particle_sample
        
        ############## ADD RF VOLTAGE FOR LONGITUDINAL - DIFFERENT FOR MADX AND XSUITE ##############
        
        #### SET CAVITY VOLTAGE - with info from Hannes
        # 6x200 MHz cavities: actcse, actcsf, actcsh, actcsi (3 modules), actcsg, actcsj (4 modules)
        # acl 800 MHz cavities
        # acfca crab cavities
        # Ions: all 200 MHz cavities: 1.7 MV, h=4653
        harmonic_nb = 4653
        nn = 'actcse.31632'
        
        # MADX sequence 
        madx.sequence.sps.elements[nn].lag = 0
        madx.sequence.sps.elements[nn].volt = (voltage/1e6)*self.particle_sample.q0 # different convention between madx and xsuite
        madx.sequence.sps.elements[nn].freq = madx.sequence['sps'].beam.freq0*harmonic_nb
        
        # Xsuite sequence 
        line[nn].lag = 0  # 0 if below transition
        line[nn].voltage = voltage # In Xsuite for ions, do not multiply by charge as in MADX
        line[nn].frequency = madx.sequence['sps'].beam.freq0*1e6*harmonic_nb
        
        twiss = line.twiss()
        
        # Save MADX sequence
        if save_madx_seq:
            madx.command.save(sequence='sps', file='{}/SPS_2021_{}_symmetric.seq'.format(self.seq_folder, 
                                                                                  self.ion_type), beam=True)  
        # Save Xsuite sequence
        if save_xsuite_seq:
            with open('{}/SPS_2021_{}_symmetric.json'.format(self.seq_folder, self.ion_type), 'w') as fid:
                json.dump(line.to_dict(), fid, cls=xo.JEncoder)
            print('\nSaved symmetric SPS in {}\n'.format(self.seq_folder))
                
        if return_xsuite_line:
            return line


    def remove_aperture_below_threshold(self, line : xt.Line, aperture_min : float) -> xt.Line:
        """Removes all aperture element in line smaller than specified horizontal aperture_min"""
        # Generate aperture table
        x_ap, y_ap, a = self.print_smallest_aperture(line)

        # Sort and show how many values that are small
        a = a.iloc[:-1] # drop last None row
        aa = a.iloc[x_ap < aperture_min]

        # Remove all apertures that are too small
        mask = [True] * len(line.elements)
        for i in aa.index:
            mask[i] = False
        line = line.filter_elements(mask) # remove these elements

        # Check aperture table of new line 
        _, _, _ = self.print_smallest_aperture(line)
        
        return line


    def print_smallest_aperture(self, line: xt.Line, add_beta_beat=False):
        """function to return and print smallest aperture values"""

        # Get aperture table
        a0 = line.check_aperture()
        a = a0[a0['is_aperture']] # remove elements without aperture
        
        if add_beta_beat:
            line.element_refs['qd.63510..1'].knl[1] = -1.07328640311457e-02
            line.element_refs['qf.63410..1'].knl[1] = 1.08678014669101e-02
            print('Beta-beat added: kk_QD = {:.6e}, kk_QF = {:.6e}'.format(line.element_refs['qd.63510..1'].knl[1]._value,
                                                                           line.element_refs['qf.63410..1'].knl[1]._value))
        
        # Get Twiss values at the aperture
        df_twiss = line.twiss().to_pandas()
        df_twiss = df_twiss[a0['is_aperture']]

        # Loop over all elements to find aperture values
        x_ap = []
        y_ap = []
        s_ap = [] # location of aperture
        ind = []

        for i, ele in enumerate(a.element):
            if ele is not None:
                x_ap.append(ele.max_x)
                y_ap.append(ele.max_y)
                s_ap.append(a.s.iloc[i])
                ind.append(i)
                
        df_twiss = df_twiss.iloc[ind]

        # Convert to numpy arrays
        x_ap = np.array(x_ap)
        x_ap_norm = x_ap / np.sqrt(df_twiss.betx.values)
        y_ap = np.array(y_ap)
        y_ap_norm = x_ap / np.sqrt(df_twiss.bety.values)
        s_ap = np.array(s_ap)
        ind = np.array(ind)

        # Find minimum PHYSICAL aperture
        print('\nPHYSICAL aperture:')
        print('Minimum X aperture is x_min={} m at s={} m'.format(x_ap[np.argmin(x_ap)], s_ap[np.argmin(x_ap)]))
        print('Minimum Y aperture is y_min={} m at s={} m'.format(y_ap[np.argmin(y_ap)], s_ap[np.argmin(y_ap)]))

        print('X aperture unique counts:')
        print(np.unique(x_ap, return_counts=True))
        print('Y aperture unique counts:\n')
        print(np.unique(y_ap, return_counts=True))
        
        # Find minimum NORMALIZED aperture
        print('\nNORMALIZED aperture:')
        print('Minimum norm X aperture is x_min={} m at s={} m'.format(x_ap_norm[np.argmin(x_ap_norm)], s_ap[np.argmin(x_ap_norm)]))
        print('Minimum norm Y aperture is y_min={} m at s={} m'.format(y_ap_norm[np.argmin(y_ap_norm)], s_ap[np.argmin(y_ap_norm)]))

        print('Norm X aperture unique counts:')
        print(np.unique(x_ap_norm, return_counts=True))
        print('Norm Y aperture unique counts:\n')
        print(np.unique(y_ap_norm, return_counts=True))
        print('\n')

        return x_ap, y_ap, x_ap_norm, y_ap_norm, a
    
    
    def set_LSE_sextupolar_errors(self, line)->xt.Line:
        """
        Add sextupolar component to the extraction LSE sextupole in SPS (normally zero-valued) to
        mimic residual sextupolar components of machine - from measurements done with Kostas Paraschou 
        on 2024-10-10 in the SPS (see elogbook https://logbook.cern.ch/elogbook-server/GET/showEventInLogbook/4160116)
        
        Parameters:
        -----------
        line : xtrack.line
            xtrack line object to search through

        Returns:
        --------
        line : xtrack.line
            xtrack line object with new sextupole values
        """
        lse_names = ['lse.12402', 'lse.20602', 'lsen.42402', 'lse.50602', 'lse.62402']
        k2_values = np.array([0.02295123,  0.03247354, -0.0141614 , -0.0314969 , -0.01139423])
        
        # Iterate over extraction sextupole, find in SPS sequence and set value
        for i, lse_name in enumerate(lse_names):
            # Iterate over SPS line
            for key in line.element_names:

                # For each slice with name, multiply k2 value with length to get integrated B field strength
                if type(line[key]) == xt.beam_elements.elements.Multipole and lse_name in key:
                    k2 = line[key].length * k2_values[i]
                    line.element_dict[key] = xt.Multipole(knl = [0, 0, k2], length=line[key].length)
                    print('{}: replaced and set to knl = {}'.format(key, line[key].knl))

        return line
    

    def excite_LSE_sextupole_from_current(self, line, I_LSE, which_LSE='lse.12402')->xt.Line:
        """
        Add sextupolar component to a extraction LSE sextupole of choice in SPS (normally zero-valued)
        Set current, then convert to normalized k strength
        
        Parameters:
        -----------
        line : xtrack.line
            xtrack line object to search through

        Returns:
        --------
        line : xtrack.line
            xtrack line object with new sextupole values
        I_LSE : float
            how much sextupolar current to excite with. The LSEs are 
            'lse.12402', 'lse.20602', 'lsen.42402', 'lse.50602' or 'lse.62402'
        which_LSE : str
            which LSE sextupole to excite with 
        """
        # Converting factor and polarity for LSE compared to currents - from Kostas
        polarity = -1.0 if which_LSE=='lse.12402' else 1.0
        K2I = -49.1477
        I2K = polarity * 1/K2I
        K2 = I_LSE * I2K
        
        print('\nExiciting LSE sextupole {} with I = {:.3f} --> K2 = {:.5f}'.format(which_LSE, I_LSE, K2))
        # Iterate over SPS sequence and set LSE value
        for key in line.element_names:
            # For each slice with name, multiply k2 value with length to get integrated B field strength
            if type(line[key]) == xt.beam_elements.elements.Multipole and which_LSE in key:
                k2 = line[key].length * K2
                line.element_dict[key] = xt.Multipole(knl = [0, 0, k2], length=line[key].length)
                print('{}: replaced and set to knl = {}'.format(key, line[key].knl))

        return line


    def set_LOE_octupolar_errors(self, line)->xt.Line:
        """
        Add octupolar component to the extraction LOE octupoles in SPS (normally zero-valued) to
        mimic residual sextupolar components of machine - from measurements done with Kostas Paraschou 
        on 2023-10-05 in the SPS (see elogbook https://logbook.cern.ch/elogbook-server/GET/showEventInLogbook/3842235)
        
        Parameters:
        -----------
        line : xtrack.line
            xtrack line object to search through

        Returns:
        --------
        line : xtrack.line
            xtrack line object with new octupolar values
        """
        loe_names = ['loe.12002', 'loe.10402']
        k3_values = np.array([4.0, -2.0])
        
        # Iterate over extraction sextupole, find in SPS sequence and set value
        for i, loe_name in enumerate(loe_names):
            # Iterate over SPS line
            for key in line.element_names:

                # For each slice with name, multiply k3 value with length to get integrated B field strength
                if type(line[key]) == xt.beam_elements.elements.Multipole and loe_name in key:
                    k3 = line[key].length * k3_values[i]
                    line.element_dict[key] = xt.Multipole(knl = [0, 0, 0, k3], length=line[key].length)
                    print('{}: replaced and set to knl = {}'.format(key, line[key].knl))

        return line
    

    def _print_multipolar_elements_in_line(self, line, order=1)->None:
        """
        Print all quadrupolar elements for a given order (default 1, i.e. quadrupole)
        
        Parameters:
        -----------
        line : xtrack.line
            xtrack line object to search through
        order : int
            multipolar order to print. Default "1" means quadrupolar components
        """
        for key in line.element_names:
            if type(line[key]) == xt.beam_elements.elements.Multipole and line[key]._order == order:
                print('{}: knl = {}'.format(key, line[key].knl))
    