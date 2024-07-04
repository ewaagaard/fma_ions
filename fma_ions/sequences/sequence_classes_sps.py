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

optics =  Path(__file__).resolve().parent.joinpath('../../data/acc-models-sps').absolute()
sequence_path = Path(__file__).resolve().parent.joinpath('../../data/sps_sequences').absolute()
error_file_path = Path(__file__).resolve().parent.joinpath('../../data/sps_sequences/magnet_errors').absolute()

@dataclass
class BeamParameters_SPS:
    """Data Container for SPS Pb default beam parameters"""
    Nb:  float = 2.46e8 # measured 2.46e8 ions per bunch on 2023-10-16
    sigma_z: float = 0.225 # in m, is the old value (close to Isabelle's and  Hannes'), but then bucket is too full if Gaussian longitudinal. 0.19 also used
    exn: float = 1.1e-6
    eyn: float = 0.9e-6

@dataclass
class BeamParameters_SPS_Binomial_2016:
    """
    Data Container for SPS Pb longitudinally binomial beam parameters, from 2016 measurements,
    after initial spill out of RF bucket has happened
    """
    Nb: float = 3.536e8 * 0.95 # injected intensity, after initial spill out of RF bucket
    sigma_z: float = 0.213 # RMS bunch length of binomial, after initial spill out of RF bucket #0.213 measured, but takes ~30 turns to stabilze
    m : float = 2.98 # binomial parameter to determine tail of parabolic distribution, after initial spill out of RF bucket
    q : float = 0.59 # q-Gaussian parameter after RF spill (third profile)
    exn: float = 1.3e-6
    eyn: float = 0.9e-6

@dataclass
class BeamParameters_SPS_Binomial_2016_before_RF_Spill:
    """
    Data Container for SPS Pb longitudinally binomial beam parameters, from 2016 measurements, 
    before initial RF spill
    """
    Nb:  float = 3.536e8  # injected bunch intensity measured with Wall Current Monitor (WCM)
    sigma_z: float = 0.286 # RMS bunch length of binomial, measured before RF spill
    m : float = 6.124 # binomial parameter to determine tail of parabolic distribution
    q : float = 0.82 # q-Gaussian parameter
    exn: float = 1.3e-6
    eyn: float = 0.9e-6

@dataclass
class BeamParameters_SPS_Oxygen:
    """Data Container for SPS oxygen beam parameters"""
    Nb:  float = 25e8 # half of (John, Bartosik 2021) for oxygen, assuming bunch splitting
    sigma_z: float = 0.225 # in m, is the old value (close to Isabelle's and  Hannes'), but then bucket is too full if Gaussian longitudinal. 0.19 also used
    sigma_z_binomial: float = 0.285 # RMS bunch length of binomial, default value to match data
    m : float = 5.3 # binomial parameter to determine tail of parabolic distribution
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
    qy0: float = 26.25
    dq1: float = -3.460734474533172e-09 
    dq2: float = -3.14426538905229e-09
    # Default SPS PB ION CHROMA VALUES: not displayed on acc-model, extracted from PTC Twiss 
    
    # Define beam type - default is Pb
    ion_type: str = 'Pb'
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
            
    
    def load_xsuite_line_and_twiss(self,
                                   beta_beat=None, 
                                   use_symmetric_lattice=False,
                                   add_non_linear_magnet_errors=False, 
                                   save_new_xtrack_line=True,
                                   deferred_expressions=False, 
                                   add_aperture=False, 
                                   plane='Y',
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
                            deferred_expressions=False,
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
            sps_fname = '{}/SPS_2021_{}{}{}{}{}.json'.format(self.seq_folder, self.ion_type, symmetric_string,
                                                                      def_exp_str, err_str, proton_optics_str)
            with open(sps_fname, 'w') as fid:
                json.dump(line.to_dict(), fid, cls=xo.JEncoder)
                
        if return_xsuite_line:
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
                                           beta_beat=0.05,
                                           save_xsuite_seq=False, 
                                           line=None,
                                           use_symmetric_lattice=False,
                                           add_non_linear_magnet_errors=False,
                                           plane='Y',
                                           add_aperture=False,
                                           voltage=3.0e6
                                           ):
        """
        Generate Xsuite line with desired beta beat, optimizer finds
        quadrupole error in first slice of last SPS quadrupole to emulate desired beta_beat
        
        Parameters:
        -----------
        beta_beat : float
            desired beta beat, i.e. relative difference between max beta function and max original beta function. If
            not 'both', the other plane will be assumed to have 0 beta-beat
        save_xsuite_seq : bool
            flag to save xsuite sequence in desired location
        line : xtrack.Line
            can provide generated line, otherwise generate new
        add_non_linear_magnet_errors : bool
            add errors from non-linear chromaticity if desired 
        add_non_linear_magnet_errors : bool
            whether to add line with non-linear chromatic errors
        plane : str
            'X' or 'Y' or 'both' - which plane(s) to find beta-beat for
        add_aperture : bool
            whether to include aperture for SPS
        voltage : float
            RF voltage in V
        
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

        # Substrings to identify line
        symmetric_string = '_symmetric' if use_symmetric_lattice else '_nominal'
        err_str = '_with_non_linear_chrom_error' if add_non_linear_magnet_errors else ''

        Qy_frac = int(100*(np.round(self.qy0 % 1, 2)))
        self._line, _ = self.load_SPS_line_with_deferred_madx_expressions(use_symmetric_lattice=use_symmetric_lattice, Qy_frac=Qy_frac,
                                                         add_non_linear_magnet_errors=add_non_linear_magnet_errors, add_aperture=add_aperture,
                                                         voltage=voltage)
        
        self._line0 = self._line.copy()
        self._twiss0 = self._line0.twiss()
        
        # Initial guess: small perturbation to initial value, then minimize loss function
        print('\nBeta-beat search for {} in {} plane(s)!\n'.format(beta_beat, plane))
        dqd0 = [self._line0['qd.63510..1'].knl[1] + 1e-6, self._line0['qf.63410..1'].knl[1] + 1e-6]
        result = minimize(self._loss_function_beta_beat, dqd0, args=(beta_beat, plane), 
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
        print('Generated sequence with Qx, Qy = ({}, {}) and beta-beat = {}!\n'.format(twiss2['qx'], twiss2['qy'], beta_beat))
        
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
        
        # Rematch the tunes with the knobs
        self._line.match(
            vary=[
                xt.Vary('kqf', step=1e-8),
                xt.Vary('kqd', step=1e-8),
                xt.Vary('kk_QD', step=1e-8),  #vary knobs and quadrupole simulatenously 
                xt.Vary('kk_QF', step=1e-8),  #vary knobs and quadrupole simulatenously 
            ],
            targets = [
                xt.Target('qx', self.qx0, tol=1e-7),
                xt.Target('qy', self.qy0, tol=1e-7),
                xt.Target('betx', value=betx_max, at=betx_max_loc, tol=1e-7),
                xt.Target('bety', value=bety_max, at=bety_max_loc, tol=1e-7)
            ])
      
        twiss3 = self._line.twiss()
        
        print('\nTunes rematched to qx = {:.4f}, qy = {:.4f}\n'.format(twiss3['qx'], twiss3['qy']))
        print('New Y beat={:.5f}, X-beat={:.5f}'.format( (np.max(twiss3['bety']) - np.max(self._twiss0['bety']))/np.max(self._twiss0['bety']),
                                                         (np.max(twiss3['betx']) - np.max(self._twiss0['betx']))/np.max(self._twiss0['betx']) ))
        
        
        # Save Xsuite sequence
        sps_fname = '{}/qy_dot{}/SPS_2021_{}{}_{}plane_{}_percent_beta_beat{}{}.json'.format(sequence_path, Qy_frac, self.ion_type, 
                                                                                 symmetric_string, plane, int(beta_beat*100), err_str,
                                                                                 proton_optics_str)
        if save_xsuite_seq:
            with open(sps_fname, 'w') as fid:
                json.dump(self._line.to_dict(), fid, cls=xo.JEncoder)
                
        return self._line              


    def _loss_function_beta_beat(self, dqd, beta_beat, plane='Y'):
        """
        Loss function to optimize to find correct beta-beat in 'X' or 'Y' or 'both'
        
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
        if plane=='Y':
            beta_beats = [0.0, beta_beat]
        elif plane=='X':
            beta_beats = [beta_beat, 0.0]
        elif plane=='both':
            beta_beats = [beta_beat, beta_beat]
        else:
            raise ValueError('Invalid plane!')
            
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

        # Flatten and slice line
        if make_thin:
            madx.use(sequence='sps')
            madx.input("seqedit, sequence=SPS;")
            madx.input("flatten;")
            madx.input("endedit;")
            madx.use("sps")
            madx.input("select, flag=makethin, slice={}, thick=false;".format(nr_slices))
            madx.input("makethin, sequence=sps, style=teapot, makedipedge=True;")

        madx.call("{}/toolkit/macro.madx".format(optics))

        # Add aperture classes
        if add_aperture:
            print('\nAdded aperture!\n')
            madx.use(sequence='sps')
            madx.call('{}/aperture/APERTURE_SPS_LS2_30-SEP-2020.seq'.format(optics))

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


    def print_smallest_aperture(self, line: xt.Line):
        """function to return and print smallest aperture values"""

        # Get aperture table
        a = line.check_aperture()
        a = a[a['is_aperture']] # remove elements without aperture

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

        # Convert to numpy arrays
        x_ap = np.array(x_ap)
        y_ap = np.array(y_ap)
        s_ap = np.array(s_ap)
        ind = np.array(ind)

        # Find minimum aperture
        print('\nMinimum X aperture is x_min={} m at s={} m'.format(x_ap[np.argmin(x_ap)], s_ap[np.argmin(x_ap)]))
        print('Minimum Y aperture is y_min={} m at s={} m'.format(y_ap[np.argmin(y_ap)], s_ap[np.argmin(y_ap)]))

        return x_ap, y_ap, a