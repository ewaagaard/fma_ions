"""
Class container for methods to track xpart particle objects at flat bottom
- for SPS
- choose context (GPU, CPU) and additional effects: SC, IBS, tune ripples
"""
from dataclasses import dataclass
import numpy as np
import xtrack as xt
import xpart as xp
import xfields as xf
import xobjects as xo

from .sequence_classes_ps import PS_sequence_maker, BeamParameters_PS
from .sequence_classes_sps import SPS_sequence_maker, BeamParameters_SPS
from .fma_ions import FMA
from .helpers import Records, _bunch_length, _geom_epsx, _geom_epsy, _sigma_delta
from .tune_ripple import Tune_Ripple_SPS

from xibs.inputs import BeamParameters, OpticsParameters
from xibs.kicks import KineticKickIBS

import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import time

@dataclass
class SPS_Flat_Bottom_Tracker:
    """
    Container to track xp.Particles at SPS flat bottom and store beam parameter results
    """
    num_part: int = 10_000
    num_turns: int = 1000
    output_folder : str = "output" 
    turn_print_interval : int = 500
    qx0: float = 26.30
    qy0: float = 26.19

    def generate_particles(self, line: xt.Line, context : xo.context, use_Gaussian_distribution=True, beamParams=None
                           ) -> xp.Particles:
        """
        Generate xp.Particles object: matched Gaussian or other types (to be implemented)
        """
        if beamParams is None:
            beamParams = BeamParameters_SPS

        if use_Gaussian_distribution:
            particles = xp.generate_matched_gaussian_bunch(_context=context,
                num_particles=self.num_part, 
                total_intensity_particles=beamParams.Nb,
                nemitt_x=beamParams.exn, 
                nemitt_y=beamParams.eyn, 
                sigma_z= beamParams.sigma_z,
                particle_ref=line.particle_ref, 
                line=line)
            
        return particles


    def track_SPS(self, 
                  save_tbt_data=True, 
                  which_context='cpu',
                  add_non_linear_magnet_errors=False, 
                  add_aperture=True,
                  beta_beat=None, 
                  beamParams=None,
                  install_SC_on_line=True, 
                  SC_mode='frozen',
                  use_Gaussian_distribution=True,
                  apply_kinetic_IBS_kicks=False,
                  harmonic_nb = 4653,
                  ibs_step = 100,
                  Qy_frac: int = 25,
                  print_lost_particle_state=True,
                  minimum_aperture_to_remove=0.025,
                  add_tune_ripple=False,
                  dq=0.01,
                  ripple_freq=50
                  ):
        """
        Run full tracking at SPS flat bottom
        
        Parameters:
        ----------
        save_tbt: bool
            whether to save turn-by-turn data from tracking
        which_context : str
            'gpu' or 'cpu'
        Qy_frac : int
            fractional part of vertical tune
        add_non_linear_magnet_errors : bool
            whether to add line with non-linear chromatic errors
        add_aperture : bool
            whether to include aperture for SPS
        beta_beat : float
            relative beta beat, i.e. relative difference between max beta function and max original beta function
        beamParams : dataclass
            container of exn, eyn, Nb and sigma_z. Default 'None' will load nominal SPS beam parameters 
        install_SC_on_line : bool
            whether to install space charge
        SC_mode : str
            type of space charge - 'frozen' (recommended), 'quasi-frozen' or 'PIC'
        use_Gaussian_distribution : bool
            whether to use Gaussian particle distribution for tracking
        add_kinetic_IBS_kicks : bool
            whether to apply kinetic kicks from xibs 
        harmonic_nb : int
            harmonic used for SPS RF system
        ibs_step : int
            turn interval at which to recalculate IBS growth rates
        Qy_frac : int
            fractional part of vertical tune, e.g. "19" for 26.19
        minimum_aperture_to_remove : float 
            minimum threshold of horizontal SPS aperture to remove, default is 0.025 (can also be set to None)
            as faulty IPM aperture has 0.01 m, which is too small
        add_tune_ripple : bool
            whether to add external tune ripple from the Tune_Ripple_SPS class
        dq : float
            amplitude for tune ripple, if applied
        ripple_freq : float
            ripple frequency in Hz
            
        Returns:
        --------
        None
        """
        # Update vertical tune if changed
        self.qy0 = int(self.qy0) + Qy_frac / 100

        # If specific beam parameters are not provided, load default SPS beam parameters
        if beamParams is None:
            beamParams = BeamParameters_SPS
        print('Beam parameters:', beamParams)

        # Select relevant context
        if which_context=='gpu':
            context = xo.ContextCupy()
        elif which_context=='cpu':
            context = xo.ContextCpu(omp_num_threads='auto')
        else:
            raise ValueError('Context is either "gpu" or "cpu"')

        # Deferred expressions only needed for tune ripple
        load_line_with_deferred_expressions = True if add_tune_ripple else False

        # Get SPS Pb line - with aperture and non-linear magnet errors if desired
        sps = SPS_sequence_maker()
        line, twiss = sps.load_xsuite_line_and_twiss(Qy_frac=Qy_frac, add_aperture=add_aperture, beta_beat=beta_beat,
                                                   add_non_linear_magnet_errors=add_non_linear_magnet_errors, 
                                                   deferred_expressions=load_line_with_deferred_expressions)
                
        if minimum_aperture_to_remove is not None:
            line = sps.remove_aperture_below_threshold(line, minimum_aperture_to_remove)

        # Add longitudinal limit rectangle - to kill particles that fall out of bucket
        bucket_length = line.get_length()/harmonic_nb
        line.unfreeze() # if you had already build the tracker
        line.append_element(element=xt.LongitudinalLimitRect(min_zeta=-bucket_length/2, max_zeta=bucket_length/2), name='long_limit')
        line.build_tracker(_context=context)

        # Generate particles object to track    
        particles = self.generate_particles(line=line, context=context, use_Gaussian_distribution=use_Gaussian_distribution,
                                            beamParams=beamParams)

        # Initialize the dataclasses and store the initial values
        tbt = Records.init_zeroes(self.num_turns)
        tbt.update_at_turn(0, particles, twiss)

        ######### IBS kinetic kicks #########
        if apply_kinetic_IBS_kicks:
            beamparams = BeamParameters.from_line(line, n_part=beamParams.Nb)
            opticsparams = OpticsParameters.from_line(line) # read from line without space  charge
            IBS = KineticKickIBS(beamparams, opticsparams)
            kinetic_kick_coefficients = IBS.compute_kick_coefficients(particles)
            print(kinetic_kick_coefficients)

        # Install SC and build tracker - optimize line if line variables for tune ripple not needed
        if install_SC_on_line:
            fma_sps = FMA()
            line = fma_sps.install_SC_and_get_line(line, beamParams, mode=SC_mode, optimize_for_tracking=(not add_tune_ripple), context=context)
            print('Installed space charge on line\n')

        # Add tune ripple
        if add_tune_ripple:
            turns_per_sec = 1/twiss['T_rev0']
            ripple_period = int(turns_per_sec/ripple_freq)  # number of turns particle makes during one ripple oscillation
            ripple = Tune_Ripple_SPS(Qy_frac=Qy_frac, beta_beat=beta_beat, num_turns=self.num_turns, ripple_period=ripple_period)
            kqf_vals, kqd_vals, _ = ripple.load_k_from_xtrack_matching(dq=dq, plane='X')

        # Start tracking 
        time00 = time.time()
        for turn in range(1, self.num_turns):
            
            if turn % self.turn_print_interval == 0:
                print('\nTracking turn {}'.format(turn))            

            ########## IBS -> Potentially re-compute the ellitest_parts integrals and IBS growth rates #########
            if apply_kinetic_IBS_kicks and ((turn % ibs_step == 0) or (turn == 1)):
                
                # We compute from values at the previous turn
                kinetic_kick_coefficients = IBS.compute_kick_coefficients(particles)
                print(
                    "\n" + "=" * 60 + "\n",
                    f"Turn {turn:d}: re-computing growth rates and kick coefficients\n",
                    kinetic_kick_coefficients,
                    "\n" + "=" * 60,
                )
                
            ########## ----- Apply IBS Kick if desired ----- ##########
            if apply_kinetic_IBS_kicks:
                IBS.apply_ibs_kick(particles)
            
            ########## ----- Exert TUNE RIPPLE if desired ----- ##########
            if add_tune_ripple:
                line.vars['kqf'] = kqf_vals[turn-1]
                line.vars['kqd'] = kqd_vals[turn-1]

            # ----- Track and update records for tracked particles ----- #
            line.track(particles, num_turns=1)
            tbt.update_at_turn(turn, particles, twiss)

            if particles.state[particles.state <= 0].size > 0:
                if print_lost_particle_state and turn % self.turn_print_interval == 0:
                    print('Lost particle state: most common code: "-{}" for {} particles out of {} lost in total'.format(np.bincount(np.abs(particles.state[particles.state <= 0])).argmax(),
                                                                                                          np.max(np.bincount(np.abs(particles.state[particles.state <= 0]))),
                                                                                                          len(particles.state[particles.state <= 0])))
        time01 = time.time()
        dt0 = time01-time00
        print('\nTracking time: {} s'.format(dt0))
        
        # Make parquet file from dictionary
        if save_tbt_data:
            tbt_dict = tbt.to_dict()
            df = pd.DataFrame(tbt_dict)
        
            return df


    def introduce_beta_beat(self, line : xt.Line, twiss : xt.TwissTable, beta_beat : float) -> xt.Line:
        """Method to introduce quadrupolar error"""

        # Create knobs controlling all quads
        ltab = line.get_table()
        line.vars['k1l.qf'] = 0
        line.vars['k1l.qd'] = 0

        qftab = ltab.rows['qf.*']
        for i, nn in enumerate(qftab.name):
            if qftab.element_type[i] == 'Multipole':
                line.element_refs[nn].knl[1] = line.vars['k1l.qf']

        qdtab = ltab.rows['qd.*']
        for i, nn in enumerate(qdtab.name):
            if qdtab.element_type[i] == 'Multipole':
                line.element_refs[nn].knl[1] = line.vars['k1l.qd']

        # First add extra knob for the quadrupole
        line.vars['kk_QD'] = 0
        line.element_refs['qd.63510..1'].knl[1] = line.vars['kk_QD']
        
        # Find where this maximum beta function occurs
        betx_max_loc = twiss.rows[np.argmax(twiss.betx)].name[0]
        betx_max = (1 + beta_beat) * twiss.rows[np.argmax(twiss.betx)].betx[0]

        # Rematch the tunes with the knobs
        line.match(
            vary=[
                xt.Vary('k1l.qf', step=1e-8),
                xt.Vary('k1l.qd', step=1e-8),
                xt.Vary('kk_QD', step=1e-8),  #vary knobs and quadrupole simulatenously 
            ],
            targets = [
                xt.Target('qx', self.qx0, tol=1e-7),
                xt.Target('qy', self.qy0, tol=1e-7),
                xt.Target('betx', value=betx_max, at=betx_max_loc, tol=1e-7)
            ])
        
        return line 


    def load_tbt_data(self, output_folder=None) -> Records:
        """
        Loads numpy data if tracking has already been made
        """
        folder_path = '{}/'.format(output_folder) if output_folder is not None else ''

        # Read the parquet file
        tbt = pd.read_parquet('{}tbt.parquet'.format(folder_path))
        return tbt


    def plot_tracking_data(self, tbt, show_plot=False):
        """Generates emittance plots from TBT data class"""

        turns = np.arange(len(tbt.exn), dtype=int) 

        # Emittances and bunch intensity 
        f, (ax1, ax2) = plt.subplots(1, 2, figsize = (10,5))

        ax1.plot(turns, tbt.exn * 1e6, alpha=0.7, lw=1.5, label='X')
        ax1.plot(turns, tbt.eyn * 1e6, lw=1.5, label='Y')
        ax2.plot(turns, tbt.Nb, alpha=0.7, lw=1.5, c='r', label='Bunch intensity')

        ax1.set_ylabel(r'$\varepsilon_{x, y}$ [$\mu$m]')
        ax1.set_xlabel('Turns')
        ax2.set_ylabel(r'$N_{b}$')
        ax2.set_xlabel('Turns')
        ax1.legend()
        f.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        # Emittances and bunch intensity 
        f2, (ax12, ax22) = plt.subplots(1, 2, figsize = (8,4))

        ax12.plot(turns, tbt.sigma_delta * 1e3, alpha=0.7, lw=1.5, label='$\sigma_{\delta}$')
        ax22.plot(turns, tbt.bunch_length, alpha=0.7, lw=1.5, label='Bunch intensity')

        ax12.set_ylabel(r'$\sigma_{\delta}$')
        ax12.set_xlabel('Turns')
        ax22.set_ylabel(r'$\sigma_{z}$ [m]')
        ax22.set_xlabel('Turns')    
        f2.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        # Save figures
        os.makedirs(self.output_folder, exist_ok=True)
        f.savefig('{}/epsilon_Nb.png'.format(self.output_folder), dpi=250)
        f2.savefig('{}/sigma_z_and_delta.png'.format(self.output_folder), dpi=250)

        if show_plot:
            plt.show()
        plt.close()
        
    
    def load_tbt_data_and_plot(self, show_plot=False):
        """Load already tracked data and plot"""
        try:
            tbt = self.load_tbt_data()
            self.plot_tracking_data(tbt, show_plot=show_plot)
        except FileNotFoundError:
            raise FileNotFoundError('Tracking data does not exist - set correct path or generate the data!')
        

    def plot_multiple_sets_of_tracking_data(self, output_str_array, string_array):
        """
        If multiple runs with turn-by-turn (tbt) data has been made, provide list with Records class objects and list
        of explaining string to generate comparative plots of emittances, bunch intensities, etc

        Parameters:
        ----------
        output_str_array : [outfolder, outfolder, ...]
            List containing string for outfolder tbt data
        string:_array : [str1, str2, ...]
            List containing strings to explain the respective tbt data objects (which parameters were used)
        """
        os.makedirs('main_plots', exist_ok=True)

        # Load TBT data 
        tbt_array = []
        for output_folder in output_str_array:
            self.output_folder = output_folder
            tbt = self.load_tbt_data(output_folder)
            tbt['turns'] = np.arange(len(tbt.Nb), dtype=int)
            tbt_array.append(tbt)

        # Emittances and bunch intensity 
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (14,5))

        # Loop over the tbt records classes 
        for i, tbt in enumerate(tbt_array):
            ax1.plot(tbt.turns, tbt.exn * 1e6, alpha=0.7, lw=1.5, label=string_array[i])
            ax2.plot(tbt.turns, tbt.eyn * 1e6, alpha=0.7, lw=1.5, label=string_array[i])
            ax3.plot(tbt.turns, tbt.Nb, alpha=0.7, lw=1.5, label=string_array[i])

        ax1.set_xlabel('Turns')
        ax2.set_xlabel('Turns')
        ax3.set_xlabel('Turns')
        ax1.set_ylabel(r'$\varepsilon_{x}^{n}$ [$\mu$m]')
        ax2.set_ylabel(r'$\varepsilon_{y}^{n}$ [$\mu$m]')
        ax3.set_ylabel(r'$N_{b}$')
        ax1.legend(fontsize=12)
        f.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        f.savefig('main_plots/result_multiple_trackings.png', dpi=250)
        plt.show()


    def generate_line(self, 
                  add_non_linear_magnet_errors=False, 
                  add_aperture=True,
                  beta_beat=None, 
                  harmonic_nb = 4653,
                  Qy_frac: int = 25,
                  minimum_aperture_to_remove=0.025,
                  deferred_expressions=False,
                  )->xt.Line:
        """

        Generate SPS lines with fixed transverse aperture, longitudinally limitRect for bucket, beta-beat
        and magnet errors

        Parameters:
        -----------
        Qy_frac : int
            fractional part of vertical tune
        add_non_linear_magnet_errors : bool
            whether to add line with non-linear chromatic errors
        add_aperture : bool
            whether to include aperture for SPS
        beta_beat : float
            relative beta beat, i.e. relative difference between max beta function and max original beta function
        harmonic_nb : int
            harmonic used for SPS RF system
        ibs_step : int
            turn interval at which to recalculate IBS growth rates
        Qy_frac : int
            fractional part of vertical tune, e.g. "19" for 26.19
        minimum_aperture_to_remove : float 
            minimum threshold of horizontal SPS aperture to remove, default is 0.025 (can also be set to None)
            as faulty IPM aperture has 0.01 m, which is too small
        deferred_expressions : bool
            whether to use deferred expressions while importing madx sequence into xsuite

        Returns:
        --------
        xt.Line, str
        """
        # Update vertical tune if changed
        self.qy0 = int(self.qy0) + Qy_frac / 100

        # Get SPS Pb line - with aperture and non-linear magnet errors if desired
        sps = SPS_sequence_maker()
        line, _ = sps.load_xsuite_line_and_twiss(Qy_frac=Qy_frac, add_aperture=add_aperture, beta_beat=beta_beat,
                                                   add_non_linear_magnet_errors=add_non_linear_magnet_errors)
                
        if minimum_aperture_to_remove is not None:
            line = sps.remove_aperture_below_threshold(line, minimum_aperture_to_remove)

        # Add longitudinal limit rectangle - to kill particles that fall out of bucket
        bucket_length = line.get_length()/harmonic_nb
        line.unfreeze() # if you had already build the tracker
        line.append_element(element=xt.LongitudinalLimitRect(min_zeta=-bucket_length/2, max_zeta=bucket_length/2), name='long_limit')

        # Generate name of file
        def_exp_str = '_deferred_exp' if deferred_expressions else ''
        BB_string = '_{}_percent_beta_beat'.format(int(beta_beat*100)) if beta_beat is not None else ''
        err_str = '_with_magnet_errors' if add_non_linear_magnet_errors else ''
        fname = 'SPS_2021_Pb_Qydot{}{}{}{}.json'.format(Qy_frac, def_exp_str, BB_string, err_str)

        return line, fname
    

    def save_lines_for_all_cases(self, output_folder : str ='lines', also_save_lines_with_deferred_expressions=True):
        """
        Generate lines for all cases of SPS flat bottom tracking: magnet errors, beta-beating, etc
        Used for instance on HTCondor where input file needs to be provided
        """
        os.makedirs(output_folder, exist_ok=True)

        # Load ideal lattice, and with BB + magnet errors
        line_ideal, f_ideal = self.generate_line(add_aperture=True, beta_beat=None, add_non_linear_magnet_errors=False)
        line_ideal_def_exp, f_ideal_def_exp = self.generate_line(add_aperture=True, beta_beat=None, add_non_linear_magnet_errors=False,
                                                           deferred_expressions=True)
        line_bb, f_bb = self.generate_line(add_aperture=True, beta_beat=0.1, add_non_linear_magnet_errors=True)
        line_bb_def_exp, f_bb_def_exp = self.generate_line(add_aperture=True, beta_beat=0.1, add_non_linear_magnet_errors=True,
                                                           deferred_expressions=True)
        line_ideal_dot19, f_ideal_dot19 = self.generate_line(Qy_frac=19, add_aperture=True, beta_beat=None, add_non_linear_magnet_errors=False)
        line_ideal_def_exp_dot19, f_ideal_def_exp_dot19 = self.generate_line(Qy_frac=19, add_aperture=True, beta_beat=None, add_non_linear_magnet_errors=False,
                                                           deferred_expressions=True)
        line_bb_dot19, f_bb_dot19 = self.generate_line(Qy_frac=19, add_aperture=True, beta_beat=0.1, add_non_linear_magnet_errors=True)
        line_bb_def_exp_dot19, f_bb_def_exp_dot19 = self.generate_line(Qy_frac=19, add_aperture=True, beta_beat=0.1, add_non_linear_magnet_errors=True,
                                                           deferred_expressions=True)
        

        lines = [line_ideal, line_bb, line_ideal_dot19, line_bb_dot19]
        lines_def_exp = [line_ideal_def_exp, line_bb_def_exp, line_ideal_def_exp_dot19, line_bb_def_exp_dot19]
        str_names = [f_ideal, f_bb, f_ideal_dot19, f_bb_dot19]
        str_names_def_exp = [f_ideal_def_exp, f_bb_def_exp, f_ideal_def_exp_dot19, f_bb_def_exp_dot19]

        # Dump lines to json files
        for i, line in enumerate(lines):
            sps_fname = f'{output_folder}/{str_names[i]}'
            print(f'Saving {sps_fname}')
            with open(sps_fname, 'w') as fid:
                json.dump(line.to_dict(), fid, cls=xo.JEncoder)
        
        # Also save strings to 
        with open(f'{output_folder}/line_names.txt', 'w') as outfile:
            outfile.write('\n'.join(str(i) for i in str_names))
        
        if also_save_lines_with_deferred_expressions:
            for i, line in enumerate(lines_def_exp):
                sps_fname_def_exp = f'{output_folder}/{str_names_def_exp[i]}'
                print(f'Saving {sps_fname_def_exp}')
                with open(sps_fname_def_exp, 'w') as fid:
                    json.dump(line.to_dict(), fid, cls=xo.JEncoder)
            
            # Also save strings to 
            with open(f'{output_folder}/line_names_def_exp.txt', 'w') as outfile:
                outfile.write('\n'.join(str(i) for i in str_names_def_exp))


    def track_SPS_with_prepared_line(self, line : xt.Line,
                                        which_context='gpu',
                                        beamParams=None,
                                        install_SC_on_line=True, 
                                        SC_mode='frozen',
                                        use_Gaussian_distribution=True,
                                        apply_kinetic_IBS_kicks=False,
                                        ibs_step = 50,
                                        ):
        """
        Run full tracking at SPS flat bottom with prepared input line, returning pandas dataframe
        
        Parameters:
        ----------
        line : xt.Line
            input line on which to do tracking
        which_context : str
            'gpu' or 'cpu'
        Qy_frac : int
            fractional part of vertical tune
        beamParams : dataclass
            container of exn, eyn, Nb and sigma_z. Default 'None' will load nominal SPS beam parameters 
        install_SC_on_line : bool
            whether to install space charge
        SC_mode : str
            type of space charge - 'frozen' (recommended), 'quasi-frozen' or 'PIC'
        use_Gaussian_distribution : bool
            whether to use Gaussian particle distribution for tracking
        add_kinetic_IBS_kicks : bool
            whether to apply kinetic kicks from xibs 
        ibs_step : int
            turn interval at which to recalculate IBS growth rates

        Returns:
        --------
        pd.DataFrame
        """
        # Initial settings for GPU device 
        gpu_device = 0

        # If specific beam parameters are not provided, load default SPS beam parameters
        if beamParams is None:
            beamParams = BeamParameters_SPS
        print('Beam parameters:', beamParams)

        # Select relevant context
        if which_context=='gpu':
            context = xo.ContextCupy()
        elif which_context=='cpu':
            context = xo.ContextCpu(omp_num_threads='auto')
        else:
            raise ValueError('Context is either "gpu" or "cpu"')

        line.build_tracker(_context=context)
        twiss = line.twiss()

        # Generate particles object to track    
        particles = self.generate_particles(line=line, context=context, use_Gaussian_distribution=use_Gaussian_distribution,
                                            beamParams=beamParams)

        # Initialize the dataclasses and store the initial values
        tbt = Records.init_zeroes(self.num_turns)
        tbt.update_at_turn(0, particles, twiss)

        ######### IBS kinetic kicks #########
        if apply_kinetic_IBS_kicks:
            beamparams = BeamParameters.from_line(line, n_part=beamParams.Nb)
            opticsparams = OpticsParameters.from_line(line) # read from line without space  charge
            IBS = KineticKickIBS(beamparams, opticsparams)
            kinetic_kick_coefficients = IBS.compute_kick_coefficients(particles)
            print(kinetic_kick_coefficients)

        # Install SC and build tracker
        if install_SC_on_line:
            fma_sps = FMA()
            line = fma_sps.install_SC_and_get_line(line, beamParams, mode=SC_mode, optimize_for_tracking=True, context=context)
            print('Installed space charge on line\n')

        # Start tracking 
        for turn in range(1, self.num_turns):
            
            if turn % self.turn_print_interval == 0:
                print('Tracking turn {}'.format(turn))            

            ########## IBS -> Potentially re-compute the ellitest_parts integrals and IBS growth rates #########
            if apply_kinetic_IBS_kicks and ((turn % ibs_step == 0) or (turn == 1)):
                
                # We compute from values at the previous turn
                kinetic_kick_coefficients = IBS.compute_kick_coefficients(particles)
                print(
                    "\n" + "=" * 60 + "\n",
                    f"Turn {turn:d}: re-computing growth rates and kick coefficients\n",
                    kinetic_kick_coefficients,
                    "\n" + "=" * 60,
                )
                
            ########## ----- Apply IBS Kick if desired ----- ##########
            if apply_kinetic_IBS_kicks:
                IBS.apply_ibs_kick(particles)
            
            # ----- Track and update records for tracked particles ----- #
            line.track(particles, num_turns=1)
            tbt.update_at_turn(turn, particles, twiss)

        # Make parquet file from dictionary
        tbt_dict = tbt.to_dict()
        df = pd.DataFrame(tbt_dict)
        
        return df