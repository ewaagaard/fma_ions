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
from .helpers import Records, Records_Growth_Rates, _bunch_length, _geom_epsx, _geom_epsy, _sigma_delta
from .tune_ripple import Tune_Ripple_SPS

from xibs.inputs import BeamParameters, OpticsParameters
from xibs.kicks import KineticKickIBS
from xibs.analytical import NagaitsevIBS

from pathlib import Path

import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import time
import cupy as cp

# Load default emittance measurement data from 2023_10_16
emittance_data_path = Path(__file__).resolve().parent.joinpath('../data/emittance_data/full_WS_data_SPS_2023_10_16.json').absolute()

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
                  ripple_plane='both',
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
        ripple_plane : str
            plane in which to add the tune ripple: 'X', 'Y' or 'both'
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
            beamParams = BeamParameters_SPS()
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
        particles.reorganize()

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
            kqf_vals, kqd_vals, _ = ripple.load_k_from_xtrack_matching(dq=dq, plane=ripple_plane)

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
        print('\nTracking time: {:.1f} s = {:.1f} h'.format(dt0, dt0/3600))
                
        # Make parquet file from dictionary
        if save_tbt_data:
            tbt_dict = tbt.to_dict()
            
            # Convert turns to seconds
            turns_per_sec = 1 / twiss.T_rev0
            seconds = self.num_turns / turns_per_sec # number of seconds we are running for
            tbt_dict['Seconds'] = np.linspace(0.0, seconds, num=int(self.num_turns))
            df = pd.DataFrame(tbt_dict)
        
            return df


    def load_tbt_data(self, output_folder=None) -> Records:
        """
        Loads numpy data if tracking has already been made
        """
        folder_path = '{}/'.format(output_folder) if output_folder is not None else ''

        # Read the parquet file
        tbt = pd.read_parquet('{}tbt.parquet'.format(folder_path))
        return tbt


    def plot_tracking_data(self, 
                           tbt : pd.DataFrame, 
                           include_emittance_measurements=False,
                           show_plot=False,
                           x_unit_in_turns=True):
        """
        Generates emittance plots from turn-by-turn (TBT) data class from simulations,
        compare with emittance measurements (default 2023-10-16) if desired.
        
        Parameters:
        tbt : pd.DataFrame
            dataframe containing the TBT data
        include_emittance_measurements : bool
            whether to include measured emittance or not
        show_plot : bool
            whether to run "plt.show()" in addtion
        x_units_in_turns : bool
            if True, x axis units will be turn, otherwise in seconds
        """

        # Convert measured emittances to turns if this unit is used, otherwise keep seconds
        
        if x_unit_in_turns:
            turns = np.arange(len(tbt.exn), dtype=int)             
            time_units = turns
        else:
            if 'Seconds' in tbt.index:
                time_units = tbt['Seconds']
            else:
                sps = SPS_sequence_maker()
                _, twiss = sps.load_xsuite_line_and_twiss()
                turns_per_sec = 1 / twiss.T_rev0
                seconds = self.num_turns / turns_per_sec # number of seconds we are running for
                tbt['Seconds'] = np.linspace(0.0, seconds, num=int(len(tbt.exn)))
                time_units = tbt['Seconds']
                
        # Load emittance measurements
        if include_emittance_measurements:
            if x_unit_in_turns:
                sps = SPS_sequence_maker()
                _, twiss = sps.load_xsuite_line_and_twiss()
                turns_per_sec = 1 / twiss.T_rev0
            
            full_data = self.load_emittance_data()
            time_units_x = (turns_per_sec * full_data['Ctime_X']) if x_unit_in_turns else full_data['Ctime_X']
            time_units_y = (turns_per_sec * full_data['Ctime_Y']) if x_unit_in_turns else full_data['Ctime_Y']

        # Emittances and bunch intensity 
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (14,5))

        ax1.plot(time_units, tbt.exn * 1e6, alpha=0.7, lw=1.5, label='Simulated')
        ax2.plot(time_units, tbt.eyn * 1e6, alpha=0.7, c='orange', lw=1.5, label='Simulated')
        ax3.plot(time_units, tbt.Nb, alpha=0.7, lw=1.5, c='r', label='Bunch intensity')

        if include_emittance_measurements:
            ax1.errorbar(time_units_x, 1e6 * np.array(full_data['N_avg_emitX']), yerr=1e6 * full_data['N_emitX_error'], 
                       color='blue', fmt="o", label="Measured")
            ax2.errorbar(time_units_y, 1e6 * np.array(full_data['N_avg_emitY']), yerr=1e6 * full_data['N_emitY_error'], 
                       color='darkorange', fmt="o", label="Measured")
            
        # Find min and max emittance values - set window limits 
        all_emit = np.concatenate((tbt.exn, tbt.eyn))
        if include_emittance_measurements:
            all_emit = np.concatenate((all_emit, np.array(full_data['N_avg_emitX']), np.array(full_data['N_avg_emitY'])))
        min_emit = 1e6 * np.min(all_emit)
        max_emit = 1e6 * np.max(all_emit)

        ax1.set_xlabel('Turns' if x_unit_in_turns else 'Time [s]')
        ax2.set_xlabel('Turns' if x_unit_in_turns else 'Time [s]')
        ax3.set_xlabel('Turns' if x_unit_in_turns else 'Time [s]')
        ax1.set_ylabel(r'$\varepsilon_{x}^{n}$ [$\mu$m]')
        ax2.set_ylabel(r'$\varepsilon_{y}^{n}$ [$\mu$m]')
        ax3.set_ylabel(r'$N_{b}$')
        ax3.set_xlabel('Turns')
        ax1.legend()
        ax2.legend()
        ax1.set_ylim(min_emit-0.08, max_emit+0.1)
        ax2.set_ylim(min_emit-0.08, max_emit+0.1)
        f.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        # Sigma_delta and bunch length
        f2, (ax12, ax22) = plt.subplots(1, 2, figsize = (8,4))

        ax12.plot(time_units, tbt.sigma_delta * 1e3, alpha=0.7, lw=1.5, label='$\sigma_{\delta}$')
        ax22.plot(time_units, tbt.bunch_length, alpha=0.7, lw=1.5, label='Bunch intensity')

        ax12.set_ylabel(r'$\sigma_{\delta}$')
        ax12.set_xlabel('Turns' if x_unit_in_turns else 'Time [s]')
        ax22.set_ylabel(r'$\sigma_{z}$ [m]')
        ax22.set_xlabel('Turns' if x_unit_in_turns else 'Time [s]')    
        f2.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        # Save figures
        os.makedirs(self.output_folder, exist_ok=True)
        f.savefig('{}/epsilon_Nb.png'.format(self.output_folder), dpi=250)
        f2.savefig('{}/sigma_z_and_delta.png'.format(self.output_folder), dpi=250)

        if show_plot:
            plt.show()
        plt.close()
        
        
    def load_tbt_data_and_plot(self, include_emittance_measurements=False, x_unit_in_turns=True, show_plot=False, output_folder=None):
        """Load already tracked data and plot"""
        try:
            tbt = self.load_tbt_data(output_folder=output_folder)
            self.plot_tracking_data(tbt, 
                                    include_emittance_measurements=include_emittance_measurements,
                                    x_unit_in_turns=x_unit_in_turns,
                                    show_plot=show_plot)
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


    def load_emittance_data(self, path : str = emittance_data_path) -> pd.DataFrame:
        """
        Loads measured emittance data from SPS MDs, processed with CCC miner
        https://github.com/ewaagaard/ccc_miner, returns pd.DataFrame
        
        Default date - 2023-10-16 with (Qx, Qy) = (26.3, 26.19) in SPS
        """
        
        # Load dictionary with emittance data
        try:
            with open(path, 'r') as fp:
                full_data = json.load(fp)
        except FileNotFoundError:
            print('\nFILE NOT FOUND - check input path!\n')
            return
        
        # Convert timestamp strings to datetime, and find spread
        full_data['TimestampX_datetime'] = pd.to_datetime(full_data['UTC_timestamp_X'])
        full_data['TimestampY_datetime'] = pd.to_datetime(full_data['UTC_timestamp_Y'])
        
        full_data['N_emitX_error'] = np.std(full_data['N_emittances_X'], axis=1)
        full_data['N_emitY_error'] = np.std(full_data['N_emittances_Y'], axis=1)
        
        # Only keep the average emittances, not full emittance tables
        #del full_data['N_emittances_X'], full_data['N_emittances_Y']
        
        df = pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in full_data.items() ]))
        
        # Remove emittance data after ramping starts, i.e. from around 48.5 s
        df = df[df['Ctime_Y'] < 48.5]
        
        return df


    def run_analytical_vs_kinetic_emittance_evolution(self,
                                                      Qy_frac : int = 25,
                                                      which_context='cpu',
                                                      add_non_linear_magnet_errors=False, 
                                                      beta_beat=None, 
                                                      beamParams=None,
                                                      ibs_step : int = 50,
                                                      show_plot=False,
                                                      print_lost_particle_state=True,
                                                      plot_longitudinal_phase_space=True,
                                                      harmonic_nb = 4653,
                                                      extra_plot_string=''
                                                      ):
        """
        Propagate emittances of Nagaitsev analytical and kinetic formalism.
        Adapted from https://fsoubelet.github.io/xibs/gallery/demo_kinetic_kicks.html#sphx-glr-gallery-demo-kinetic-kicks-py
        
        Parameters:
        -----------
        which_context : str
            'gpu' or 'cpu'
        Qy_frac : int
            fractional part of vertical tune
        add_non_linear_magnet_errors : bool
            whether to add line with non-linear chromatic errors
        beta_beat : float
            relative beta beat, i.e. relative difference between max beta function and max original beta function
        beamParams : dataclass
            container of exn, eyn, Nb and sigma_z. Default 'None' will load nominal SPS beam parameters 
        ibs_step : int
            turn interval at which to recalculate IBS growth rates
        Qy_frac : int
            fractional part of vertical tune, e.g. "19" for 26.19
        print_lost_particle_state : bool
            whether to print the state of lost particles
        plot_longitudinal_phase_space
            whether to plot the final longitudinal particle distribution 
        """
        # Update vertical tune if changed
        self.qy0 = int(self.qy0) + Qy_frac / 100
        
        # If specific beam parameters are not provided, load default SPS beam parameters
        if beamParams is None:
            beamParams = BeamParameters_SPS()
        print('Beam parameters:', beamParams)

        # Select relevant context
        if which_context=='gpu':
            context = xo.ContextCupy()
        elif which_context=='cpu':
            context = xo.ContextCpu(omp_num_threads='auto')
        else:
            raise ValueError('Context is either "gpu" or "cpu"')

        # Get SPS Pb line - with aperture and non-linear magnet errors if desired
        sps = SPS_sequence_maker()
        line, twiss = sps.load_xsuite_line_and_twiss(Qy_frac=Qy_frac, add_aperture=False, beta_beat=beta_beat,
                                                   add_non_linear_magnet_errors=add_non_linear_magnet_errors)
        line.discard_tracker()
        line.build_tracker(_context=context)
                
        # Find bucket length
        bucket_length = line.get_length()/harmonic_nb
        max_zeta = bucket_length/2

        # Generate particles object to track    
        particles = self.generate_particles(line=line, context=context, beamParams=beamParams)

        ######### IBS kinetic kicks and analytical model #########
        beamparams = BeamParameters.from_line(line, n_part=beamParams.Nb)
        opticsparams = OpticsParameters.from_line(line) # read from line without space  charge
        IBS = KineticKickIBS(beamparams, opticsparams)
        NIBS = NagaitsevIBS(beamparams, opticsparams)

        # Initialize the dataclasses
        kicked_tbt = Records_Growth_Rates.init_zeroes(self.num_turns)
        analytical_tbt = Records_Growth_Rates.init_zeroes(self.num_turns)
         
        # Store the initial values
        kicked_tbt.update_at_turn(0, particles, twiss)
        analytical_tbt.update_at_turn(0, particles, twiss)
        
        
        # Calculate initial growth rates and initialize
        kinetic_kick_coefficients = IBS.compute_kick_coefficients(particles)
        growth_rates = NIBS.growth_rates(analytical_tbt.epsilon_x[0], analytical_tbt.epsilon_y[0], 
                                         analytical_tbt.sigma_delta[0], analytical_tbt.bunch_length[0])
        kicked_tbt.Tx[0] = kinetic_kick_coefficients.Kx
        kicked_tbt.Ty[0] = kinetic_kick_coefficients.Ky
        kicked_tbt.Tz[0] = kinetic_kick_coefficients.Kz
        analytical_tbt.Tx[0] = growth_rates.Tx
        analytical_tbt.Ty[0] = growth_rates.Ty
        analytical_tbt.Tz[0] = growth_rates.Tz
        
        
        # We loop here now
        time00 = time.time()
        for turn in range(1, self.num_turns):
            # ----- Potentially re-compute the IBS growth rates and kick coefficients ----- #
            if (turn % ibs_step == 0) or (turn == 1):
                print(f"Turn {turn:d}: re-computing diffusion and friction terms")
                # Compute kick coefficients from the particle distribution at this moment
                kinetic_kick_coefficients = IBS.compute_kick_coefficients(particles)
                # Compute analytical values from those at the previous turn
                growth_rates = NIBS.growth_rates(
                    analytical_tbt.epsilon_x[turn - 1],
                    analytical_tbt.epsilon_y[turn - 1],
                    analytical_tbt.sigma_delta[turn - 1],
                    analytical_tbt.bunch_length[turn - 1],
                )

            if turn % self.turn_print_interval == 0:
                print('\nTracking turn {}'.format(turn))       
                
            # Add the growth rates
            kicked_tbt.Tx[turn] = kinetic_kick_coefficients.Kx
            kicked_tbt.Ty[turn] = kinetic_kick_coefficients.Ky
            kicked_tbt.Tz[turn] = kinetic_kick_coefficients.Kz
            analytical_tbt.Tx[turn] = growth_rates.Tx
            analytical_tbt.Ty[turn] = growth_rates.Ty
            analytical_tbt.Tz[turn] = growth_rates.Tz
        
            # ----- Manually Apply IBS Kick and Track Turn ----- #
            IBS.apply_ibs_kick(particles)
            line.track(particles, num_turns=1)
        
            # ----- Update records for tracked particles ----- #
            kicked_tbt.update_at_turn(turn, particles, twiss)
        
            # ----- Compute analytical Emittances from previous turn values & update records----- #
            ana_emit_x, ana_emit_y, ana_sig_delta, ana_bunch_length = NIBS.emittance_evolution(
                analytical_tbt.epsilon_x[turn - 1],
                analytical_tbt.epsilon_y[turn - 1],
                analytical_tbt.sigma_delta[turn - 1],
                analytical_tbt.bunch_length[turn - 1],
            )
            analytical_tbt.epsilon_x[turn] = ana_emit_x
            analytical_tbt.epsilon_y[turn] = ana_emit_y
            analytical_tbt.sigma_delta[turn] = ana_sig_delta
            analytical_tbt.bunch_length[turn] = ana_bunch_length
            
            if print_lost_particle_state and turn % self.turn_print_interval == 0:
                print('Particles out of bucket: {}'.format(len(particles.zeta[(particles.zeta < max_zeta) & (particles.zeta > -max_zeta)])))
        
        time01 = time.time()
        dt0 = time01-time00
        print('\nTracking time: {:.1f} s = {:.1f} h'.format(dt0, dt0/3600))
        
        # Save the data
        os.makedirs('output_data_and_plots_{}'.format(which_context), exist_ok=True)
        df_kick = kicked_tbt.to_pandas()
        df_analytical = analytical_tbt.to_pandas()
        
        # Plot the results
        turns = np.arange(self.num_turns, dtype=int)  # array of turns
        fig, axs = plt.subplot_mosaic([["epsx", "epsy"], ["sigd", "bl"]], sharex=True, figsize=(15, 7))
        
        # Plot from tracked & kicked particles
        axs["epsx"].plot(turns, kicked_tbt.epsilon_x * 1e6, lw=2, label="Kinetic Kicks")
        axs["epsy"].plot(turns, kicked_tbt.epsilon_y * 1e6, lw=2, label="Kinetic Kicks")
        axs["sigd"].plot(turns, kicked_tbt.sigma_delta * 1e3, lw=2, label="Kinetic Kicks")
        axs["bl"].plot(turns, kicked_tbt.bunch_length * 1e3, lw=2, label="Kinetic Kicks")
        
        # Plot from analytical values
        axs["epsx"].plot(turns, analytical_tbt.epsilon_x * 1e6, lw=2.5, label="Analytical")
        axs["epsy"].plot(turns, analytical_tbt.epsilon_y * 1e6, lw=2.5, label="Analytical")
        axs["sigd"].plot(turns, analytical_tbt.sigma_delta * 1e3, lw=2.5, label="Analytical")
        axs["bl"].plot(turns, analytical_tbt.bunch_length * 1e3, lw=2.5, label="Analytical")
        
        # Axes parameters
        axs["epsx"].set_ylabel(r"$\varepsilon_x$ [$\mu$m]")
        axs["epsy"].set_ylabel(r"$\varepsilon_y$ [$\mu$m]")
        axs["sigd"].set_ylabel(r"$\sigma_{\delta}$ [$10^{-3}$]")
        axs["bl"].set_ylabel(r"Bunch length [mm]")
        
        for axis in (axs["epsy"], axs["bl"]):
            axis.yaxis.set_label_position("right")
            axis.yaxis.tick_right()
        
        for axis in (axs["sigd"], axs["bl"]):
            axis.set_xlabel("Turn Number")
        
        for axis in axs.values():
            axis.yaxis.set_major_locator(plt.MaxNLocator(3))
            axis.legend(loc=9, ncols=4)
        
        fig.align_ylabels((axs["epsx"], axs["sigd"]))
        fig.align_ylabels((axs["epsy"], axs["bl"]))
        plt.tight_layout()
        fig.savefig('output_data_and_plots_{}/analytical_vs_kinetic_emittance{}.png'.format(which_context, extra_plot_string), dpi=250)


        ############# GROWTH RATES #############
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (13,5))

        ax1.plot(turns, analytical_tbt.Tx, alpha=0.7, lw=1.5, label='Analytical Nagaitsev')
        ax1.plot(turns, kicked_tbt.Tx, label='Kinetic')
        ax2.plot(turns, analytical_tbt.Ty, alpha=0.7, lw=1.5, label='Analytical Nagaitsev')
        ax2.plot(turns, kicked_tbt.Ty, label='Kinetic')
        ax3.plot(turns, analytical_tbt.Tz, alpha=0.7, lw=1.5, label='Analytical Nagaitsev')
        ax3.plot(turns, kicked_tbt.Tz, label='Kinetic')

        ax1.set_ylabel(r'$T_{x}$')
        ax1.set_xlabel('Turns')
        ax2.set_ylabel(r'$T_{y}$')
        ax2.set_xlabel('Turns')
        ax3.set_ylabel(r'$T_{z}$')
        ax3.set_xlabel('Turns')
        ax1.legend(fontsize=12)
        plt.tight_layout()
        f.savefig('output_data_and_plots_{}/analytical_vs_kinetic_growth_rates{}.png'.format(which_context, extra_plot_string), dpi=250)

        
        # Plot longitudinal phase space if desired
        if plot_longitudinal_phase_space:
            fig3, ax3 = plt.subplots(1, 1, figsize = (10,5))
            fig3.suptitle(r'SPS PB. Last turn {} Longitudinal Phase Space. $\sigma_{{z}}$={} m'.format(self.num_turns, beamParams.sigma_z), fontsize=16)
            ax3.plot(particles.zeta.get() if which_context=='gpu' else particles.zeta, 
                     particles.delta.get()*1000 if which_context=='gpu' else particles.delta*1000, 
                     '.', markersize=3)
            ax3.axvline(x=max_zeta, color='r', linestyle='dashed')
            ax3.axvline(x=-max_zeta, color='r', linestyle='dashed')
            ax3.set_xlabel(r'$\zeta$ [m]')
            ax3.set_ylabel(r'$\delta$ [1e-3]')
            plt.tight_layout()
            fig3.savefig('output_data_and_plots_{}/SPS_Pb_ions_longitudinal_bucket_{}turns{}.png'.format(which_context, self.num_turns, extra_plot_string), dpi=250)

        if show_plot:
            plt.show()
        plt.close()