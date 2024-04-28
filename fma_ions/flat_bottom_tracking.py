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

from .sequences import PS_sequence_maker, BeamParameters_PS
from .sequences import SPS_sequence_maker, BeamParameters_SPS, BeamParameters_SPS_Oxygen
from .fma_ions import FMA
from .helpers import Records, Records_Growth_Rates, Full_Records, _bunch_length, _geom_epsx, _geom_epsy, _sigma_delta
from .tune_ripple import Tune_Ripple_SPS
from .longitudinal import generate_parabolic_distribution
from .longitudinal import generate_binomial_distribution_from_PS_extr

from xibs.inputs import BeamParameters, OpticsParameters
from xibs.kicks import KineticKickIBS
from xibs.analytical import NagaitsevIBS

from pathlib import Path

import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import gaussian_kde
import time

# Load default emittance measurement data from 2023_10_16
emittance_data_path = Path(__file__).resolve().parent.joinpath('../data/emittance_data/full_WS_data_SPS_2023_10_16.json').absolute()
Nb_data_path = Path(__file__).resolve().parent.joinpath('../data/emittance_data/Nb_processed_SPS_2023_10_16.json').absolute()

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
    ion_inj_ctime : float = 0.725 # ion injection happens at this time in cycle, important for WS

    def generate_particles(self, line: xt.Line, context : xo.context, distribution_type='gaussian', beamParams=None,
                           engine=None, m=5.3, num_particles_linear_in_zeta=5, xy_norm_default=0.1) -> xp.Particles:
        """
        Generate xp.Particles object: matched Gaussian or longitudinally parabolic

        Parameters:
        -----------
        distribution_type : str
            'gaussian', 'parabolic', 'binomial' or 'linear_in_zeta'
        m : float
            binomial parameter to determine tail of parabolic distribution
        num_particles_linear_in_zeta : int
            number of equally spaced macroparticles linear in zeta
        xy_norm_default : float
            if building particles linear in zeta, what is the default normalized transverse coordinates (exact center not ideal for 
            if we want to study space charge and resonances)
        """
        if beamParams is None:
            beamParams = BeamParameters_SPS

        if distribution_type=='gaussian':
            particles = xp.generate_matched_gaussian_bunch(_context=context,
                num_particles=self.num_part, 
                total_intensity_particles=beamParams.Nb,
                nemitt_x=beamParams.exn, 
                nemitt_y=beamParams.eyn, 
                sigma_z= beamParams.sigma_z,
                particle_ref=line.particle_ref, 
                line=line,
                engine=engine)
        elif distribution_type=='parabolic':
            particles = generate_parabolic_distribution(
                num_particles=self.num_part, 
                total_intensity_particles=beamParams.Nb,
                nemitt_x=beamParams.exn, 
                nemitt_y=beamParams.eyn, 
                sigma_z= beamParams.sigma_z,
                line=line, _context=context)
        elif distribution_type=='binomial':
            # Also calculate SPS separatrix for plotting
            particles, self._zeta_separatrix, self._delta_separatrix = generate_binomial_distribution_from_PS_extr(num_particles=self.num_part,
                                                                    nemitt_x=beamParams.exn, nemitt_y=beamParams.eyn,
                                                                    sigma_z=beamParams.sigma_z_binomial, total_intensity_particles=beamParams.Nb,
                                                                    line=line, m=m, return_separatrix_coord=True)
        elif distribution_type=='linear_in_zeta':
            
            # Find suitable zeta range - make linear spacing between close to center of RF bucket and to separatrix
            zetas = np.linspace(0.05, 0.7, num=num_particles_linear_in_zeta)

            # Build the particle object
            particles = xp.build_particles(line = line, particle_ref = line.particle_ref,
                                        x_norm=xy_norm_default, y_norm=xy_norm_default, delta=0.0, zeta=zetas,
                                        nemitt_x = beamParams.exn, nemitt_y = beamParams.eyn, _context=context)
        else:   
            raise ValueError('Only Gaussian, parabolic and binomial distributions are implemented!')
            
        return particles


    def track_SPS(self, 
                  save_tbt_data=True, 
                  ion_type='Pb',
                  which_context='cpu',
                  add_non_linear_magnet_errors=False, 
                  add_aperture=True,
                  beta_beat=None, 
                  beamParams=None,
                  install_SC_on_line=True, 
                  SC_mode='frozen',
                  distribution_type='gaussian',
                  apply_kinetic_IBS_kicks=False,
                  harmonic_nb = 4653,
                  auto_recompute_ibs_coefficients=False,
                  auto_recompute_coefficients_percent=5,
                  ibs_step = 5000,
                  Qy_frac: int = 19,
                  print_lost_particle_state=True,
                  minimum_aperture_to_remove=0.025,
                  add_tune_ripple=False,
                  ripple_plane='both',
                  dq=0.01,
                  ripple_freq=50,
                  engine=None,
                  save_full_particle_data=False,
                  full_particle_data_interval=None,
                  update_particles_and_sc_for_binomial=False,
                  plane_for_beta_beat='Y'
                  ):
        """
        Run full tracking at SPS flat bottom
        
        Parameters:
        ----------
        save_tbt: bool
            whether to return turn-by-turn data from tracking
        ion_type : str
            which ion to use: currently available are 'Pb' and 'O'
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
        distribution_type : str
            'gaussian' or 'parabolic' or 'binomial': particle distribution for tracking
        add_kinetic_IBS_kicks : bool
            whether to apply kinetic kicks from xibs 
        harmonic_nb : int
            harmonic used for SPS RF system
        auto_recompute_ibs_coefficients : bool
            whether to automatically recalculate IBS coefficients for a given emittance change
        auto_recompute_coefficients_percent : float
            relative emittance change after which to recompute 
        ibs_step : int
            if auto_recompute_ibs_coefficients=False, the turn interval at which to recalculate IBS growth rates
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
        engine : str
            if Gaussian distribution, which single RF harmonic matcher engine to use. None, 'pyheadtail' or 'single-rf-harmonic'.
        save_full_particle_data : bool
            whether to save all particle phase space data (default False), else only ensemble properties
        full_particle_data_interval : int
            starting from turn 1, interval between which to save full particle data. Default 'None' will save only first or last turn
        update_particles_and_sc_for_binomial : bool
            whether to "pre-track" particles for 50 turns if binomial distribution with particles outside RF bucket is generated, 
            then updating space charge to new distribution
        plane_for_beta_beat : str
            plane in which beta-beat exists: 'X', 'Y' (default) or 'both'

        Returns:
        --------
        None
        """
        # Update vertical tune if changed
        self.qy0 = int(self.qy0) + Qy_frac / 100

        # If specific beam parameters are not provided, load default SPS beam parameters - for Pb or O
        if beamParams is None:
            if ion_type=='Pb':
                beamParams = BeamParameters_SPS()
            if ion_type=='O':
                beamParams = BeamParameters_SPS_Oxygen()
            if distribution_type == 'binomial':
                beamParams.Nb = beamParams.Nb / 0.9108 # assume 8% of particles are lost outside of PS bucket, have to compensate for comparison
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

        # Get SPS Pb line - select ion
        if ion_type=='Pb':
            sps = SPS_sequence_maker()
        elif ion_type=='O':
            sps = SPS_sequence_maker(ion_type='O', Q_PS=4., Q_SPS=8., m_ion=15.9949) 
        else:
            raise ValueError('Only Pb and O ions implemented so far!')
            
        # Extract line with aperture, beta-beat and non-linear magnet errors if desired
        line, twiss = sps.load_xsuite_line_and_twiss(Qy_frac=Qy_frac, add_aperture=add_aperture, beta_beat=beta_beat,
                                                   add_non_linear_magnet_errors=add_non_linear_magnet_errors, 
                                                   deferred_expressions=load_line_with_deferred_expressions,
                                                   plane=plane_for_beta_beat)
                
        if minimum_aperture_to_remove is not None:
            line = sps.remove_aperture_below_threshold(line, minimum_aperture_to_remove)

        # Add longitudinal limit rectangle - to kill particles that fall out of bucket
        bucket_length = line.get_length()/harmonic_nb
        line.unfreeze() # if you had already build the tracker
        line.append_element(element=xt.LongitudinalLimitRect(min_zeta=-bucket_length/2, max_zeta=bucket_length/2), name='long_limit')
        line.build_tracker(_context=context)

        # Generate particles object to track    
        particles = self.generate_particles(line=line, context=context, distribution_type=distribution_type,
                                            beamParams=beamParams, engine=engine)
        particles.reorganize()


        ######### IBS kinetic kicks #########
        if apply_kinetic_IBS_kicks:
            beamparams = BeamParameters.from_line(line, n_part=beamParams.Nb)
            opticsparams = OpticsParameters.from_line(line) # read from line without space  charge
            if auto_recompute_ibs_coefficients:
                IBS = KineticKickIBS(beamparams, opticsparams, auto_recompute_coefficients_percent=auto_recompute_coefficients_percent)
                print('\nAutomatic IBS coefficient recomputation when change exceeds {} percent\n'.format(auto_recompute_coefficients_percent))
            else: 
                IBS = KineticKickIBS(beamparams, opticsparams)
                print('\nFixed IBS coefficient recomputation at interval = {} steps\n'.format(ibs_step))
            kinetic_kick_coefficients = IBS.compute_kick_coefficients(particles)
            print(kinetic_kick_coefficients)

        # Install SC and build tracker - optimize line if line variables for tune ripple not needed
        if install_SC_on_line:
            fma_sps = FMA()
            line = fma_sps.install_SC_and_get_line(line, beamParams, mode=SC_mode, optimize_for_tracking=(not add_tune_ripple), 
                                                   distribution_type=distribution_type, context=context)
            print('Installed space charge on line\n')
            
        # If distribution is binimial, pre-track particles and removed killed ones, and update SC parameters
        if distribution_type=='binomial' and update_particles_and_sc_for_binomial:
            
            # First tracking particles for 50 turns, remove all outside RF bucket
            print('\nStart pre-tracking of binomial...')
            for turn in range(1, 50):
                line.track(particles, num_turns=1)
            print('Finished pre-tracking: most common lost code: "-{}" for {} particles out of {} lost in total'.format(np.bincount(np.abs(particles.state[particles.state <= 0])).argmax(),
                                                                                                  np.max(np.bincount(np.abs(particles.state[particles.state <= 0]))),
                                                                                                  len(particles.state[particles.state <= 0])))
            particles.hide_lost_particles() # remove lost particles
            
            # Re-install space charge
            if install_SC_on_line:
                # Build a line without spacecharge (recycling the track kernel)
                line = line.filter_elements(exclude_types_starting_with='SpaceCh')
                
                # Re-install space charge, but with new parameters
                beamParams_updated = beamParams # copy old beam parameters
                beamParams_updated.sigma_z_binomial = np.std(particles.zeta[particles.state > 0])  # update Binomial RMS bunch length
                beamParams_updated.Nb = particles.weight[particles.state > 0][0]*len(particles.x[particles.state > 0])
                beamParams_updated.exn = np.float64(_geom_epsx(particles, twiss) * particles.beta0[0] * particles.gamma0[0])
                beamParams_updated.eyn = np.float64(_geom_epsy(particles, twiss) * particles.beta0[0] * particles.gamma0[0])
                
                # Install space charge and build tracker
                line = fma_sps.install_SC_and_get_line(line, beamParams_updated, mode=SC_mode, optimize_for_tracking=(not add_tune_ripple), 
                                                       distribution_type=distribution_type, context=context)
                print('Re-built space charge with updated binomial beam parameters: {}'.format(beamParams_updated))

        # Add tune ripple
        if add_tune_ripple:
            turns_per_sec = 1/twiss['T_rev0']
            ripple_period = int(turns_per_sec/ripple_freq)  # number of turns particle makes during one ripple oscillation
            ripple = Tune_Ripple_SPS(Qy_frac=Qy_frac, beta_beat=beta_beat, num_turns=self.num_turns, ripple_period=ripple_period)
            kqf_vals, kqd_vals, _ = ripple.load_k_from_xtrack_matching(dq=dq, plane=ripple_plane)

        # Initialize the dataclasses and store the initial values
        if not save_full_particle_data:
            tbt = Records.init_zeroes(self.num_turns)  # only emittances and bunch intensity
        else:
            # First find the interval at which to save data
            if full_particle_data_interval is None:
                full_data_ind = np.array([0, self.num_turns-1])
            else:
                full_data_ind = np.arange(0, self.num_turns, full_particle_data_interval)
                if not (self.num_turns-1 in full_data_ind):  # at last turn if in index array
                    full_data_ind  = np.append(full_data_ind, self.num_turns-1)
            tbt = Full_Records.init_zeroes(len(particles.x[particles.state > 0]), len(full_data_ind), 
                                           which_context=which_context, full_data_turn_ind=full_data_ind) # full particle data
        tbt.update_at_turn(0, particles, twiss)

        # Start tracking 
        time00 = time.time()
        for turn in range(1, self.num_turns):
            
            if turn % self.turn_print_interval == 0:
                print('\nTracking turn {}'.format(turn))            

            ########## IBS -> Potentially re-compute the ellitest_parts integrals and IBS growth rates #########
            if apply_kinetic_IBS_kicks and ((turn % ibs_step == 0) or (turn == 1)) and not auto_recompute_ibs_coefficients:
                
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

            # Update ensemble records or full records
            if not save_full_particle_data:
                tbt.update_at_turn(turn, particles, twiss)
            else:
                if turn in full_data_ind:
                    print(f'Updating full particle dictionary at turn {turn}')
                    update_ind = np.where(turn==full_data_ind)[0][0]
                    tbt.update_at_turn(update_ind, particles, twiss)

            if particles.state[particles.state <= 0].size > 0:
                if print_lost_particle_state and turn % self.turn_print_interval == 0:
                    print('Lost particle state: most common code: "-{}" for {} particles out of {} lost in total'.format(np.bincount(np.abs(particles.state[particles.state <= 0])).argmax(),
                                                                                                          np.max(np.bincount(np.abs(particles.state[particles.state <= 0]))),
                                                                                                          len(particles.state[particles.state <= 0])))
        
        time01 = time.time()
        dt0 = time01-time00
        print('\nTracking time: {:.1f} s = {:.1f} h'.format(dt0, dt0/3600))

        if apply_kinetic_IBS_kicks and auto_recompute_ibs_coefficients:
            print('\nNumber of times auto-recomputed growth rates: {}\n'.format(IBS._number_of_coefficients_computations))
                
        # Make parquet file from dictionary
        if save_tbt_data:
            
            # If not full particle data is saved, return pandas version of  TBT dictionary with particle data
            if not save_full_particle_data:
                tbt_dict = tbt.to_dict()
                # Convert turns to seconds
                turns_per_sec = 1 / twiss.T_rev0
                seconds = self.num_turns / turns_per_sec # number of seconds we are running for
                tbt_dict['Seconds'] = np.linspace(0.0, seconds, num=int(self.num_turns))
                tbt = pd.DataFrame(tbt_dict)
            
            return tbt


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
            print('Set time units to turns')
        else:
            if 'Seconds' in tbt.columns:
                time_units = tbt['Seconds']
            else:
                sps = SPS_sequence_maker()
                _, twiss = sps.load_xsuite_line_and_twiss()
                turns_per_sec = 1 / twiss.T_rev0
                seconds = len(tbt.exn) / turns_per_sec # number of seconds we are running for
                tbt['Seconds'] = np.linspace(0.0, seconds, num=int(len(tbt.exn)))
                time_units = tbt['Seconds'].copy()
                print('Set time units to seconds')

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
        

    def plot_multiple_sets_of_tracking_data(self, output_str_array, string_array, compact_mode=False,
                                            include_emittance_measurements=False, x_unit_in_turns=True,
                                            bbox_to_anchor_position=(0.0, 1.3),
                                            labelsize = 20,
                                            ylim=None, ax_for_legend=2,
                                            distribution_type='gaussian',
                                            legend_font_size=11.5):
        """
        If multiple runs with turn-by-turn (tbt) data has been made, provide list with Records class objects and list
        of explaining string to generate comparative plots of emittances, bunch intensities, etc

        Parameters:
        ----------
        output_str_array : [outfolder, outfolder, ...]
            List containing string for outfolder tbt data
        string:_array : [str1, str2, ...]
            List containing strings to explain the respective tbt data objects (which parameters were used)
        compact_mode : bool
            whether to slim plot in more compact format 
        include_emittance_measurements : bool
            whether to include measured emittance or not
        x_units_in_turns : bool
            if True, x axis units will be turn, otherwise in seconds
        bbox_to_anchor_position : tuple
            x-y coordinates of relative plot position for legend
        labelsize : int
            labelsize for axes
        ylim : list
            lower and upper bounds for emittance plots, if None (default), automatic limits are set
        ax_for_legend : int
            which axis to use to place legend, either 1 or 2 (default is 2)
        distribution_type : str
            'gaussian' or 'parabolic' or 'binomial': particle distribution for tracking
        legend_font_size : int
            labelsize for legend
        """
        os.makedirs('main_plots', exist_ok=True)
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 18,
                "axes.titlesize": 18,
                "axes.labelsize": labelsize,
                "xtick.labelsize": 15,
                "ytick.labelsize": 15,
                "legend.fontsize": 15,
                "figure.titlesize": 20,
            }
        )

        # Load TBT data 
        tbt_array = []
        for output_folder in output_str_array:
            self.output_folder = output_folder
            tbt = self.load_tbt_data(output_folder)
            tbt['turns'] = np.arange(len(tbt.Nb), dtype=int)
            tbt_array.append(tbt)

        # If binomial distribution, find index corresponding to after 30 turns (when distribution has stabilized)
        if distribution_type=='binomial':
            ii = tbt['turns'] > 30
            print('\nSetting binomial turn index\n')
        else:
            ii = tbt['turns'] > -1 # select all turns
            print('\nGaussian beam - select all turns\n')

        # Convert measured emittances to turns if this unit is used, otherwise keep seconds
        if x_unit_in_turns:         
            time_units = tbt['turns']
            print('Set time units to turns')
        else:
            if 'Seconds' in tbt.columns:
                time_units = tbt['Seconds']
            else:
                sps = SPS_sequence_maker()
                _, twiss = sps.load_xsuite_line_and_twiss()
                turns_per_sec = 1 / twiss.T_rev0
                seconds = len(tbt.exn) / turns_per_sec # number of seconds we are running for
                time_units = np.linspace(0.0, seconds, num=int(len(tbt.exn))) 
                print('Set time units to seconds')

        # Load emittance measurements
        if include_emittance_measurements:
            if x_unit_in_turns:
                sps = SPS_sequence_maker()
                _, twiss = sps.load_xsuite_line_and_twiss()
                turns_per_sec = 1 / twiss.T_rev0
            
            full_data = self.load_emittance_data()
            time_units_x = (turns_per_sec * full_data['Ctime_X']) if x_unit_in_turns else full_data['Ctime_X']
            time_units_y = (turns_per_sec * full_data['Ctime_Y']) if x_unit_in_turns else full_data['Ctime_Y']

            df_Nb = self.load_Nb_data()
            time_Nb = (turns_per_sec * df_Nb['ctime']) if x_unit_in_turns else df_Nb['ctime']

        # Normal, or compact mode
        if compact_mode:
            # Emittances and bunch intensity 
            f = plt.figure(figsize = (6, 6))
            gs = f.add_gridspec(3, hspace=0, height_ratios= [1, 2, 2])
            (ax3, ax2, ax1) = gs.subplots(sharex=True, sharey=False)

            # Plot measurements, if desired                
            if include_emittance_measurements:
                ax3.plot(time_Nb, df_Nb['Nb'], color='blue', marker="o", ms=2.5, alpha=0.7, label="Measured")
            
            # Loop over the tbt records classes 
            for i, tbt in enumerate(tbt_array):
                ax1.plot(time_units, tbt.exn * 1e6, alpha=0.7, lw=1.5, label=string_array[i])
                ax2.plot(time_units, tbt.eyn * 1e6, alpha=0.7, lw=1.5, label=string_array[i])
                ax3.plot(time_units[ii], tbt.Nb[ii], alpha=0.7, lw=2.5, label=string_array[i])
                
            # Include wire scanner data - subtract ion injection cycle time
            if include_emittance_measurements:
                ax1.errorbar(time_units_x - self.ion_inj_ctime, 1e6 * np.array(full_data['N_avg_emitX']), yerr=1e6 * full_data['N_emitX_error'], 
                           color='blue', fmt="o", label="Measured")
                ax2.errorbar(time_units_y - self.ion_inj_ctime, 1e6 * np.array(full_data['N_avg_emitY']), yerr=1e6 * full_data['N_emitY_error'], 
                           color='blue', fmt="o", label="Measured")
                
            ax1.set_ylabel(r'$\varepsilon_{x}^{n}$ [$\mu$m rad]')
            ax2.set_ylabel(r'$\varepsilon_{y}^{n}$ [$\mu$m rad]')
            ax3.set_ylabel(r'$N_{b}$')
            #ax1.text(0.94, 0.94, 'X', color='darkgreen', fontsize=20, transform=ax1.transAxes)
            #ax2.text(0.02, 0.94, 'Y', color='darkgreen', fontsize=20, transform=ax2.transAxes)
            ax1.set_xlabel('Turns' if x_unit_in_turns else 'Time [s]')
            ax2.set_xlabel('Turns' if x_unit_in_turns else 'Time [s]')
            if ylim is not None:
                ax1.set_ylim(ylim[0], ylim[1])
                ax2.set_ylim(ylim[0], ylim[1])
            if ax_for_legend == 2:
                ax2.legend(fontsize=legend_font_size, loc='upper left', bbox_to_anchor=bbox_to_anchor_position)
            elif ax_for_legend == 1:
                ax1.legend(fontsize=legend_font_size, loc='upper left', bbox_to_anchor=bbox_to_anchor_position)
            
            for ax in f.get_axes():
                ax.label_outer()
            
            f.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            f.savefig('main_plots/result_multiple_trackings_compact.png', dpi=250)
            plt.show()
                #ax3.plot(tbt.turns, tbt.Nb, alpha=0.7, lw=1.5, label=string_array[i])
            
        else:
            # Emittances and bunch intensity 
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (14,5))
    
            # Loop over the tbt records classes 
            for i, tbt in enumerate(tbt_array):
                ax1.plot(tbt.turns, tbt.exn * 1e6, alpha=0.7, lw=1.5, label=string_array[i])
                ax2.plot(tbt.turns, tbt.eyn * 1e6, alpha=0.7, lw=1.5, label=string_array[i])
                ax3.plot(tbt.turns, tbt.Nb, alpha=0.7, lw=1.5, label=string_array[i])
    
            if include_emittance_measurements:
                ax1.errorbar(time_units_x, 1e6 * np.array(full_data['N_avg_emitX']), yerr=1e6 * full_data['N_emitX_error'], 
                           color='blue', fmt="o", label="Measured")
                ax2.errorbar(time_units_y, 1e6 * np.array(full_data['N_avg_emitY']), yerr=1e6 * full_data['N_emitY_error'], 
                           color='darkorange', fmt="o", label="Measured")
    
            ax1.set_xlabel('Turns')
            ax2.set_xlabel('Turns')
            ax3.set_xlabel('Turns')
            ax1.set_ylabel(r'$\varepsilon_{x}^{n}$ [$\mu$m]')
            ax2.set_ylabel(r'$\varepsilon_{y}^{n}$ [$\mu$m]')
            ax3.set_ylabel(r'$N_{b}$')
            ax1.legend(fontsize=10)
            f.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            f.savefig('main_plots/result_multiple_trackings.png', dpi=250)
            plt.show()

    def load_full_records_json(self, output_folder=None) -> Full_Records:
        """
        Loads json file with full particle data from tracking
        """
        folder_path = '{}/'.format(output_folder) if output_folder is not None else ''

        # Read the json file
        tbt = Full_Records.from_json("{}tbt.json".format(folder_path))

        return tbt


    def plot_normalized_phase_space_from_tbt(self, 
                                             output_folder=None, 
                                             include_density_map=True, 
                                             use_only_particles_killed_last=False,
                                             plot_min_aperture=True,
                                             min_aperture=0.025):
        """
        Generate normalized phase space in X and Y to follow particle distribution
        
        Parameters:
        -----------
        output_folder : str
            path to data. default is 'None', assuming then that data is in the same directory
        include_density_map : bool
            whether to add color gradient of how tightly packed particles are
        use_only_particles_killed_last : bool
            whether to use the 'kill' index only based on particles killed in the last tracking run
        plot_min_aperture : bool
            whether to include line with minimum X and Y aperture along machine
        min_aperture : float
            default minimum aperture in X and Y (TIDP is collimator limiting y-plane at s=463m)
        """
        tbt_dict = self.load_full_records_json(output_folder=output_folder)

        # Output directory
        os.makedirs('output_plots', exist_ok=True)
        
        # Final dead and alive indices
        alive_ind_final = tbt_dict.state[:, -1] > 0
        dead_ind_final = tbt_dict.state[:, -1] < 1
        dead_ind_lost_in_last_round =  (tbt_dict.state[:, -2] > 0) & (tbt_dict.state[:, -1] < 1)  # particles alive in last tracking round but finally dead
        
        if use_only_particles_killed_last:
            dead_ind_final = dead_ind_lost_in_last_round
            alive_ind_final = np.invert(dead_ind_final)
            extra_ind = '_killed_in_last_round'
        else:
            extra_ind = ''

        # Convert to normalized phase space
        sps = SPS_sequence_maker()
        line, twiss = sps.load_xsuite_line_and_twiss(add_aperture=True)
        
        # Check minimum aperture and plot
        if plot_min_aperture:
            line = sps.remove_aperture_below_threshold(line, min_aperture)
            x_ap, y_ap, a = sps.print_smallest_aperture(line)
            ind_x, ind_y = np.argmin(x_ap), np.argmin(y_ap)
            
            # Find beta functions at these points
            df = twiss.to_pandas()
            betx_min_ap = df.iloc[np.abs(df['s'] - a.iloc[ind_x].s).argmin()].betx
            bety_min_ap = df.iloc[np.abs(df['s'] - a.iloc[ind_y].s).argmin()].bety
            
            # Min aperture - convert to normalized coord
            min_aperture_norm = np.array([x_ap[ind_x] / np.sqrt(betx_min_ap), y_ap[ind_y] / np.sqrt(bety_min_ap)])
        
        X = tbt_dict.x / np.sqrt(twiss['betx'][0]) 
        PX = twiss['alfx'][0] / np.sqrt(twiss['betx'][0]) * tbt_dict.x + np.sqrt(twiss['betx'][0]) * tbt_dict.px
        Y = tbt_dict.y / np.sqrt(twiss['bety'][0]) 
        PY = twiss['alfy'][0] / np.sqrt(twiss['bety'][0]) * tbt_dict.y + np.sqrt(twiss['bety'][0]) * tbt_dict.py
        
        planes = ['X', 'Y']
        Us = [X, Y]
        PUs = [PX, PY]
        
        # Iterate over X and Y
        for i, U in enumerate(Us):
            PU = PUs[i]
            
            ### First plot first and last turn of normalized phase space
            # Generate histograms in all planes to inspect distribution
            bin_heights, bin_borders = np.histogram(U[:, 0], bins=60)
            bin_widths = np.diff(bin_borders)
            bin_centers = bin_borders[:-1] + bin_widths / 2
            #bin_heights = bin_heights/np.max(bin_heights) # normalize bin heights
            
            # Only plot final alive particles
            bin_heights2, bin_borders2 = np.histogram(U[alive_ind_final, -1], bins=60)
            bin_widths2 = np.diff(bin_borders2)
            bin_centers2 = bin_borders2[:-1] + bin_widths2 / 2
            #bin_heights2 = bin_heights2/np.max(bin_heights2) # normalize bin heights
            
            # Plot alive particles sorted by density
            if include_density_map:
                # First turn
                x, y = U[alive_ind_final, 0], PU[alive_ind_final, 0]
                xy = np.vstack([x,y]) # Calculate the point density
                z = gaussian_kde(xy)(xy)
                idx = z.argsort()  # Sort the points by density, so that the densest points are plotted last
                x, y, z = x[idx], y[idx], z[idx]
                
                # Last turn
                x2, y2 = U[alive_ind_final, -1], PU[alive_ind_final, -1]
                xy2 = np.vstack([x2, y2]) # Calculate the point density
                z2 = gaussian_kde(xy2)(xy2)
                idx2 = z2.argsort()  # Sort the points by density, so that the densest points are plotted last
                x2, y2, z2 = x2[idx2], y2[idx2], z2[idx2]
    
            # Plot longitudinal phase space, initial and final state
            fig, ax = plt.subplots(3, 1, figsize = (10, 12), sharex=True)
            
            # Plot initial particles
            if include_density_map:
                ax[0].scatter(x, y, c=z, cmap='cool', s=2, label='Alive')
            else:   
                ax[0].plot(U[alive_ind_final, 0], PU[alive_ind_final, 0], '.', 
                    color='blue', markersize=3.6, label='Alive')
            ax[0].plot(U[dead_ind_final, 0], PU[dead_ind_final, 0], '.', 
                    color='darkred', markersize=3.6, label='Finally dead')
            if plot_min_aperture:
                ax[0].axvline(x=min_aperture_norm[i], ls='-', color='red', alpha=0.7, label='Min. aperture')
                ax[0].axvline(x=-min_aperture_norm[i], ls='-', color='red', alpha=0.7, label=None)
    
            # Plot final particles
            if include_density_map:
                ax[1].scatter(x2, y2, c=z2, cmap='cool', s=2, label='Alive')
            else:   
                ax[1].plot(U[alive_ind_final, -1], PU[alive_ind_final, -1], '.', 
                    color='blue', markersize=3.6, label='Alive')
            ax[1].plot(U[dead_ind_final, -1], PU[dead_ind_final, -1], '.', 
                    color='darkred', markersize=3.6, label='Finally dead')
            if plot_min_aperture:
                ax[1].axvline(x=min_aperture_norm[i], ls='-', color='red', alpha=0.7, label='Min. aperture')
                ax[1].axvline(x=-min_aperture_norm[i], ls='-', color='red', alpha=0.7, label=None)
            ax[1].legend(loc='upper right', fontsize=13)
            
            # Plot initial and final particle distribution
            ax[2].bar(bin_centers, bin_heights, width=bin_widths, alpha=1.0, color='darkturquoise', label='Initial')
            ax[2].bar(bin_centers2, bin_heights2, width=bin_widths2, alpha=0.5, color='lime', label='Final (alive)')
            ax[2].legend(loc='upper right', fontsize=13)
            
            # Adjust axis limits and plot turn
            x_lim = np.max(min_aperture_norm) + 0.001
            ax[0].set_ylim(-x_lim, x_lim)
            ax[0].set_xlim(-x_lim, x_lim)
            ax[1].set_ylim(-x_lim, x_lim)

            
            ax[0].text(0.02, 0.91, 'Turn {}'.format(tbt_dict.full_data_turn_ind[0]+1), fontsize=15, transform=ax[0].transAxes)
            ax[1].text(0.02, 0.91, 'Turn {}'.format(tbt_dict.full_data_turn_ind[-1]+1), fontsize=15, transform=ax[1].transAxes)
                
            ax[2].set_xlabel(r'${}$'.format(planes[i]))
            ax[2].set_ylabel('Counts')
            ax[0].set_ylabel('$P{}$'.format(planes[i]))
            ax[1].set_ylabel('$P{}$'.format(planes[i]))
            plt.tight_layout()
            fig.savefig('output_plots/SPS_Pb_{}_phase_space.png'.format(planes[i]), dpi=250)
            plt.close()

    def plot_longitudinal_phase_space_trajectories(self, 
                                                   output_folder=None, 
                                                   include_sps_separatrix=True):
        """
        Plot color-coded trajectories in longitudinal phase space based on turns
        
        Parameters:
        -----------
        output_folder : str
            path to data. default is 'None', assuming then that data is in the same directory
        include_sps_separatrix : bool
            whether to plot line of SPS RF seperatrix
        """
        tbt_dict = self.load_full_records_json(output_folder=output_folder)

        # Output directory
        os.makedirs('output_plots', exist_ok=True)
        
        # Get SPS zeta separatrix
        if include_sps_separatrix:
            sps = SPS_sequence_maker()
            sps_line, twiss = sps.load_xsuite_line_and_twiss()
            _, zeta_separatrix, delta_separatrix = generate_binomial_distribution_from_PS_extr(num_particles=50,
                                                                             nemitt_x= BeamParameters_SPS.exn, nemitt_y=BeamParameters_SPS.eyn,
                                                                             sigma_z=BeamParameters_SPS.sigma_z, total_intensity_particles=BeamParameters_SPS.Nb,
                                                                             line=sps_line, return_separatrix_coord=True)
            
        # Create a color map based on number of turns
        num_turns = len(tbt_dict.x[0])
        num_particles = len(tbt_dict.x)
        colors = cm.viridis(np.linspace(0, 1, num_turns))    
    
        # plot longitudinal phase space trajectories of all particles
        fig, ax = plt.subplots(1, 1, figsize = (8, 4.5))
        for i in range(num_particles):
            print(f'Plotting particle {i+1}')
            ax.scatter(tbt_dict.zeta[i, :], tbt_dict.delta[i, :] * 1e3, c=range(num_turns), marker='.')
        if include_sps_separatrix:
            ax.plot(zeta_separatrix, delta_separatrix * 1e3, '-', color='red', alpha=0.7, label='SPS RF separatrix')
            ax.plot(zeta_separatrix, -delta_separatrix * 1e3, '-', color='red', alpha=0.7, label=None)
        ax.set_ylim(-1.4, 1.4)
        ax.set_xlim(-0.85, 0.85)
        ax.set_xlabel(r'$\zeta$ [m]')
        ax.set_ylabel(r'$\delta$ [1e-3]')

        # Adding color bar for the number of turns
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Number of Turns')

        plt.tight_layout()
        fig.savefig('output_plots/SPS_Pb_longitudinal_trajectories.png', dpi=250)




    def plot_longitudinal_phase_space_all_slices_from_tbt(self, 
                                                          output_folder=None, 
                                                          include_sps_separatrix=True,
                                                          include_density_map=True, 
                                                          use_only_particles_killed_last=False):
        """
        Generate longitudinal phase space plots for all turns where they have been recorded
        
        Parameters:
        -----------
        output_folder : str
            path to data. default is 'None', assuming then that data is in the same directory
        include_sps_separatrix : bool
            whether to plot line of SPS RF seperatrix
        include_density_map : bool
            whether to add color gradient of how tightly packed particles are
        use_only_particles_killed_last : bool
            whether to use the 'kill' index only based on particles killed in the last tracking run
        """
        tbt_dict = self.load_full_records_json(output_folder=output_folder)

        # Output directory
        os.makedirs('output_plots', exist_ok=True)
        
        # Get SPS zeta separatrix
        if include_sps_separatrix:
            sps = SPS_sequence_maker()
            sps_line, twiss = sps.load_xsuite_line_and_twiss()
            _, zeta_separatrix, delta_separatrix = generate_binomial_distribution_from_PS_extr(num_particles=50,
                                                                             nemitt_x= BeamParameters_SPS.exn, nemitt_y=BeamParameters_SPS.eyn,
                                                                             sigma_z=BeamParameters_SPS.sigma_z, total_intensity_particles=BeamParameters_SPS.Nb,
                                                                             line=sps_line, return_separatrix_coord=True)
        # Final dead and alive indices
        alive_ind_final = tbt_dict.state[:, -1] > 0
        dead_ind_final = tbt_dict.state[:, -1] < 1
        dead_ind_lost_in_last_round =  (tbt_dict.state[:, -2] > 0) & (tbt_dict.state[:, -1] < 1)  # particles alive in last tracking round but finally dead
        
        if use_only_particles_killed_last:
            dead_ind_final = dead_ind_lost_in_last_round
            alive_ind_final = np.invert(dead_ind_final)
            extra_ind = '_killed_in_last_round'
        else:
            extra_ind = ''
        
        # Iterate over all turns that were recorded
        for i in range(len(tbt_dict.full_data_turn_ind)):
    
            print('Plotting data from turn {}'.format(tbt_dict.full_data_turn_ind[i]))
            # Plot longitudinal phase space, initial and final state
            plt.close()
            fig, ax = plt.subplots(1, 1, figsize = (8, 4.5))
            
            # Plot alive particles sorted by density
            if include_density_map:
                # First turn
                x, y = tbt_dict.zeta[alive_ind_final, i], tbt_dict.delta[alive_ind_final, i]*1000
                xy = np.vstack([x,y]) # Calculate the point density
                z = gaussian_kde(xy)(xy)
                idx = z.argsort()  # Sort the points by density, so that the densest points are plotted last
                x, y, z = x[idx], y[idx], z[idx]
                
            # Plot initial particles
            if include_density_map:
                ax.scatter(x, y, c=z, cmap='cool', s=2, label='Alive' if not use_only_particles_killed_last else 'Not killed in last turns')
            else:   
                ax.plot(tbt_dict.zeta[alive_ind_final, i], tbt_dict.delta[alive_ind_final, i]*1000, '.', 
                    color='blue', markersize=3.6, label='Alive' if not use_only_particles_killed_last else 'Not killed in last turns')
            ax.plot(tbt_dict.zeta[dead_ind_final, i], tbt_dict.delta[dead_ind_final, i]*1000, '.', 
                    color='darkred', markersize=3.6, label='Finally dead' if not use_only_particles_killed_last else 'Killed in last turns')
            if include_sps_separatrix:
                ax.plot(zeta_separatrix, delta_separatrix * 1e3, '-', color='red', alpha=0.7, label='SPS RF separatrix')
                ax.plot(zeta_separatrix, -delta_separatrix * 1e3, '-', color='red', alpha=0.7, label=None)
            ax.set_ylim(-1.4, 1.4)
            ax.set_xlim(-0.85, 0.85)
            ax.text(0.02, 0.91, 'Turn {}'.format(tbt_dict.full_data_turn_ind[i]), fontsize=15, transform=ax.transAxes)
            
            ax.legend(loc='upper right', fontsize=11)
            ax.set_xlabel(r'$\zeta$ [m]')
            ax.set_ylabel(r'$\delta$ [1e-3]')
            plt.tight_layout()
            fig.savefig('output_plots/SPS_Pb_longitudinal{}_turn_{}.png'.format(extra_ind, int(tbt_dict.full_data_turn_ind[i])), dpi=250)
            
        
    def plot_last_and_first_turn_longitudinal_phase_space_from_tbt(self, output_folder=None, include_sps_separatrix=False,
                                               include_density_map=True):
        """
        Generate longitudinal phase space plots from full particle tracking data
        
        Parameters:
        -----------
        output_folder : str
            path to data. default is 'None', assuming then that data is in the same directory
        include_sps_separatrix : bool
            whether to plot line of SPS RF seperatrix
        include_density_map : bool
            whether to add color gradient of how tightly packed particles are
        """
        
        tbt_dict = self.load_full_records_json(output_folder=output_folder)

        # Output directory
        os.makedirs('output_plots', exist_ok=True)
        
        # Get SPS zeta separatrix
        if include_sps_separatrix:
            sps = SPS_sequence_maker()
            sps_line, twiss = sps.load_xsuite_line_and_twiss()
            _, zeta_separatrix, delta_separatrix = generate_binomial_distribution_from_PS_extr(num_particles=50,
                                                                             nemitt_x= BeamParameters_SPS.exn, nemitt_y=BeamParameters_SPS.eyn,
                                                                             sigma_z=BeamParameters_SPS.sigma_z, total_intensity_particles=BeamParameters_SPS.Nb,
                                                                             line=sps_line, return_separatrix_coord=True)
        
        # Final dead and alive indices
        alive_ind_final = tbt_dict.state[:, -1] > 0
        dead_ind_final = tbt_dict.state[:, -1] < 1
        
        # Generate histograms in all planes to inspect distribution
        bin_heights, bin_borders = np.histogram(tbt_dict.zeta[:, 0], bins=60)
        bin_widths = np.diff(bin_borders)
        bin_centers = bin_borders[:-1] + bin_widths / 2
        #bin_heights = bin_heights/np.max(bin_heights) # normalize bin heights
        
        # Only plot final alive particles
        bin_heights2, bin_borders2 = np.histogram(tbt_dict.zeta[alive_ind_final, -1], bins=60)
        bin_widths2 = np.diff(bin_borders2)
        bin_centers2 = bin_borders2[:-1] + bin_widths2 / 2
        #bin_heights2 = bin_heights2/np.max(bin_heights2) # normalize bin heights
        
        # Plot alive particles sorted by density
        if include_density_map:
            # First turn
            x, y = tbt_dict.zeta[alive_ind_final, 0], tbt_dict.delta[alive_ind_final, 0]*1000
            xy = np.vstack([x,y]) # Calculate the point density
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()  # Sort the points by density, so that the densest points are plotted last
            x, y, z = x[idx], y[idx], z[idx]
            
            # Last turn
            x2, y2 = tbt_dict.zeta[alive_ind_final, -1], tbt_dict.delta[alive_ind_final, -1]*1000
            xy2 = np.vstack([x2, y2]) # Calculate the point density
            z2 = gaussian_kde(xy2)(xy2)
            idx2 = z2.argsort()  # Sort the points by density, so that the densest points are plotted last
            x2, y2, z2 = x2[idx2], y2[idx2], z2[idx2]

        # Plot longitudinal phase space, initial and final state
        fig, ax = plt.subplots(3, 1, figsize = (10, 12), sharex=True)
        
        # Plot initial particles
        if include_density_map:
            ax[0].scatter(x, y, c=z, cmap='cool', s=2, label='Alive')
        else:   
            ax[0].plot(tbt_dict.zeta[alive_ind_final, 0], tbt_dict.delta[alive_ind_final, 0]*1000, '.', 
                color='blue', markersize=3.6, label='Alive')
        ax[0].plot(tbt_dict.zeta[dead_ind_final, 0], tbt_dict.delta[dead_ind_final, 0]*1000, '.', 
                color='darkred', markersize=3.6, label='Finally dead')
        if include_sps_separatrix:
            ax[0].plot(zeta_separatrix, delta_separatrix * 1e3, '-', color='red', alpha=0.7, label='SPS RF separatrix')
            ax[0].plot(zeta_separatrix, -delta_separatrix * 1e3, '-', color='red', alpha=0.7, label=None)
        
        # Plot final particles
        if include_density_map:
            ax[1].scatter(x2, y2, c=z2, cmap='cool', s=2, label='Alive')
        else:   
            ax[1].plot(tbt_dict.zeta[alive_ind_final, -1], tbt_dict.delta[alive_ind_final, -1]*1000, '.', 
                color='blue', markersize=3.6, label='Alive')
        ax[1].plot(tbt_dict.zeta[dead_ind_final, -1], tbt_dict.delta[dead_ind_final, -1]*1000, '.', 
                color='darkred', markersize=3.6, label='Finally dead')
        if include_sps_separatrix:
            ax[1].plot(zeta_separatrix, delta_separatrix * 1e3, '-', color='red', alpha=0.7, label='SPS RF separatrix')
            ax[1].plot(zeta_separatrix, -delta_separatrix * 1e3, '-', color='red', alpha=0.7, label=None)
        ax[1].legend(loc='upper right', fontsize=13)
        
        # Plot initial and final particle distribution
        ax[2].bar(bin_centers, bin_heights, width=bin_widths, alpha=1.0, color='darkturquoise', label='Initial')
        ax[2].bar(bin_centers2, bin_heights2, width=bin_widths2, alpha=0.5, color='lime', label='Final (alive)')
        ax[2].legend(loc='upper right', fontsize=13)
        
        # Adjust axis limits and plot turn
        ax[0].set_ylim(-1.4, 1.4)
        ax[0].set_xlim(-0.85, 0.85)
        ax[1].set_ylim(-1.4, 1.4)
        ax[1].set_xlim(-0.85, 0.85)
        ax[2].set_xlim(-0.85, 0.85)
        #ax[2].set_ylim(-0.05, 1.1)
        
        ax[0].text(0.02, 0.91, 'Turn {}'.format(tbt_dict.full_data_turn_ind[0]+1), fontsize=15, transform=ax[0].transAxes)
        ax[1].text(0.02, 0.91, 'Turn {}'.format(tbt_dict.full_data_turn_ind[-1]+1), fontsize=15, transform=ax[1].transAxes)
            
        ax[2].set_xlabel(r'$\zeta$ [m]')
        ax[2].set_ylabel('Counts')
        ax[0].set_ylabel(r'$\delta$ [1e-3]')
        ax[1].set_ylabel(r'$\delta$ [1e-3]')
        plt.tight_layout()
        fig.savefig('output_plots/SPS_Pb_longitudinal.png', dpi=250)
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


    def load_Nb_data(self, path : str = Nb_data_path, index=0) -> pd.DataFrame:
        """
        Loads measured FBCT bunch intensity data from SPS MDs, processed with CCC miner
        https://github.com/ewaagaard/ccc_miner, returns pd.DataFrame
        
        Default date - 2023-10-16 with (Qx, Qy) = (26.3, 26.19) in SPS
        """
        # Load dictionary with emittance data
        try:
            with open(path, 'r') as fp:
                Nb_dict = json.load(fp)
        except FileNotFoundError:
            print('\nFILE NOT FOUND - check input path!\n')
            return
        
        # Create new dictionary wiht numpy arrays - divide by charge
        new_dict = {'ctime' : Nb_dict['ctime'],
                    'Nb' : np.array(Nb_dict['Nb1'])[:, index] / 82}
        
        # Create dataframe with four bunches
        df_Nb = pd.DataFrame(new_dict)
        df_Nb = df_Nb[(df_Nb['ctime'] < 46.55) & (df_Nb['ctime'] > 0.0)]
        
        return df_Nb
        
        
    def run_analytical_vs_kinetic_emittance_evolution(self,
                                                      Qy_frac : int = 25,
                                                      which_context='cpu',
                                                      add_non_linear_magnet_errors=False, 
                                                      beta_beat=None, 
                                                      beamParams=None,
                                                      ibs_step : int = 50,
                                                      context = None,
                                                      show_plot=False,
                                                      print_lost_particle_state=True,
                                                      install_longitudinal_rect=False,
                                                      plot_longitudinal_phase_space=True,
                                                      harmonic_nb = 4653,
                                                      extra_plot_string='',
                                                      return_data=False
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
        context : xo.context
            external xobjects context, if none is provided a new will be generated
        Qy_frac : int
            fractional part of vertical tune, e.g. "19" for 26.19
        print_lost_particle_state : bool
            whether to print the state of lost particles
        install_longitudinal_rect : bool
            whether to install the longitudinal LimitRect or not, to kill particles outside of bucket
        plot_longitudinal_phase_space : bool
            whether to plot the final longitudinal particle distribution
        return_data : bool
            whether to return dataframes of analytical data or not
        """
        # Update vertical tune if changed
        self.qy0 = int(self.qy0) + Qy_frac / 100
        
        # If specific beam parameters are not provided, load default SPS beam parameters
        if beamParams is None:
            beamParams = BeamParameters_SPS()
        print('Beam parameters:', beamParams)

        # Select relevant context
        if context is None:
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
        
        # Find bucket length
        bucket_length = line.get_length()/harmonic_nb
        max_zeta = bucket_length/2

        if install_longitudinal_rect:
            line.unfreeze() # if you had already build the tracker
            line.append_element(element=xt.LongitudinalLimitRect(min_zeta=-bucket_length/2, max_zeta=bucket_length/2), name='long_limit')
            print('\nInstalled longitudinal limitRect\n')
        else:
            line.discard_tracker()
        line.build_tracker(_context=context)
                
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
            
            # Print how many particles are lost or outside of the bucket, depending on if longitudinal LimitRect is installed
            if print_lost_particle_state and turn % self.turn_print_interval == 0:
                if particles.state[particles.state <= 0].size > 0 and install_longitudinal_rect:
                    print('Lost particle state: most common code: "-{}" for {} particles out of {} lost in total'.format(np.bincount(np.abs(particles.state[particles.state <= 0])).argmax(),
                                                                                                          np.max(np.bincount(np.abs(particles.state[particles.state <= 0]))),
                                                                                                          len(particles.state[particles.state <= 0])))
                else:
                    print('Particles out of bucket: {}'.format(len(particles.zeta) - len(particles.zeta[(particles.zeta < max_zeta) & (particles.zeta > -max_zeta)])))
        
        time01 = time.time()
        dt0 = time01-time00
        print('\nTracking time: {:.1f} s = {:.1f} h'.format(dt0, dt0/3600))
        
        # Save the data
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

        ax1.plot(turns, kicked_tbt.Tx, label='Kinetic')
        ax1.plot(turns, analytical_tbt.Tx, alpha=0.7, lw=1.5, label='Analytical Nagaitsev')
        ax2.plot(turns, kicked_tbt.Ty, label='Kinetic')
        ax2.plot(turns, analytical_tbt.Ty, alpha=0.7, lw=1.5, label='Analytical Nagaitsev')
        ax3.plot(turns, kicked_tbt.Tz, label='Kinetic')
        ax3.plot(turns, analytical_tbt.Tz, alpha=0.7, lw=1.5, label='Analytical Nagaitsev')

        ax1.set_ylabel(r'$T_{x}$')
        ax1.set_xlabel('Turns')
        ax2.set_ylabel(r'$T_{y}$')
        ax2.set_xlabel('Turns')
        ax3.set_ylabel(r'$T_{z}$')
        ax3.set_xlabel('Turns')
        ax1.legend(fontsize=10)
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

        if return_data:
            return df_kick, df_analytical