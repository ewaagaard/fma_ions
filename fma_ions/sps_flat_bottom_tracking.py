"""
Class container for methods to track xpart particle objects at flat bottom
- for SPS
- choose context (GPU, CPU) and additional effects: SC, IBS, tune ripples
"""
from dataclasses import dataclass
import os
import numpy as np
import xtrack as xt
import xpart as xp
import xfields as xf
import xobjects as xo

from .beam_parameters import BeamParameters_SPS, BeamParameters_SPS_Binomial_2016, BeamParameters_SPS_Binomial_2016_before_RF_capture, BeamParameters_SPS_Oxygen, BeamParameters_SPS_Proton

from .sequences import SPS_sequence_maker
from .tune_ripple import Tune_Ripple_SPS
from .fma_ions import FMA
from .helpers_and_functions import Records, Zeta_Container, Longitudinal_Monitor, Records_Growth_Rates
from .longitudinal import generate_particles_transverse_gaussian, build_particles_linear_in_zeta, return_separatrix_coordinates

from xibs.inputs import BeamParameters, OpticsParameters
from xibs.kicks import KineticKickIBS
from xibs.analytical import NagaitsevIBS

import matplotlib.pyplot as plt
import time


@dataclass
class SPS_Flat_Bottom_Tracker:
    """
    Container to track xp.Particles at SPS flat bottom and store beam parameter results
    """
    num_part: int = 20_000
    num_turns: int = 1000
    output_folder : str = "output" 
    turn_print_interval : int = 10_000
    qx0: float = 26.30
    qy0: float = 26.25
    ion_inj_ctime : float = 0.725 # ion injection happens at this time in cycle, important for WS
    proton_optics : str = 'q26'

    def generate_particles(self, 
                           line: xt.Line, 
                           context=None, 
                           distribution_type='gaussian', 
                           beamParams=None,
                           matched_for_PS_extraction=False,
                           scale_factor_Qs=None) -> xp.Particles:
        """
        Generate xp.Particles object: matched Gaussian or longitudinally parabolic

        Parameters:
        -----------
        line: xt.Line
        context : xo.context
        distribution_type : str
            'gaussian', 'qgaussian', 'parabolic', 'binomial' or 'linear_in_zeta'
        beamParams : dataclass
            container of exn, eyn, Nb and sigma_z. Default 'None' will load nominal SPS beam parameters 
        matched_for_PS_extraction : bool
            whether to match particle object to before RF capture at SPS injection. If 'True', particles will be matched to PS extraction 
        scale_factor_Qs : float
            if not None, factor by which we scale Qs (V_RF, h) and divide sigma_z and Nb for similar space charge effects
        """
        if context is None:
            context = xo.ContextCpu(omp_num_threads='auto')
            print('CPU context generated')
        
        # Load beam parameters if not custom provided
        if beamParams is None:
            if distribution_type in ['gaussian', 'parabolic']:
                beamParams = BeamParameters_SPS()
            elif distribution_type in ['qgaussian', 'binomial']:
                beamParams = BeamParameters_SPS_Binomial_2016_before_RF_capture if matched_for_PS_extraction else BeamParameters_SPS_Binomial_2016()
        
        # Generate particles
        if distribution_type == 'linear_in_zeta':
            particles = build_particles_linear_in_zeta(beamParams, line, scale_factor_Qs=scale_factor_Qs)
        else:
            particles = generate_particles_transverse_gaussian(beamParams, line, longitudinal_distribution_type=distribution_type, num_part=self.num_part, 
                                                               _context=context, matched_for_PS_extraction=matched_for_PS_extraction)
    
        return particles


    def track_SPS(self, 
                  ion_type='Pb',
                  which_context='cpu',
                  add_non_linear_magnet_errors=False, 
                  add_aperture=True,
                  beta_beat=None, 
                  beamParams=None,
                  install_SC_on_line=True, 
                  SC_mode='frozen',
                  distribution_type='gaussian',
                  add_tune_ripple=False,
                  ripple_plane='both',
                  dq=0.01,
                  ripple_freq=50,
                  apply_kinetic_IBS_kicks=False,
                  harmonic_nb = 4653,
                  ibs_step = 5000,
                  minimum_aperture_to_remove=0.025,
                  matched_for_PS_extraction=False,
                  plane_for_beta_beat='both',
                  num_spacecharge_interactions=1080,
                  voltage=3.0e6,
                  scale_factor_Qs=None,
                  install_beam_monitors=True,
                  nturns_profile_accumulation_interval = 100,
                  nbins = 140,
                  cycle_mode_to_minimize_dx_dpx='dx',
                  target_dx_and_dpx=None,
                  also_keep_delta_profiles=False
                  ):
        """
        Run full tracking at SPS flat bottom
        
        Parameters:
        ----------
        ion_type : str
            which ion to use: currently available are 'Pb' and 'O'
        which_context : str
            'gpu' or 'cpu'
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
            'gaussian' or 'qgaussian' or 'parabolic' or 'binomial': particle distribution for tracking
        add_tune_ripple : bool
            whether to add external tune ripple from the Tune_Ripple_SPS class
        ripple_plane : str
            plane in which to add the tune ripple: 'X', 'Y' or 'both'
        dq : float
            amplitude for tune ripple, if applied
        ripple_freq : float
            ripple frequency in Hz
        add_kinetic_IBS_kicks : bool
            whether to apply kinetic kicks from xibs 
        harmonic_nb : int
            harmonic used for SPS RF system
        ibs_step : int
            Turn interval at which to recalculate IBS growth rates
        minimum_aperture_to_remove : float 
            minimum threshold of horizontal SPS aperture to remove, default is 0.025 (can also be set to None)
            as faulty IPM aperture has 0.01 m, which is too small
        matched_for_PS_extraction : bool
            whether to match particle object to before RF capture at SPS injection. If 'True', particles will be matched to PS extraction 
        plane_for_beta_beat : str
            plane in which beta-beat exists: 'X', 'Y' (default) or 'both'
        num_spacecharge_interactions : int
            number of SC interactions per turn
        voltage : float
            RF voltage in V
        scale_factor_Qs : float
            if not None, factor by which we scale Qs (V_RF, h) and divide sigma_z and Nb for similar space charge effects
        install_beam_monitors : bool
            whether to install beam profile monitors at H and V Wire Scanner locations in SPS, that will record beam profiles
        nturns_profile_accumulation_interval : int
            turn interval between which to aggregate transverse and longitudinal particles for histogram
        nbins : int
            number of bins for histograms of transverse and longitudinal monitors
        z_kick_num_integ_per_sigma : int
            number of longitudinal kicks per sigma
        cycle_mode_to_minimize_dx_dpx : str
            options: None, 'dx', 'dpx', 'both' and 'custum' --> whether to cycle line to minimum Dx at the start, 
            minimum D'x or minimize both. None will not perform any cycling. Default is to minimize initial dispersion,
            where IBS kicks are applied
        target_dx_and_dpx : list
            if cycle_mode chosen to be 'custom' above, provide list [dx_target, dpx_target] to cycle sequence as close as possible 
            to these values
        also_keep_delta_profiles : bool
            whether to keep aggregated delta coordinates in Zeta_Container or not
            
        Returns:
        --------
        tbt : Records
            dataclass containing ensemble quantities and beam profile monitor data
        """

        # If specific beam parameters are not provided, load default SPS beam parameters - for Pb or O
        if beamParams is None:
            if ion_type=='Pb':
                beamParams = BeamParameters_SPS()
            if ion_type=='O':
                beamParams = BeamParameters_SPS_Oxygen()
            if ion_type=='proton':
                beamParams = BeamParameters_SPS_Proton()
                harmonic_nb = 4620 # update harmonic number
            if distribution_type in ['binomial', 'qgaussian']:
                beamParams = BeamParameters_SPS_Binomial_2016_before_RF_capture if matched_for_PS_extraction else BeamParameters_SPS_Binomial_2016()
        print('Beam parameters:', beamParams)

        # Decide if longitudinal space charge kick is needed - for proton typically not needed, but required for ion synchrotron tune
        z_kick_num_integ_per_sigma=0 if ion_type == 'proton' else 10

        # Select relevant context
        if which_context=='gpu':
            context = xo.ContextCupy()
        elif which_context=='cpu':
            context = xo.ContextCpu(omp_num_threads='auto')
        else:
            raise ValueError('Context is either "gpu" or "cpu"')


        # Get SPS Pb line - select ion or proton
        if ion_type=='Pb' or ion_type=='proton':
            sps = SPS_sequence_maker(ion_type=ion_type, proton_optics=self.proton_optics, qx0=self.qx0, qy0=self.qy0)
        elif ion_type=='O':
            sps = SPS_sequence_maker(ion_type='O', Q_PS=4., Q_SPS=8., m_ion=15.9949, proton_optics=self.proton_optics, qx0=self.qx0, qy0=self.qy0) 
        else:
            raise ValueError('Only Pb and O ions implemented so far!')
        
        '''
        # Deferred expressions only needed for tune ripple and if re-matching after cycling sequence
        if add_tune_ripple or (cycle_mode_to_minimize_dx_dpx is not None): 
            load_line_with_deferred_expressions = True 
        else:
            load_line_with_deferred_expressions = False
        print('\nLoaded line with deferred expression: {}'.format(load_line_with_deferred_expressions))
        '''  
        
        # Extract line with aperture, beta-beat and non-linear magnet errors if desired
        line, twiss = sps.load_xsuite_line_and_twiss(add_aperture=add_aperture, beta_beat=beta_beat,
                                                   add_non_linear_magnet_errors=add_non_linear_magnet_errors, 
                                                   deferred_expressions=True, # needed for tune matching
                                                   plane=plane_for_beta_beat, voltage=voltage)
        print('{} optics: Qx = {:.3f}, Qy = {:.3f}'.format(self.proton_optics, twiss['qx'], twiss['qy']))
        
        # Rematch tunes to ensure correct values
        line.match(
            vary=[
                xt.Vary('kqf', step=1e-8),
                xt.Vary('kqd', step=1e-8),
            ],
            targets = [
                xt.Target('qx', self.qx0, tol=1e-8),
                xt.Target('qy', self.qy0, tol=1e-8),
            ])
        
        # Remove unrealistic aperture below limit
        if minimum_aperture_to_remove is not None and add_aperture:
            line = sps.remove_aperture_below_threshold(line, minimum_aperture_to_remove)
        
        if cycle_mode_to_minimize_dx_dpx is not None:
            
            if cycle_mode_to_minimize_dx_dpx == 'dx':
                penalty = np.abs(twiss.dx)
            elif cycle_mode_to_minimize_dx_dpx == 'dpx':    
                penalty = np.abs(twiss.dpx)
            elif cycle_mode_to_minimize_dx_dpx == 'both':   
                penalty = twiss.dx**2 + twiss.dpx**2 
            elif cycle_mode_to_minimize_dx_dpx == 'custom':
                penalty = (twiss.dx - target_dx_and_dpx[0])**2 + (twiss.dpx - - target_dx_and_dpx[1])**2
            else:
                raise ValueError("No valid cycling mode - choose 'dx', 'dpx' or 'both'")
            line = line.cycle(index_first_element=np.argmin(penalty))
            del twiss # delete old twiss table
                        
            twiss = line.twiss()
            print('Cycled sequence: Qx = {:.4f}, Qy = {:.4f}, starting Dx = {:3f} m, starting Dxprime = {:.3f}m\n'.format(twiss['qx'], twiss['qy'], twiss.dx[0], twiss.dpx[0]))


        # If scaling synchrotron tune
        if scale_factor_Qs is not None:
            line, sigma_z_new, Nb_new = sps.change_synchrotron_tune_by_factor(scale_factor_Qs, line, beamParams.sigma_z, beamParams.Nb)
            beamParams.sigma_z = sigma_z_new # update parameters
            beamParams.Nb = Nb_new # update parameters 
            harmonic_nb *= scale_factor_Qs
            print('Updated beam parameters with new Qs:')
            print(beamParams)
            
        ################# Longitudinal limit rect and Beam Profile Monitors (at Wire Scanner locations) #################
        # Add longitudinal limit rectangle - to kill particles that fall out of bucket
        bucket_length = line.get_length()/harmonic_nb
        print('\nBucket length is {:.4f} m'.format(bucket_length))
        line.unfreeze() # if you had already build the tracker
        line.append_element(element=xt.LongitudinalLimitRect(min_zeta=-bucket_length/2, max_zeta=bucket_length/2), name='long_limit')
        
        # Create horizontal beam monitor
        if install_beam_monitors:
            monitorH = xt.BeamProfileMonitor(
                start_at_turn=nturns_profile_accumulation_interval/2, stop_at_turn=self.num_turns,
                frev=1,
                sampling_frequency=1/nturns_profile_accumulation_interval,
                n=nbins,
                x_range=0.04,
                y_range=0.04)
            line.insert_element(index='bwsrc.51637', element=monitorH, name='monitorH')

            # Create vertical beam monitor
            monitorV = xt.BeamProfileMonitor(
                start_at_turn=nturns_profile_accumulation_interval/2, stop_at_turn=self.num_turns,
                frev=1,
                sampling_frequency=1/nturns_profile_accumulation_interval,
                n=nbins,
                x_range=0.04,
                y_range=0.04)
            line.insert_element(index='bwsrc.41677', element=monitorV, name='monitorV')
        
        line.build_tracker(_context=context)
        #######################################################################################################################

        # Generate particles object to track    
        particles = self.generate_particles(line=line, context=context, distribution_type=distribution_type,
                                            beamParams=beamParams, scale_factor_Qs=scale_factor_Qs, 
                                            matched_for_PS_extraction=matched_for_PS_extraction)
        
        # Initialize the dataclasses and store the initial values
        tbt = Records.init_zeroes(self.num_turns)  # only emittances and bunch intensity
        tbt.update_at_turn(0, particles, twiss)
        
        # Track particles for one turn         
        if matched_for_PS_extraction:
            line.track(particles, num_turns=1)
            print('Distribution matched for PS extraction - pre-tracked 1 turn, {} particles killed'.format(len(particles.state[particles.state <= 0])))
        # particles.reorganize() # needed?

        # Install SC and build tracker - optimize line if line variables for tune ripple not needed
        if install_SC_on_line:
            
            # Whether the longitudinal space charge kick should be included or not
            add_Z_kick_for_SC = True if z_kick_num_integ_per_sigma > 0 else False
            print('\nParticle type is {}, number of longitudinal SC kicks: {}'.format(ion_type, z_kick_num_integ_per_sigma))
            
            fma_sps = FMA(num_spacecharge_interactions=num_spacecharge_interactions)
            line = fma_sps.install_SC_and_get_line(line=line,
                                                   beamParams=beamParams, 
                                                   mode=SC_mode, 
                                                   optimize_for_tracking=(not add_tune_ripple), 
                                                   distribution_type=distribution_type, 
                                                   context=context,
                                                   add_Z_kick_for_SC=add_Z_kick_for_SC,
                                                   z_kick_num_integ_per_sigma=z_kick_num_integ_per_sigma)
            print('Installed {} space charge interactions with {} z kick intergrations per sigma on line\n'.format(num_spacecharge_interactions,
                                                                                                                   z_kick_num_integ_per_sigma))
            
        # Add tune ripple
        if add_tune_ripple:
            turns_per_sec = 1/twiss['T_rev0']
            ripple_period = int(turns_per_sec/ripple_freq)  # number of turns particle makes during one ripple oscillation
            ripple = Tune_Ripple_SPS(beta_beat=beta_beat, num_turns=self.num_turns, ripple_period=ripple_period, qx0=self.qx0, qy0=self.qy0)
            kqf_vals, kqd_vals, _ = ripple.load_k_from_xtrack_matching(dq=dq, plane=ripple_plane)
            
        ######### IBS kinetic kicks #########
        if apply_kinetic_IBS_kicks:
            #  friction and diffusion terms of the kinetic theory of gases
            ibs_kick = xf.IBSKineticKick(num_slices=50)

            # Install the IBS kinetic kick element
            line.configure_intrabeam_scattering(
                element=ibs_kick, name="ibskick", index=-1, update_every=ibs_step
            )
            print('\nFixed IBS coefficient recomputation at interval = {} steps\n'.format(ibs_step))
        
        # Install longitudinal beam profile monitor if desired
        if install_beam_monitors:
            zetas = Zeta_Container.init_zeroes(len(particles.x), nturns_profile_accumulation_interval, 
                                               which_context=which_context)
            zetas.update_at_turn(0, particles)
    
            # Longitudinal monitor - initiate class and define bucket length
            zeta_monitor = Longitudinal_Monitor.init_monitor(num_z_bins=nbins, n_turns_tot=self.num_turns, 
                                                             nturns_profile_accumulation_interval=nturns_profile_accumulation_interval)
            zmin_hist, zmax_hist = -0.55*bucket_length, 0.55*bucket_length
            delta_min_hist = 1.2 * np.min(context.nparray_from_context_array(particles.delta))
            delta_max_hist = 1.2 * np.max(context.nparray_from_context_array(particles.delta))

        # Start tracking 
        time00 = time.time()
        for turn in range(1, self.num_turns):
            
            if turn % self.turn_print_interval == 0:
                print('\nTracking turn {}'.format(turn))            
            
            ########## ----- Exert TUNE RIPPLE if desired ----- ##########
            if add_tune_ripple:
                line.vars['kqf'] = kqf_vals[turn-1]
                line.vars['kqd'] = kqd_vals[turn-1]
            
            # ----- Track and update records for tracked particles ----- #
            line.track(particles, num_turns=1)

            # Update TBT, and save zetas
            tbt.update_at_turn(turn, particles, twiss)
            if install_beam_monitors:
                zetas.update_at_turn(turn % nturns_profile_accumulation_interval, particles) 

            # Merge all longitudinal coordinates to profile and stack
            if (turn+1) % nturns_profile_accumulation_interval == 0 and install_beam_monitors:
                
                # Generate and stack histogram
                zeta_monitor.convert_zetas_and_stack_histogram(zetas, num_z_bins=nbins, z_range=(zmin_hist, zmax_hist),
                                                               delta_range=(delta_min_hist, delta_max_hist))

                # Initialize new zeta containers
                del zetas
                zetas = Zeta_Container.init_zeroes(len(particles.x), nturns_profile_accumulation_interval, 
                                           which_context=which_context)
                zetas.update_at_turn(0, particles) # start from turn, but 0 in new dataclass
                
            # Print number and cause of lost particles
            if particles.state[particles.state <= 0].size > 0:
                if turn % self.turn_print_interval == 0:
                    print('Lost particle state: most common code: "-{}" for {} particles out of {} lost in total'.format(np.bincount(np.abs(particles.state[particles.state <= 0])).argmax(),
                                                                                                          np.max(np.bincount(np.abs(particles.state[particles.state <= 0]))),
                                                                                                          len(particles.state[particles.state <= 0])))
            
        time01 = time.time()
        dt0 = time01-time00
        print('\nTracking time: {:.1f} s = {:.1f} h'.format(dt0, dt0/3600))
                
        # Convert turns to seconds
        turns_per_sec = 1 / twiss.T_rev0
        num_seconds = self.num_turns / turns_per_sec # number of seconds we are running for
        seconds_array = np.linspace(0.0, num_seconds, num=int(self.num_turns))

        # If beam profile monitors have been active
        if install_beam_monitors:
            tbt.append_profile_monitor_data(monitorH, monitorV, zeta_monitor, seconds_array, 
                                            also_keep_delta_profiles=also_keep_delta_profiles)
            
        return tbt


        
    def run_analytical_vs_kinetic_emittance_evolution(self,
                                                      Qy_frac : int = 25,
                                                      which_context='cpu',
                                                      add_non_linear_magnet_errors=False, 
                                                      beta_beat=None, 
                                                      distribution_type='gaussian',
                                                      beamParams=None,
                                                      ibs_step : int = 200,
                                                      context = None,
                                                      show_plot=False,
                                                      install_longitudinal_rect=True,
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
        distribution_type : str
            'gaussian' or 'qgaussian' or 'binomial'
        beamParams : dataclass
            container of exn, eyn, Nb and sigma_z. Default 'None' will load nominal SPS beam parameters 
        ibs_step : int
            turn interval at which to recalculate IBS growth rates
        context : xo.context
            external xobjects context, if none is provided a new will be generated
        Qy_frac : int
            fractional part of vertical tune, e.g. "19" for 26.19
        install_longitudinal_rect : bool
            whether to install the longitudinal LimitRect or not, to kill particles outside of bucket
        plot_longitudinal_phase_space : bool
            whether to plot the final longitudinal particle distribution
        return_data : bool
            whether to return dataframes of analytical data or not
        """
        os.makedirs('output_plots', exist_ok=True)
        
        # Update vertical tune if changed
        self.qy0 = int(self.qy0) + Qy_frac / 100
        
        # If specific beam parameters are not provided, load default SPS beam parameters
        if beamParams is None:
            if distribution_type == 'gaussian':
                beamParams = BeamParameters_SPS()
            elif distribution_type == 'binomial':
                beamParams = BeamParameters_SPS_Binomial_2016() # Assume after RF Spill
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
        print('{} optics: Qx = {:.3f}, Qy = {:.3f}'.format(self.proton_optics, twiss['qx'], twiss['qy']))
        
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
        particles = self.generate_particles(line=line, context=context, beamParams=beamParams, distribution_type=distribution_type)

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
        
        print('Initial:\nAnalytical: {}\nKinetic: {}\n'.format(growth_rates, kinetic_kick_coefficients))
        
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
            if turn % self.turn_print_interval == 0:
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
        fig.savefig('output_plots/analytical_vs_kinetic_emittance{}.png'.format(extra_plot_string), dpi=250)


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
        f.savefig('output_plots/analytical_vs_kinetic_growth_rates{}.png'.format(extra_plot_string), dpi=250)


        ############# GROWTH RATES --> same plot #############
        f2, ax = plt.subplots(1, 1, figsize = (8,6))

        ax.plot(turns, kicked_tbt.Tx, color='blue', label='Kinetic $T_{x}$')
        ax.plot(turns, analytical_tbt.Tx, ls='--', color='turquoise', alpha=0.7, lw=1.5, label='Analytical Nagaitsev $T_{x}$')
        ax.plot(turns, kicked_tbt.Ty, color='orange', alpha=0.9, label='Kinetic $T_{y}$')
        ax.plot(turns, analytical_tbt.Ty, color='red', ls='--', alpha=0.7, lw=1.5, label='Analytical Nagaitsev $T_{y}$')
        ax.plot(turns, kicked_tbt.Tz, color='green', alpha=0.9, label='Kinetic $T_{z}$')
        ax.plot(turns, analytical_tbt.Tz, alpha=0.7, color='lime', ls='--', lw=1.5, label='Analytical Nagaitsev $T_{z}$')

        ax.set_ylabel(r'$T_{x, y ,z}$')
        ax.set_xlabel('Turns')
        ax.legend(fontsize=10)
        plt.tight_layout()
        f2.savefig('output_plots/combined_analytical_vs_kinetic_growth_rates{}.png'.format(extra_plot_string), dpi=250)        


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
            fig3.savefig('output_plots/SPS_Pb_ions_longitudinal_bucket_{}turns{}.png'.format(self.num_turns, extra_plot_string), dpi=250)

        if show_plot:
            plt.show()
        plt.close()

        if return_data:
            return df_kick, df_analytical
        
        
    def print_kinetic_and_analytical_growth_rates(self, 
                                                   distribution_type='gaussian', 
                                                   beamParams=None,
                                                   beta_beat=None,
                                                   add_non_linear_magnet_errors=False
                                                   )->None:
        """Calculate kinetic and Nagaitsev analytical growth rates"""
        # If specific beam parameters are not provided, load default SPS beam parameters
        if beamParams is None:
            if distribution_type == 'gaussian':
                beamParams = BeamParameters_SPS()
            elif distribution_type == 'binomial' or distribution_type=='qgaussian': 
                beamParams = BeamParameters_SPS_Binomial_2016() # Assume after RF Spill
        print('Beam parameters:', beamParams)

        # Select relevant context
        context = xo.ContextCpu(omp_num_threads='auto')

        # Get SPS Pb line - with aperture and non-linear magnet errors if desired
        sps = SPS_sequence_maker(proton_optics=self.proton_optics, qx0=self.qx0, qy0=self.qy0)
        line, twiss = sps.load_xsuite_line_and_twiss(add_aperture=False, beta_beat=beta_beat,
                                                   add_non_linear_magnet_errors=add_non_linear_magnet_errors)

        line.build_tracker(_context=context)
                
        # Generate particles object to track    
        particles = self.generate_particles(line=line, context=context, beamParams=beamParams, distribution_type=distribution_type)

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
        
        print('Initial:\nAnalytical: {}\nKinetic: {}\n'.format(growth_rates, kinetic_kick_coefficients))