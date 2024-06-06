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

from .sequences import SPS_sequence_maker, BeamParameters_SPS, BeamParameters_SPS_Oxygen, BeamParameters_SPS_Proton
from .fma_ions import FMA
from .helpers import Records, Zeta_Container, Longitudinal_Monitor, _geom_epsx, _geom_epsy, Records_Growth_Rates
from .tune_ripple import Tune_Ripple_SPS
from .longitudinal import generate_parabolic_distribution
from .longitudinal import generate_binomial_distribution_from_PS_extr

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
    num_part: int = 10_000
    num_turns: int = 1000
    output_folder : str = "output" 
    turn_print_interval : int = 500
    qx0: float = 26.30
    qy0: float = 26.19
    ion_inj_ctime : float = 0.725 # ion injection happens at this time in cycle, important for WS
    proton_optics : str = 'q26'

    def generate_particles(self, 
                           line: xt.Line, 
                           context : xo.context, 
                           distribution_type='gaussian', 
                           beamParams=None,
                           engine=None, 
                           num_particles_linear_in_zeta=5, 
                           xy_norm_default=0.1,
                           scale_factor_Qs=None,
                           only_one_zeta=False) -> xp.Particles:
        """
        Generate xp.Particles object: matched Gaussian or longitudinally parabolic

        Parameters:
        -----------
        distribution_type : str
            'gaussian', 'parabolic', 'binomial' or 'linear_in_zeta'
        num_particles_linear_in_zeta : int
            number of equally spaced macroparticles linear in zeta
        xy_norm_default : float
            if building particles linear in zeta, what is the default normalized transverse coordinates (exact center not ideal for 
            if we want to study space charge and resonances)
        scale_factor_Qs : float
            if not None, factor by which we scale Qs (V_RF, h) and divide sigma_z and Nb for similar space charge effects
        only_one_zeta : bool
            for 'linear_in_zeta' distribution, whether to select only one particle in zeta (penultimate in amplitude) or not
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
            print('\nGaussian distribution generated.')
        elif distribution_type=='parabolic':
            particles = generate_parabolic_distribution(
                num_particles=self.num_part, 
                total_intensity_particles=beamParams.Nb,
                nemitt_x=beamParams.exn, 
                nemitt_y=beamParams.eyn, 
                sigma_z= beamParams.sigma_z,
                line=line, _context=context)
            print('\nParabolic distribution generated.')
        elif distribution_type=='binomial':
            # Also calculate SPS separatrix for plotting
            print('\nCreating binomial with sigma_z = {:.3f} m'.format(beamParams.sigma_z_binomial))
            particles, self._zeta_separatrix, self._delta_separatrix = generate_binomial_distribution_from_PS_extr(num_particles=self.num_part,
                                                                    nemitt_x=beamParams.exn, nemitt_y=beamParams.eyn,
                                                                    sigma_z=beamParams.sigma_z_binomial, total_intensity_particles=beamParams.Nb,
                                                                    line=line, m=beamParams.m, return_separatrix_coord=True)
            print('\nBinomial distribution generated.')
        elif distribution_type=='linear_in_zeta':
            
            # Find suitable zeta range - make linear spacing between close to center of RF bucket and to separatrix
            factor = scale_factor_Qs if scale_factor_Qs is not None else 1.0
            zetas = np.linspace(0.05, 0.7 / factor, num=num_particles_linear_in_zeta)
            
            # If only want penultimate particle in amplitude, select this one
            if only_one_zeta:
                zetas = np.array(zetas[-2])

            # Build the particle object
            particles = xp.build_particles(line = line, particle_ref = line.particle_ref,
                                        x_norm=xy_norm_default, y_norm=xy_norm_default, delta=0.0, zeta=zetas,
                                        nemitt_x = beamParams.exn, nemitt_y = beamParams.eyn, _context=context)
        else:   
            raise ValueError('Only Gaussian, parabolic and binomial distributions are implemented!')
            
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
                  apply_kinetic_IBS_kicks=False,
                  harmonic_nb = 4653,
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
                  pretrack_particles_and_update_sc_for_binomial=False,
                  plane_for_beta_beat='Y',
                  num_spacecharge_interactions=1080,
                  voltage=3.0e6,
                  scale_factor_Qs=None,
                  only_one_zeta=False,
                  install_beam_monitors=True,
                  nturns_profile_accumulation_interval = 100,
                  nbins = 160
                  ):
        """
        Run full tracking at SPS flat bottom
        
        Parameters:
        ----------
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
        auto_recompute_coefficients_percent : float
            relative emittance change after which to recompute 
        ibs_step : int
            Turn interval at which to recalculate IBS growth rates
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
        pretrack_particles_and_update_sc_for_binomial : bool
            whether to "pre-track" particles for 50 turns if binomial distribution with particles outside RF bucket is generated, 
            then updating space charge to new distribution
        plane_for_beta_beat : str
            plane in which beta-beat exists: 'X', 'Y' (default) or 'both'
        num_spacecharge_interactions : int
            number of SC interactions per turn
        voltage : float
            RF voltage in V
        scale_factor_Qs : float
            if not None, factor by which we scale Qs (V_RF, h) and divide sigma_z and Nb for similar space charge effects
        only_one_zeta : bool
            for 'linear_in_zeta' distribution, whether to select only one particle in zeta (penultimate in amplitude) or not
        install_beam_monitors : bool
            whether to install beam profile monitors at H and V Wire Scanner locations in SPS, that will record beam profiles
        nturns_profile_accumulation_interval : int
            turn interval between which to aggregate transverse and longitudinal particles for histogram
        nbins : int
            number of bins for histograms of transverse and longitudinal monitors

        Returns:
        --------
        tbt : pd.DataFrame
        """
        # Update vertical tune if changed
        self.qy0 = int(self.qy0) + Qy_frac / 100

        # If specific beam parameters are not provided, load default SPS beam parameters - for Pb or O
        if beamParams is None:
            if ion_type=='Pb':
                beamParams = BeamParameters_SPS()
            if ion_type=='O':
                beamParams = BeamParameters_SPS_Oxygen()
            if ion_type=='proton':
                beamParams = BeamParameters_SPS_Proton()
                harmonic_nb = 4620 # update harmonic number
            if distribution_type == 'binomial' and pretrack_particles_and_update_sc_for_binomial:
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

        # Get SPS Pb line - select ion or proton
        if ion_type=='Pb' or ion_type=='proton':
            sps = SPS_sequence_maker(ion_type=ion_type, proton_optics=self.proton_optics)
        elif ion_type=='O':
            sps = SPS_sequence_maker(ion_type='O', Q_PS=4., Q_SPS=8., m_ion=15.9949) 
        else:
            raise ValueError('Only Pb and O ions implemented so far!')
            
        # Extract line with aperture, beta-beat and non-linear magnet errors if desired
        line, twiss = sps.load_xsuite_line_and_twiss(Qy_frac=Qy_frac, add_aperture=add_aperture, beta_beat=beta_beat,
                                                   add_non_linear_magnet_errors=add_non_linear_magnet_errors, 
                                                   deferred_expressions=load_line_with_deferred_expressions,
                                                   plane=plane_for_beta_beat, voltage=voltage)
                
        # Remove unrealistic aperture below limit
        if minimum_aperture_to_remove is not None and add_aperture:
            line = sps.remove_aperture_below_threshold(line, minimum_aperture_to_remove)

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
                x_range=0.05,
                y_range=0.05)
            line.insert_element(index='bwsrc.51637', element=monitorH, name='monitorH')

            # Create vertical beam monitor
            monitorV = xt.BeamProfileMonitor(
                start_at_turn=nturns_profile_accumulation_interval/2, stop_at_turn=self.num_turns,
                frev=1,
                sampling_frequency=1/nturns_profile_accumulation_interval,
                n=nbins,
                x_range=0.05,
                y_range=0.05)
            line.insert_element(index='bwsrc.41677', element=monitorV, name='monitorV')
        
        line.build_tracker(_context=context)
        #######################################################################################################################

        # Generate particles object to track    
        particles = self.generate_particles(line=line, context=context, distribution_type=distribution_type,
                                            beamParams=beamParams, engine=engine, scale_factor_Qs=scale_factor_Qs,
                                            only_one_zeta=only_one_zeta)
        particles.reorganize()


        ######### IBS kinetic kicks #########
        if apply_kinetic_IBS_kicks:
            beamparams = BeamParameters.from_line(line, n_part=beamParams.Nb)
            opticsparams = OpticsParameters.from_line(line) # read from line without space  charge
            IBS = KineticKickIBS(beamparams, opticsparams)
            print('\nFixed IBS coefficient recomputation at interval = {} steps\n'.format(ibs_step))
            kinetic_kick_coefficients = IBS.compute_kick_coefficients(particles)
            print(kinetic_kick_coefficients)

        # Install SC and build tracker - optimize line if line variables for tune ripple not needed
        if install_SC_on_line:
            fma_sps = FMA(num_spacecharge_interactions=num_spacecharge_interactions)
            line = fma_sps.install_SC_and_get_line(line=line, beamParams=beamParams, mode=SC_mode, optimize_for_tracking=(not add_tune_ripple), 
                                                   distribution_type=distribution_type, context=context)
            print('Installed space charge on line\n')
            
        # If distribution is binimial, pre-track particles and removed killed ones, and update SC parameters
        if distribution_type=='binomial' and pretrack_particles_and_update_sc_for_binomial:
            
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
        tbt = Records.init_zeroes(self.num_turns)  # only emittances and bunch intensity
        tbt.update_at_turn(0, particles, twiss)
        
        # Install longitudinal beam profile monitor if desired
        if install_beam_monitors:
            zetas = Zeta_Container.init_zeroes(len(particles.x), nturns_profile_accumulation_interval, 
                                               which_context=which_context)
            zetas.update_at_turn(0, particles)
    
            # Longitudinal monitor - initiate class and define bucket length
            zeta_monitor = Longitudinal_Monitor.init_monitor(num_z_bins=nbins, n_turns_tot=self.num_turns, 
                                                             nturns_profile_accumulation_interval=nturns_profile_accumulation_interval)
            zmin_hist, zmax_hist = -0.55*bucket_length, 0.55*bucket_length

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

            # Update TBT, and save zetas
            tbt.update_at_turn(turn, particles, twiss)
            if install_beam_monitors:
                zetas.update_at_turn(turn % nturns_profile_accumulation_interval, particles) 

            # Merge all longitudinal coordinates to profile and stack
            if (turn+1) % nturns_profile_accumulation_interval == 0 and install_beam_monitors:
                
                # Generate and stack histogram
                zeta_monitor.convert_zetas_and_stack_histogram(zetas, num_z_bins=nbins, z_range=(zmin_hist, zmax_hist))

                # Initialize new zeta containers
                del zetas
                zetas = Zeta_Container.init_zeroes(len(particles.x), nturns_profile_accumulation_interval, 
                                           which_context=which_context)
                zetas.update_at_turn(0, particles) # start from turn, but 0 in new dataclass
                
            # Print number and cause of lost particles
            if particles.state[particles.state <= 0].size > 0:
                if print_lost_particle_state and turn % self.turn_print_interval == 0:
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
            tbt.append_profile_monitor_data(monitorH, monitorV, zeta_monitor, seconds_array)
            
        return tbt


        
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