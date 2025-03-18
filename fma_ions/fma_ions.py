"""
Main class to perform Frequency Map Analysis
"""
from dataclasses import dataclass

import numpy as np
import xtrack as xt
import xpart as xp
import xfields as xf
import xobjects as xo
import os 
from scipy.interpolate import griddata

##### Plot settings 
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 20,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 15,
        "figure.titlesize": 20,
    }
)


import NAFFlib
from .sequences import PS_sequence_maker
from .sequences import SPS_sequence_maker
from .beam_parameters import BeamParameters_SPS
from .resonance_lines import resonance_lines
from .helpers_and_functions import FMA_keeper

@dataclass
class FMA:
    """
    Performs tracking and frequency map analysis
    
    Parameters:
    ----------
    use_uniform_beam : bool
        if True generate a transverse pencil distribution, otherwise 2D polar grid
    num_turns : int
        number of turns to track in total
    num_spacecharge_interactions : int
        number of SC interactions per turn
    delta0 : float
        relative momentum offset dp/p
    z0 : float
        initial longitudinal offset zeta
    mode : str
        space charge model: 'frozen', 'semi-frozen' or 'pic' (frozen recommended)
    n_theta : int
        if POLAR coordinates used, number of divisions for theta coordinates for particles in normalized coordinates
    n_r : int
        if POLAR coordinates used, number of divisions for r coordinates for particles in normalized coordinates
    r_min : float
        minimum radial distance from beam center to generate particles, to avoid zero amplitude oscillations for FMA
    n_linear : 
        if UNIFORM distribution is used, default number of points if uniform linear grid for normalized X and Y are used
    n_sigma : float
        maximum radial distance, i.e. max number of beam sizes sigma to generate particles
    Q_min_SPS, Q_min_PS : float, optional
        min tune to filter out from synchrotron frequency
    """
    use_uniform_beam: bool = True
    num_turns: int = 1200
    num_spacecharge_interactions: int = 1080 
    delta0: float = 0.0
    z0: float = 0.0
    n_theta: int = 50
    n_r: int = 100
    r_min: float = 0.1
    n_linear: int = 100
    n_sigma: float = 10.0
    mode: str = 'frozen'  # for now, only frozen space charge is available
    Q_min_SPS: float = 0.05
    Q_min_PS: float = 0.015
    
    
    def install_SC_and_get_line(self, 
                                line, 
                                beamParams, 
                                mode='frozen', 
                                optimize_for_tracking=True,
                                context=None,
                                distribution_type='gaussian',
                                pic_solver = 'FFTSolver2p5D',
                                add_Z_kick_for_SC=True,
                                use_binomial_dist_after_RF_spill=True,
                                z_kick_num_integ_per_sigma=10):
        """
        Install frozen Space Charge (SC) and generate particles with provided Xsuite line and beam parameters
        
        Parameters:
        ----------
        line : xtrack.line
            xsuite line to track through
        beamParams : dataclass 
            beam parameters (data class containing Nb, sigma_z, exn, eyn)
        mode : str
            type of space charge - 'frozen', 'quasi-frozen' and 'PIC'
        optimize_for_tracking : flag
            remove multiple drift spaces and line variables. Should be 'False' if knobs are used
        context : xo.context
            xojebts context for tracking
        distribution_type : str
            'gaussian' or 'qGaussian' or 'parabolic' or 'binomial' or 'linear_in_zeta': particle distribution for tracking
        pic_solver : str
            Choose solver between `FFTSolver2p5DAveraged` and `FFTSolver2p5D`
        add_Z_kick_for_SC : bool
            whether to install longitudinal kick for frozen space charge, otherwise risks of being non-symplectic
        use_binomial_dist_after_RF_spill : bool
            for binomial distributions, whether to use measured parameters after initial spill out of RF bucket (or before)
        z_kick_num_integ_per_sigma : int
            number of longitudinal kicks per sigma
        
        Returns:
        -------
        line : xtrack.line
            xtrack line with space charge installed 
        """
        # Choose default context if not given, and remove tracker if already exists 
        if context is None:
            context = xo.ContextCpu()
        line.discard_tracker()
        
        # Extract Twiss table from before installing space charge
        #line.build_tracker(_context=context, compile=False)
        twiss_xtrack = line.twiss()

        print('\nInstalling space charge on line...')
        
        # Initialize longitudinal profile for beams 
        if distribution_type=='gaussian' or distribution_type=='linear_in_zeta':
            q_val = 1.0
            print('\nGaussian longitudinal SC profile')
        elif distribution_type=='binomial' or distribution_type=='qgaussian':
            q_val = beamParams.q
            print('\nBinomial or qGaussian longitudinal SC profile, using parameters after spill: {}, and q = {}'.format(use_binomial_dist_after_RF_spill, 
                                                                                                            beamParams.q))
        elif distribution_type=='parabolic':
            raise ValueError('Parabolic not yet implemented for frozen!')
        
        lprofile = xf.LongitudinalProfileQGaussian(
                number_of_particles = beamParams.Nb,
                sigma_z = beamParams.sigma_z,
                z0=0.,
                q_parameter=q_val)

        print('\nInstalled SC with {} interactions.'.format(self.num_spacecharge_interactions))
        print(lprofile)
        print(line.particle_ref.show())
        print(beamParams)

        # Install frozen space charge as base 
        xf.install_spacecharge_frozen(line = line,
                           particle_ref = line.particle_ref,
                           longitudinal_profile = lprofile,
                           nemitt_x = beamParams.exn, nemitt_y = beamParams.eyn,
                           sigma_z = beamParams.sigma_z,
                           num_spacecharge_interactions = self.num_spacecharge_interactions)
        
        # Select mode - frozen is default
        if mode == 'frozen':
            pass # Already configured in line
        elif mode == 'quasi-frozen':
            xf.replace_spacecharge_with_quasi_frozen(
                                            line,
                                            update_mean_x_on_track=True,
                                            update_mean_y_on_track=True)
        elif mode == 'PIC':
            _, _ = xf.replace_spacecharge_with_PIC(
                    line=line,
                    n_sigmas_range_pic_x=8,
                    n_sigmas_range_pic_y=8,
                    nx_grid=256, ny_grid=256, nz_grid=100,
                    n_lims_x=7, n_lims_y=3,
                    z_range=(-3*beamParams.sigma_z, 3*beamParams.sigma_z),
                    solver=pic_solver)
        else:
            raise ValueError(f'Invalid mode: {mode}')

        # Build tracker for line, optimize elements if desired
        line.build_tracker(_context = context)
        if optimize_for_tracking:
            line.optimize_for_tracking()

        # Install longitudinal kick for frozen and quasi-frozen
        if add_Z_kick_for_SC and (mode != 'PIC'):
            tt = line.get_table()
            tt_sc = tt.rows[tt.element_type=='SpaceChargeBiGaussian']
            for nn in tt_sc.name:
                line[nn].z_kick_num_integ_per_sigma = z_kick_num_integ_per_sigma

            print('\nInstalled longitudinal SC kicks')

        # Find integer tunes from Twiss - BEFORE installing space charge
        self._Qx_int = int(twiss_xtrack['qx'])
        self._Qy_int = int(twiss_xtrack['qy'])

        return line


    def generate_particles(self, line, beamParams, 
                           make_single_Jy_trace=False,
                           y_norm0=0.05, context=None):
        """
        Generate xpart particle object from beam parameters 
    
        Parameters:
        ----------
        line : xtrack.line
            xsuite line to track through
        beamParams : dataclass 
            beam parameters (data class containing Nb, sigma_z, exn, eyn)
        make_single_Jy_trace : bool
            flag to create single trace with unique vertical action Jy, 
            with varying action Jx. "Trace" instead of "grid", if uniform beam is used
        y_norm0 : float
            starting normalized Y coordinate for the single Jy trace 
        context : xo.context
            xojebts context for tracking
        
        Returns
        -------
        particles : xpart particles object after tracking
        """
        # Choose default context if not given, and remove tracker if already exists 
        if context is None:
            context = xo.ContextCpu()    

        ##### Generate particles #####
        print('\nGenerating particles with delta = {:.2e} and z = {:.2e}'.format(self.delta0, self.z0))
        if self.use_uniform_beam:     
            print('Making UNIFORM distribution...')
            # Generate arrays of normalized coordinates 
            x_values = np.linspace(self.r_min, self.n_sigma, num=self.n_linear)  
            y_values = np.linspace(self.r_min, self.n_sigma, num=self.n_linear)  
        
            # Select single trace, or create a meshgrid for the uniform beam distribution
            if make_single_Jy_trace: 
                x_norm = x_values
                y_norm = y_norm0 * np.ones(len(x_norm))
                print('Making single-trace particles object with length {}\n'.format(len(y_norm)))
            else:
                X, Y = np.meshgrid(x_values, y_values)
                x_norm, y_norm = X.flatten(), Y.flatten()
        else:
            print('Making POLAR distribution...')
            x_norm, y_norm, _, _ = xp.generate_2D_polar_grid(
                                                            theta_range=(0.01, np.pi/2-0.01),
                                                            ntheta = self.n_theta,
                                                            r_range = (self.r_min, self.n_sigma),
                                                            nr = self.n_r)
        # Store initial normalized coordinates
        self._x_norm, self._y_norm = x_norm, y_norm
            
        # Build the particle object
        particles = xp.build_particles(line = line, particle_ref = line.particle_ref,
                                       x_norm=x_norm, y_norm=y_norm, delta=self.delta0, zeta=self.z0,
                                       nemitt_x = beamParams.exn, nemitt_y = beamParams.eyn, _context=context)
        
        print('\nBuilt particle object of size {}...'.format(len(particles.x)))
        
        return particles
        
    
    def track_particles(self, particles, line, context: xo.context):
        """
        Track particles through lattice with space charge elments installed
        
        Parameters:
        ----------
        particles : xpart.particles
            particles object from xpart
        line : xtrack.line
            xsuite line to track through
        context : xo.context
            which context particles are tracked on

        Returns:
        -------
        x, y - numpy.ndarrays
            arrays containing turn-by-turn data coordinates
        """          
        #### TRACKING #### 
        # Track the particles and return turn-by-turn coordinates

        # Instantiate TBT data keeper
        num_part = len(particles.x)
        tbt = FMA_keeper.init_zeroes(self.num_turns, num_part, self._x_norm, self._y_norm)

        print('\nStarting tracking...')

        # Perform the tracking
        for turn in range(self.num_turns):
            if turn % 50 == 0:
                print('Turn {}'.format(turn))
        
            tbt.update_at_turn(turn, particles, context)
        
            # Track the particles
            line.track(particles)

        
        print('Finished tracking.\n')

        # Set particle trajectories of dead particles that got lost in tracking
        self._kill_ind = context.nparray_from_context_array(particles.state) < 1
        self._kill_ind_exists = True

        return tbt
    
  
    
    def run_FMA(self, tbt, Qmin=0.0, remove_dead_particles=True):
        """
        Run FMA analysis for given turn-by-turn coordinates
        
        Parameters:
        ----------
        tbt : FMA_keeper
            object storing all fma_data
        Qmin : float
            if desired, filter out some lower frequencies
        
        Returns:
        --------
        Qx, Qy, d : np.ndarrays
            numpy arrays with turn-by-turn action, tune and diffusion data
        """
        
        # Initialize empty arrays for tunes of particles, during first and second half of run - at split_ind
        Qx_1, Qy_1 = np.zeros(len(tbt.x)), np.zeros(len(tbt.y))
        Qx_2, Qy_2 = np.zeros(len(tbt.x)), np.zeros(len(tbt.y))
        split_ind = int(self.num_turns/2)
        
        # Iterate over particles to find tune
        for i_part in range(len(tbt.x)):
            
            if i_part % 2000 == 0:
                print('NAFF algorithm of particle {}'.format(i_part))
            
            # Find dominant frequency with NAFFlib - also remember to subtract mean 
            Qx_1_raw = NAFFlib.get_tunes(tbt.x[i_part, :split_ind] \
                                            - np.mean(tbt.x[i_part, :split_ind]), 2)[0]
            Qx_1[i_part] = Qx_1_raw[np.argmax(Qx_1_raw > Qmin)]  # find most dominant tune larger than this value
            
            Qy_1_raw = NAFFlib.get_tunes(tbt.y[i_part, :split_ind] \
                                            - np.mean(tbt.y[i_part, :split_ind]), 2)[0]
            Qy_1[i_part] = Qy_1_raw[np.argmax(Qy_1_raw > Qmin)]
                
            Qx_2_raw = NAFFlib.get_tunes(tbt.x[i_part, split_ind:] \
                                            - np.mean(tbt.x[i_part, split_ind:]), 2)[0]
            Qx_2[i_part] = Qx_2_raw[np.argmax(Qx_2_raw > Qmin)]
                
            Qy_2_raw = NAFFlib.get_tunes(tbt.y[i_part, :split_ind] \
                                            - np.mean(tbt.y[i_part, :split_ind]), 2)[0]
            Qy_2[i_part] = Qy_2_raw[np.argmax(Qy_2_raw > Qmin)]
        
        # Change all zero-valued tunes to NaN
        Qx_1[Qx_1 == 0.0] = np.nan
        Qy_1[Qy_1 == 0.0] = np.nan
        Qx_2[Qx_2 == 0.0] = np.nan
        Qy_2[Qy_2 == 0.0] = np.nan
        
        # Remove dead particles from particle index - if exists
        if remove_dead_particles and self._kill_ind_exists:
            Qx_1[self._kill_ind] = np.nan
            Qy_1[self._kill_ind] = np.nan
            Qx_2[self._kill_ind] = np.nan
            Qy_2[self._kill_ind] = np.nan
        
        # Find FMA diffusion of tunes
        d = np.log(np.sqrt( (Qx_2 - Qx_1)**2 + (Qy_2 - Qy_1)**2))
    
        return Qx_2, Qy_2, d
       
        
        
    def run_SPS(self,
                ion_type='Pb',
                qx0=26.31,
                qy0=26.25,
                add_beta_beat=False,
                I_LSE=None,
                make_single_Jy_trace=False,
                use_symmetric_lattice=False,
                add_non_linear_magnet_errors=False,
                which_context = 'cpu',
                beamParams = None
                ):
        """
        Default FMA analysis for SPS Pb ions, plot final results and tune diffusion in initial distribution
        
        Parameters:
        ----------
        load_tbt_data : bool 
            whether to load turn-by-turn (TBT) data from tracking is already saved
        save_tune_data : bool 
            flag to store results Qx, Qy, d from FMA
        ion_type : str
            which ion to use: currently available are 'Pb' and 'O'
        qx0 : float
            horizontal tune
        qy0 : float
            vertical tune
        add_beta_beat : bool
            whether to add ~7% RMS beta-beat in both planes
        I_LSE : float
            sextupolar LSE current, to excite sextupole if desired
        make_single_Jy_trace : bool 
            flag to create single trace with unique vertical action
            Jy, with varying action Jx. "Trace" instead of "grid", if uniform beam is used
        use_symmetric_lattice : bool
            flag to use symmetric lattice without QFA and QDA
        add_non_linear_magnet_errors : bool
            whether to add non-linear chromatic errors for SPS
        which_context : str
            context for particle tracking - 'gpu' or 'cpu'
        beamParams : fma_ions.BeamParameters_SPS
            class with all beam parameters. If not given, default is loaded

        Returns:
        --------
        None
        """
        # Select relevant context
        if which_context=='gpu':
            context = xo.ContextCupy()
        elif which_context=='cpu':
            context = xo.ContextCpu(omp_num_threads='auto')
        else:
            raise ValueError('Context is either "gpu" or "cpu"')

        if beamParams is None:
            beamParams = BeamParameters_SPS()
            if ion_type=='O':
                beamParams.Nb = beamParams.Nb_O  # update to new oxygen intensity

        # Get SPS Pb line - select ion
        if ion_type=='Pb':
            sps_seq = SPS_sequence_maker(qx0=qx0, qy0=qy0)
        elif ion_type=='O':
            sps_seq = SPS_sequence_maker(ion_type='O', Q_PS=4., Q_SPS=8., m_ion=15.9949)
        else:
            raise ValueError('Only Pb and O ions implemented so far!')
        print(beamParams)
        
        # Load line
        line0, twiss_sps =  sps_seq.load_xsuite_line_and_twiss(use_symmetric_lattice=use_symmetric_lattice,
                                                               add_non_linear_magnet_errors=add_non_linear_magnet_errors)
        if add_beta_beat:
            line = sps_seq.add_beta_beat_to_line(line)
        
        # Excite sextupole if desired
        if I_LSE is not None:
            line = sps_seq.excite_LSE_sextupole_from_current(line, I_LSE=I_LSE, which_LSE='lse.12402')      

        # Add space charge elements to line, build tracker, generate particles
        line = self.install_SC_and_get_line(line0, beamParams, context=context)
        line.build_tracker(_context=context)    
        particles = self.generate_particles(line, beamParams, make_single_Jy_trace, context=context)

        # Track particles, run FMA analysis
        tbt = self.track_particles(particles, line, context)
        Qx, Qy, d = self.run_FMA(tbt, Qmin=self.Q_min_SPS)

        # Add interger tunes to fractional tunes, then store in dictionary
        Qx += int(twiss_sps['qx'])
        Qy += int(twiss_sps['qy'])
        tbt.add_tune_data_to_dict(Qx, Qy, d)

        # Add set tunes
        tbt.Qx0 = twiss_sps['qx']
        tbt.Qy0 = twiss_sps['qy']

        return tbt

        
@dataclass
class FMA_plotter:
    """
    Container for plotting classes and post-processing of FMA run data

    plot_range : list
        doubly nested list for plot interval
    output_folder : str 
        folder where to save data
    plot_order : int
        order to include in resonance diagram
    periodicity : int 
        periodicity in tune diagram
    """

    plot_range  = [[26.0, 26.35], [26.0, 26.35]]
    output_folder: str = 'output'
    plot_order: int = 4
    periodicity: int = 6

    def load_records_dict_from_json(self, output_folder=None):
        """
        Loads json file with particle data from tracking
        """
        folder_path = '{}/'.format(output_folder) if output_folder is not None else ''
        print('Loading data from {}tbt.json'.format(folder_path))

        # Read the json file, return either instanced class or dictionary
        try:
            tbt_dict = FMA_keeper.dict_from_json("{}tbt.json".format(folder_path))
            return tbt_dict
        except FileNotFoundError:
            print('Did not find dictionary!')


    def plot_FMA(self, tbt_dict=None, case_name='', 
                    plot_initial_distribution=True):   
        """
        Plots FMA diffusion and possibly initial distribution
        
        Parameters:
        ----------
        tbt_dict :  dict
            turn-by-turn dictionary containing arrays with input data from self.run_FMA. If none, will try to load
            saved dictionary
        Qx_set, Qy_set : float 
            set tunes, input data from Twiss
        case_name : str
            name string for scenario
            
        Returns:
        --------
        None
        """
        output_loc = f'{self.output_folder}/output' if self.output_folder is not None else 'output'
        os.makedirs(output_loc, exist_ok=True)

        if tbt_dict is None:
            tbt_dict = self.load_records_dict_from_json(output_folder=self.output_folder)

        fig = plt.figure(figsize=(9,6), constrained_layout=True)
        tune_diagram = resonance_lines(self.plot_range[0],
                    self.plot_range[1], np.arange(1, self.plot_order+1), self.periodicity)
        tune_diagram.plot_resonance(figure_object = fig, interactive=False)

        # Combine Qx, Qy, and d into a single array for sorting
        data = list(zip(tbt_dict['Qx'], tbt_dict['Qy'], tbt_dict['d']))
        
        # Sort the data based on the 'd' values
        sorted_data = sorted(data, key=lambda x: x[2], reverse=False)
        
        # Unpack the sorted data
        sorted_Qx, sorted_Qy, sorted_d = zip(*sorted_data)

        plt.scatter(sorted_Qx, sorted_Qy, s=5.0, c=sorted_d, marker='o', lw = 0.1, zorder=10, cmap=plt.cm.jet) #, alpha=0.55)
        plt.plot(tbt_dict['Qx0'], tbt_dict['Qy0'], 'o', color='k', markerfacecolor='red', zorder=20, markersize=11, label="Set tune")
        plt.xlabel('$Q_{x}$')
        plt.ylabel('$Q_{y}$')
        cbar=plt.colorbar()
        cbar.set_label('d',fontsize='18')
        cbar.ax.tick_params(labelsize='18')
        plt.legend(loc='upper left')
        plt.clim(-20.5,-4.5)
        fig.savefig('{}/FMA_plot{}.png'.format(output_loc, case_name), dpi=250)

        if plot_initial_distribution:
            self.plot_initial_distribution(tbt_dict, case_name, output_loc)

        plt.show()


    def plot_initial_distribution(self, tbt_dict, case_name=''): 
        """
        Plot initial distribution, interpolating between discrete points
        
        Parameters:
        ----------
        tbt_dict :  dict
            turn-by-turn dictionary containing arrays with input data from self.run_FMA.
        case_name : str
            name string for scenario
        output_loc : str
            where to save data
        """ 
        fig2 = plt.figure(figsize=(8,6), constrained_layout=True)
        fig2.suptitle('Initial Distribution', fontsize='18')
        
        # Combine x0_norm, y0_norm and d into a single array for sorting
        data = list(zip(tbt_dict['x0_norm'], tbt_dict['y0_norm'], tbt_dict['d']))
        x_string = '$\sigma_{x}$'
        y_string = '$\sigma_{y}$'

        # Sort the data based on the 'd' values
        sorted_data = sorted(data, key=lambda x: x[2], reverse=False)
        
        # Unpack the sorted data
        sorted_x, sorted_y, sorted_d = zip(*sorted_data)
    
        plt.scatter(sorted_x, sorted_y, s=5.5, c=sorted_d, marker='o', lw = 0.1, zorder=10, cmap=plt.cm.jet)  # without interpolation
        plt.tick_params(axis='both', labelsize='18')
        plt.xlabel('{}'.format(x_string), fontsize='20')
        plt.ylabel('{}'.format(y_string), fontsize='20')
        plt.clim(-20.5,-4.5)
        cbar=plt.colorbar()
        cbar.set_label('d',fontsize='18')
        cbar.ax.tick_params(labelsize='18')
        fig2.savefig('{}/Initial_norm_distribution{}.png'.format(self.output_folder, case_name), dpi=250)


    def plot_tune_over_action(self, twiss, 
                            load_tune_data=False,
                            load_up_to_turn=None,
                            also_show_plot=True, 
                            resonance_order=5, 
                            case_name=None,
                            plane='X'):
    
        """
        Loads generated turn-by-turn data and plots tunes Qx, Qy over action Jx, Jy
        
        Parameters:
        -----------
        twiss : xtrack.twisstable
            twiss table from xtrack
        load_tbt_data : bool
            load tune data if FMA has already been done
        load_up_to_turn : int, optional
            turn up to which to load tbt data
        also_show_plot : bool 
            whether to include "plt.show()"
        resonance_order : int
            order up to which resonance should be plotted
        case_name : str, optional
            additional string to add to figure heading 
        plane : str
            'X' or 'Y'
        
        Returns:
        --------
        Jx, Jy, Qx, Qy, d : np.ndarrays
            numpy arrays with turn-by-turn action, tune and diffusion data
        """
        if load_up_to_turn is None:
            load_up_to_turn = self.num_turns
        
        # Load tracking data 
        x, y, px, py  = self.load_tracking_data()

        # Calculate normalized coordinates
        X = x / np.sqrt(twiss['betx'][0]) 
        PX = twiss['alfx'][0] / np.sqrt(twiss['betx'][0]) * x + np.sqrt(twiss['betx'][0]) * px
        Y = y / np.sqrt(twiss['bety'][0]) 
        PY = twiss['alfy'][0] / np.sqrt(twiss['bety'][0]) * y + np.sqrt(twiss['bety'][0]) * py
        
        # Calculate action for each particle
        Jx = X**2 + PX **2
        Jy = Y**2 + PY **2

        # Try to load tunes and action if already exists, otherwise perform FMA again
        if load_tune_data:
            try:
                Qx = np.load('{}/Qx.npy'.format(self.output_folder))
                Qy = np.load('{}/Qy.npy'.format(self.output_folder))
                d = np.load('{}/d.npy'.format(self.output_folder))
                print('\nLoaded existing tune and action data!\n')

            except FileNotFoundError:
                raise FileNotFoundError('\nDid not find existing tune and action data - initializing FMA!\n')
        else:
            
            # Find tunes of particles up to desired turn
            Qx, Qy, d = self.run_FMA(x[:, :load_up_to_turn], y[:, :load_up_to_turn])
            
            # Save the tunes and action, if possible
            os.makedirs(self.output_folder, exist_ok=True)
            np.save('{}/Qx.npy'.format(self.output_folder), Qx)
            np.save('{}/Qy.npy'.format(self.output_folder), Qy)
            np.save('{}/d.npy'.format(self.output_folder), d)
        
            print('Saved tune and action data.\n')
                
        # Plot initial action 
        fig, ax = plt.subplots(1, 2, figsize=(12,6))
        name_str = 'Tune over action' if case_name is None else 'Tune over action - {}'.format(case_name)
        fig.suptitle(name_str)

        ax[0].plot(Jx[:, 0], Qx, 'o', color='b', alpha=0.5, markersize=2.5)
        ax[1].plot(Jy[:, 0], Qy, 'o', color='r', alpha=0.5, markersize=2.5)
        
        ax[0].set_ylabel(r"$Q_{x}$")
        ax[0].set_xlabel(r"$J_{x}$")
        ax[1].set_ylabel(r"$Q_{y}$")
        ax[1].set_xlabel(r"$J_{y}$")
        
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')
        
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        
        # Plot tune over normalized beam size, in horizontal
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        if plane == 'X':
            ax2.scatter(self._x_norm, Qx, s=5.5, c=self._x_norm, marker='o', zorder=10, cmap=plt.cm.cool)
            ax2.set_ylabel("$Q_{x}$")
            ax2.set_xlabel("$\sigma_{x}$")
        elif plane == 'Y':
            ax2.plot(self._y_norm, Qy, 'o', color='r', alpha=0.5, markersize=2.5)       
            ax2.set_ylabel("$\Q_{y}$")
            ax2.set_xlabel("$\sigma_{y}$")
        else:
            raise ValueError('\nInvalid plane specified!\n')
        
        # Add colorbar, normalized to beam size (in sigmas)
        fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(min(self._x_norm), max(self._x_norm)), cmap='cool'),
                ax=ax2, orientation='vertical', label='$\sigma_{x}$')
        
        fig2.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        
        fig.savefig('{}/{}_Tune_over_action.png'.format(self.output_folder, case_name), dpi=250)
        fig2.savefig('{}/{}_Tune_over_normalized_{}.png'.format(self.output_folder, case_name, plane), dpi=250)
        
        if also_show_plot:
            plt.show()

        return Jx, Jy, Qx, Qy, d
    
    
    def plot_normalized_phase_space(self, twiss, 
                                    particle_index=None,
                                    also_show_plot=True,
                                    case_name=None,
                                    plane='X'
                                    ):
        """
        Generate phase space plots in X and Y from generated turn-by-turn data
        
        Parameters:
        -----------
        twiss : xtrack.twisstable 
            twiss table from xtrack
        particle_index : numpy.ndarray, optional
            array with particle index, if not given takes all
        also_show_plot : bool 
            whether to include "plt.show()"
        case_name : str, optional
            extra string to add to case name
        plane : str
            'X' or 'Y'
            
        Returns:
        -------
        None
        """
        x, y, px, py  = self.load_tracking_data()
        
        if particle_index is not None:
            i = particle_index
        else:
            i = np.arange(1, len(x)) # particle index
        
        X = x / np.sqrt(twiss['betx'][0]) 
        PX = twiss['alfx'][0] / np.sqrt(twiss['betx'][0]) * x + np.sqrt(twiss['betx'][0]) * px
        Y = y / np.sqrt(twiss['bety'][0]) 
        PY = twiss['alfy'][0] / np.sqrt(twiss['bety'][0]) * y + np.sqrt(twiss['bety'][0]) * py

        # Calculate action for each particle
        Jx = X**2 + PX **2
        Jy = Y**2 + PY **2

        # Generate two figures - one in normalized phase space, one in polar action space (Jx, phi)
        fig, ax = plt.subplots(1, 1, figsize=(8,6))
        fig.suptitle('Normalized phase space' if case_name is None else 'Normalized phase space - {}'.format(case_name), 
                        fontsize=16)

        fig2, ax2 = plt.subplots(1, 1, figsize=(8,6))
        fig2.suptitle('Polar action space' if case_name is None else 'Polar action space - {}'.format(case_name), 
                        fontsize=16)

        # Calculate phase space angle
        if plane == 'X':
            phi = np.arctan2(X, PX)          
        elif plane == 'Y':
            phi = np.arctan2(Y, PY)
        else:
            raise ValueError('Plane invalid - has to be "X" or "Y"')
            
        # Take colors from colormap of normalized phase space
        colors = plt.cm.cool(np.linspace(0, 1, len(self._x_norm)))

        for particle in i:
            
            # Mix black and colorbar 
            color=colors[particle] if particle % 2 == 0 else 'k'
                
            # Plot normalized phase space and action space
            if plane == 'X':
                ax.plot(X[particle, :], PX[particle, :], 'o', color=color, alpha=0.5, markersize=1.5)
                ax.set_ylabel(r"$P_{x}$")
                ax.set_xlabel(r"$X$")
                
                ax2.plot(phi[particle, :], Jx[particle, :], 'o', color=color, alpha=0.5, markersize=1.5)
                ax2.set_ylabel(r"$J_{x}$")
                ax2.set_xlabel(r"$\phi$ [rad]")
            elif plane == 'Y':
                ax.plot(Y[particle, :], PY[particle, :], 'o', color=color, alpha=0.5, markersize=1.5)
                ax.set_ylabel(r"$P_{y}$")
                ax.set_xlabel(r"$Y$")
                
                ax2.plot(phi[particle, :], Jy[particle, :], 'o', color=color, alpha=0.5, markersize=1.5)
                ax2.set_ylabel(r"$J_{y}$")
                ax2.set_xlabel(r"$\phi$ [rad]")
                
        # Add colorbar, normalized to beam size (in sigmas)
        fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(min(self._x_norm), max(self._x_norm)), cmap='cool'),
                ax=ax, orientation='vertical', label='$\sigma_{x}$')
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        
        fig2.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(min(self._x_norm), max(self._x_norm)), cmap='cool'),
                ax=ax2, orientation='vertical', label='$\sigma_{x}$')
        fig2.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        
        # Save figures
        fig.savefig('{}/{}_Normalized_phase_space.png'.format(self.output_folder, case_name), dpi=250)
        fig2.savefig('{}/{}_Polar_action_space_{}.png'.format(self.output_folder, case_name, plane), dpi=250)
        
        if also_show_plot:
            plt.show()
        
        plt.close()