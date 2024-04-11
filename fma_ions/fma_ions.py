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
from .sequence_classes_ps import PS_sequence_maker, BeamParameters_PS
from .sequence_classes_sps import SPS_sequence_maker, BeamParameters_SPS
from .resonance_lines import resonance_lines

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
    output_folder : str 
        folder where to save data
    plot_order : int
        order to include in resonance diagram
    periodicity : int 
        periodicity in tune diagram
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
    output_folder: str = 'output_fma'
    plot_order: int = 4
    periodicity: int = 16
    Q_min_SPS: float = 0.05
    Q_min_PS: float = 0.015
    
    
    def install_SC_and_get_line(self, 
                                line, 
                                beamParams, 
                                mode='frozen', 
                                optimize_for_tracking=True,
                                context=None,
                                distribution_type='gaussian',
                                pic_solver = 'FFTSolver2p5DAveraged'):
        """
        Install frozen Space Charge (SC) and generate particles with provided Xsuite line and beam parameters
        
        Parameters:
        ----------
        line : xtrack.line
            xsuite line to track through
        beamParams : dataclass 
            beam parameters (data class containing Nb, sigma_z, exn, eyn)
        mode : str
            type of space charge ('frozen' is recommended)
        optimize_for_tracking : flag
            remove multiple drift spaces and line variables. Should be 'False' if knobs are used
        context : xo.context
            xojebts context for tracking
        distribution_type : str
            'gaussian' or 'parabolic' or 'binomial': particle distribution for tracking
        pic_solver : str
            Choose solver between `FFTSolver2p5DAveraged` and `FFTSolver2p5D`
        
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
        line.build_tracker(_context=context, compile=False)
        if optimize_for_tracking:
            line.optimize_for_tracking()
        twiss_xtrack = line.twiss()

        print('\nInstalling space charge on line...')
        # Initialize longitudinal profile for beams 
        if distribution_type=='gaussian':
            sigma_z_RMS = beamParams.sigma_z
            q_val = 1.0
            print('\nGaussian longitudinal SC profile')
        elif distribution_type=='binomial':
            sigma_z_RMS = beamParams.sigma_z_binomial
            q_val = 0.8
            print('\nBinomial longitudinal SC profile')
        elif distribution_type=='parabolic':
            raise ValueError('Parabolic not yet implemented for frozen!')
        
        lprofile = xf.LongitudinalProfileQGaussian(
                number_of_particles = beamParams.Nb,
                sigma_z = sigma_z_RMS,
                z0=0.,
                q_parameter=q_val)

        # Install frozen space charge as base 
        xf.install_spacecharge_frozen(line = line,
                           particle_ref = line.particle_ref,
                           longitudinal_profile = lprofile,
                           nemitt_x = beamParams.exn, nemitt_y = beamParams.eyn,
                           sigma_z = sigma_z_RMS,
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

        # Build tracker for line
        line.build_tracker(_context = context)
        #twiss_xtrack_with_sc = line.twiss()  # --> better to do Twiss before installing SC, if enough SC interactions

        # Find integer tunes from Twiss - BEFORE installing space charge
        self._Qx_int = int(twiss_xtrack['qx'])
        self._Qy_int = int(twiss_xtrack['qy'])

        return line


    def generate_particles(self, line, beamParams, 
                           make_single_Jy_trace=False,
                           y_norm0=0.05):
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
        
        Returns
        -------
        particles : xpart particles object after tracking
        """
        
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
                                       nemitt_x = beamParams.exn, nemitt_y = beamParams.eyn)
        
        print('\nBuilt particle object of size {}...'.format(len(particles.x)))
        
        return particles
        
    
    def track_particles(self, particles, line, save_tbt_data=True):
        """
        Track particles through lattice with space charge elments installed
        
        Parameters:
        ----------
        particles : xpart.particles
            particles object from xpart
        line : xtrack.line
            xsuite line to track through
        save_tbt: bool
            whether to save turn-by-turn data from tracking
        
        Returns:
        -------
        x, y - numpy.ndarrays
            arrays containing turn-by-turn data coordinates
        """          
        #### TRACKING #### 
        # Track the particles and return turn-by-turn coordinates
        x = np.zeros([len(particles.x), self.num_turns]) 
        y = np.zeros([len(particles.y), self.num_turns])
        px = np.zeros([len(particles.px), self.num_turns]) 
        py = np.zeros([len(particles.py), self.num_turns])
        
        print('\nStarting tracking...')
        i = 0
        for turn in range(self.num_turns):
            if i % 20 == 0:
                print('Tracking turn {}'.format(i))
        
            x[:, i] = particles.x
            y[:, i] = particles.y
            px[:, i] = particles.px
            py[:, i] = particles.py
        
            # Track the particles
            line.track(particles)
            i += 1
        
        print('Finished tracking.\n')
        print('Average X and Y of tracking: {} and {}'.format(np.mean(x), np.mean(y)))
        
        # Set particle trajectories of dead particles that got lost in tracking
        self._kill_ind = particles.state < 1
        self._kill_ind_exists = True
        
        if save_tbt_data:
            os.makedirs(self.output_folder, exist_ok=True)
            np.save('{}/x.npy'.format(self.output_folder), x)
            np.save('{}/y.npy'.format(self.output_folder), y)
            np.save('{}/px.npy'.format(self.output_folder), px)
            np.save('{}/py.npy'.format(self.output_folder), py)
            np.save('{}/x0_norm.npy'.format(self.output_folder), self._x_norm)
            np.save('{}/y0_norm.npy'.format(self.output_folder), self._y_norm)
            np.save('{}/state.npy'.format(self.output_folder), self._kill_ind)
            print('Saved tracking data.')
        

        return x, y
    
    
    def load_tracking_data(self):
        """
        Loads numpy data if tracking has already been made
        
        Returns:
        --------
        x, y, px, py : np.ndarrays
            numpy arrays with turn-by-turn data
        """
        try:
            if self.delta0 == 0.0 and self.z0 == 0.0:
                print('\nLoading on-momentum data!\n')
            else:
                print('\nLoading off-momentum data!\n')
                                
            x=np.load('{}/x.npy'.format(self.output_folder))
            y=np.load('{}/y.npy'.format(self.output_folder))
            px=np.load('{}/px.npy'.format(self.output_folder))
            py=np.load('{}/py.npy'.format(self.output_folder))
            self._x_norm =np.load('{}/x0_norm.npy'.format(self.output_folder))
            self._y_norm =np.load('{}/y0_norm.npy'.format(self.output_folder))
            
            # If index with killed particles exist, raise flag for FMA analysis
            try:
                self._kill_ind_exists = True
                self._kill_ind = np.load('{}/state.npy'.format(self.output_folder))
            except FileNotFoundError:
                print('\nKill index does not exist - proceeding with all particles\n')
                self._kill_ind_exists = False
            
            return x, y, px, py

        except FileNotFoundError:
            raise FileNotFoundError('Tracking data does not exist - set correct path or generate the data!')

    
    def load_tune_data(self):
        """
        Loads numpy data of tunes if FMA has already been done
        
        Returns:
        --------
        Qx, Qy, d : np.ndarrays
            numpy arrays with turn-by-turn tune and diffusion data
        """
        try:
            Qx = np.load('{}/Qx.npy'.format(self.output_folder))
            Qy = np.load('{}/Qy.npy'.format(self.output_folder))
            d = np.load('{}/d.npy'.format(self.output_folder))
            return Qx, Qy, d
        except FileNotFoundError:
            raise FileNotFoundError('Tune data does not exist - set correct path or perform FMA!')

  
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
    
    
    def run_FMA(self, x_tbt_data, y_tbt_data, Qmin=0.0, remove_dead_particles=True):
        """
        Run FMA analysis for given turn-by-turn coordinates
        
        Parameters:
        ----------
        x_tbt_data, y_tbt_data : np.ndarrays
            arrays of turn-by-turn data for particles 
        Qmin : float
            if desired, filter out some lower frequencies
        
        Returns:
        --------
        Qx, Qy, d : np.ndarrays
            numpy arrays with turn-by-turn action, tune and diffusion data
        """
        
        # Initialize empty arrays for tunes of particles, during first and second half of run - at split_ind
        Qx_1, Qy_1 = np.zeros(len(x_tbt_data)), np.zeros(len(x_tbt_data))
        Qx_2, Qy_2 = np.zeros(len(x_tbt_data)), np.zeros(len(x_tbt_data))
        split_ind = int(self.num_turns/2)
        
        # Iterate over particles to find tune
        for i_part in range(len(x_tbt_data)):
            
            if i_part % 2000 == 0:
                print('NAFF algorithm of particle {}'.format(i_part))
            
            # Find dominant frequency with NAFFlib - also remember to subtract mean 
            Qx_1_raw = NAFFlib.get_tunes(x_tbt_data[i_part, :split_ind] \
                                            - np.mean(x_tbt_data[i_part, :split_ind]), 2)[0]
            Qx_1[i_part] = Qx_1_raw[np.argmax(Qx_1_raw > Qmin)]  # find most dominant tune larger than this value
            
            Qy_1_raw = NAFFlib.get_tunes(y_tbt_data[i_part, :split_ind] \
                                            - np.mean(y_tbt_data[i_part, :split_ind]), 2)[0]
            Qy_1[i_part] = Qy_1_raw[np.argmax(Qy_1_raw > Qmin)]
                
            Qx_2_raw = NAFFlib.get_tunes(x_tbt_data[i_part, split_ind:] \
                                            - np.mean(x_tbt_data[i_part, split_ind:]), 2)[0]
            Qx_2[i_part] = Qx_2_raw[np.argmax(Qx_2_raw > Qmin)]
                
            Qy_2_raw = NAFFlib.get_tunes(y_tbt_data[i_part, :split_ind] \
                                            - np.mean(y_tbt_data[i_part, :split_ind]), 2)[0]
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
       
        
    def plot_FMA(self, d, Qx, Qy, Qh_set, Qv_set, case_name, 
                 plot_range, plot_initial_distribution=True):   
        """
        Plots FMA diffusion and possibly initial distribution
        
        Parameters:
        ----------
        d, Qx, Qy :  numpy.ndarray
            arrays with input data from self.run_FMA
        Qx_set, Qy_set : float 
            set tunes, input data from Twiss
        case_name : str
            name string for scenario
            
        Returns:
        --------
        None
        """
        fig = plt.figure(figsize=(9,6))
        tune_diagram = resonance_lines(plot_range[0],
                    plot_range[1], np.arange(1, self.plot_order+1), self.periodicity)
        tune_diagram.plot_resonance(figure_object = fig, interactive=False)

        # Combine Qx, Qy, and d into a single array for sorting
        data = list(zip(Qx, Qy, d))
        
        # Sort the data based on the 'd' values
        sorted_data = sorted(data, key=lambda x: x[2], reverse=False)
        
        # Unpack the sorted data
        sorted_Qx, sorted_Qy, sorted_d = zip(*sorted_data)

        plt.scatter(sorted_Qx, sorted_Qy, s=5.0, c=sorted_d, marker='o', lw = 0.1, zorder=10, cmap=plt.cm.jet) #, alpha=0.55)
        plt.plot(Qh_set, Qv_set, 'o', color='k', markerfacecolor='red', zorder=20, markersize=11, label="Set tune")
        plt.xlabel('$Q_{x}$')
        plt.ylabel('$Q_{y}$')
        cbar=plt.colorbar()
        cbar.set_label('d',fontsize='18')
        cbar.ax.tick_params(labelsize='18')
        plt.legend(loc='upper left')
        plt.clim(-20.5,-4.5)
        fig.tight_layout(pad=0.6, w_pad=0.5, h_pad=1.0)
        fig.savefig('{}/FMA_plot_{}.png'.format(self.output_folder, case_name), dpi=250)
        
        
    def plot_initial_distribution(self, x, y, d, case_name, use_normalized_coordinates=True,
                                  interpolate_initial_distribution=False, also_show_plot=False): 
        """
        Plot initial distribution, interpolating between discrete points
        
        Parameters:
        ----------
        x, y, d :  numpy.ndarray
            arrays with input data generated from self.run_FMA
        case_name : str
            name string for scenario
        use_normalized_coordinates : bool
            flag whether to normalize coordinates w.r.t beam size
        interpolate_initial_distribution : bool
            interpolate initial particle distribution into colormesh, or keep the points as they are 
        also_show_plot : bool 
            whether to run plt.show()
            
        Returns:
        --------
        None
        """ 
        fig2=plt.figure(figsize=(8,6))
        XX,YY = np.meshgrid(np.unique(x[:,0]), np.unique(y[:,0]))
        fig2.suptitle('Initial Distribution', fontsize='18')
        
        # Combine Qx, Qy, and d into a single array for sorting
        if use_normalized_coordinates:
            data = list(zip(self._x_norm, self._y_norm, d))
            x_string = '$\sigma_{x}$'
            y_string = '$\sigma_{y}$'
        else:
            data = list(zip(x[:,0], y[:,0], d))
            x_string = 'x [m]'
            y_string = 'y [m]'
        
        # Sort the data based on the 'd' values
        sorted_data = sorted(data, key=lambda x: x[2], reverse=False)
        
        # Unpack the sorted data
        sorted_x, sorted_y, sorted_d = zip(*sorted_data)
        
        if interpolate_initial_distribution:
            Z = griddata((x[:,0], y[:,0]), d, (XX,YY), method='cubic') # linear alternative
            Zm = np.ma.masked_invalid(Z)
            plt.pcolormesh(XX,YY,Zm,cmap=plt.cm.jet)
        else:
            plt.scatter(sorted_x, sorted_y, s=5.5, c=sorted_d, marker='o', lw = 0.1, zorder=10, cmap=plt.cm.jet)  # without interpolation
        
        plt.tick_params(axis='both', labelsize='18')
        plt.xlabel('{}'.format(x_string), fontsize='20')
        plt.ylabel('{}'.format(y_string), fontsize='20')
        plt.clim(-20.5,-4.5)
        cbar=plt.colorbar()
        cbar.set_label('d',fontsize='18')
        cbar.ax.tick_params(labelsize='18')
        fig2.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        fig2.savefig('{}/{}_Initial_distribution.png'.format(self.output_folder, case_name), dpi=250)
        
        if also_show_plot:
            plt.show()
        
        
    def plot_centroid_from_tbt_data(self, x_data=None, y_data=None, load_tbt_data=False, also_show_plot=False):
        """
        Generate centroid plot from turn-by-turn data to observe e.g. synchrotron motion
        
        Parameters:
        ----------
        x_tbt_data, y_tbt_data :  numpy.ndarray
            arrays with turn-by-turn data
        
        Returns:
        --------
        None
        """
        if load_tbt_data:
            x_tbt_data, y_tbt_data, _, _ = self.load_tracking_data()
        else:
            x_tbt_data, y_tbt_data = x_data, y_data
        fig = plt.figure(figsize=(10,7))
        fig.suptitle('Centroid evolution')
        ax = fig.add_subplot(2, 1, 1)  # create an axes object in the figure
        ax.plot(np.mean(x_tbt_data, axis=0), marker='o', color='b', markersize=3)
        ax.set_ylabel("Centroid $X$ [m]")
        ax.set_xlabel("#turns")
        ax = fig.add_subplot(2, 1, 2)  # create a second axes object in the figure
        ax.plot(np.mean(y_tbt_data, axis=0), marker='o', color='y', markersize=3)
        ax.set_ylabel("Centroid $Y$ [m]")
        ax.set_xlabel("#turns")
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        
        if also_show_plot:
            plt.show()
        
        
    def run_SPS(self, load_tbt_data=False, 
                save_tune_data=True, 
                Qy_frac=25,
                make_single_Jy_trace=False,
                use_symmetric_lattice=False,
                add_non_linear_magnet_errors=False
                ):
        """
        Default FMA analysis for SPS Pb ions, plot final results and tune diffusion in initial distribution
        
        Parameters:
        ----------
        load_tbt_data : bool 
            whether to load turn-by-turn (TBT) data from tracking is already saved
        save_tune_data : bool 
            flag to store results Qx, Qy, d from FMA
        Qy_frac : int
            fractional vertical tune 
        make_single_Jy_trace : bool 
            flag to create single trace with unique vertical action
            Jy, with varying action Jx. "Trace" instead of "grid", if uniform beam is used
        use_symmetric_lattice : bool
            flag to use symmetric lattice without QFA and QDA
        add_non_linear_magnet_errors : bool
            whether to add non-linear chromatic errors for SPS
        
        Returns:
        --------
        None
        """
        beamParams = BeamParameters_SPS
        sps_seq = SPS_sequence_maker()
        line0, twiss_sps =  sps_seq.load_xsuite_line_and_twiss(Qy_frac=Qy_frac, use_symmetric_lattice=use_symmetric_lattice,
                                                               add_non_linear_magnet_errors=add_non_linear_magnet_errors)
        
        # Install SC, track particles and observe tune diffusion
        if load_tbt_data:
            try:
                x, y, _, _ = self.load_tracking_data()
            except FileExistsError:
                line = self.install_SC_and_get_line(line0, beamParams)
                particles = self.generate_particles(line, beamParams, make_single_Jy_trace)
                x, y = self.track_particles(particles, line)
        else:
            line = self.install_SC_and_get_line(line0, beamParams)
            particles = self.generate_particles(line, beamParams, make_single_Jy_trace)
            x, y = self.track_particles(particles, line)
  
        # Extract diffusion coefficient from FMA of turn-by-turn data
        if load_tbt_data:
            Qx, Qy, d = self.load_tune_data()
        else:
            Qx, Qy, d = self.run_FMA(x, y, Qmin=self.Q_min_SPS)
            
        if save_tune_data and not load_tbt_data:
            os.makedirs(self.output_folder, exist_ok=True)
            np.save('{}/Qx.npy'.format(self.output_folder), Qx)
            np.save('{}/Qy.npy'.format(self.output_folder), Qy)
            np.save('{}/d.npy'.format(self.output_folder), d)
        
        # Tunes from Twiss
        Qh_set = twiss_sps['qx']
        Qv_set = twiss_sps['qy']
        
        # Add interger tunes to fractional tunes 
        Qx += int(twiss_sps['qx'])
        Qy += int(twiss_sps['qy'])
        
        # Make tune footprint, need plot range
        plot_range  = [[26.0, 26.35], [26.0, 26.35]]
   
        self.plot_FMA(d, Qx, Qy, Qh_set, Qv_set,'SPS', plot_range)
        self.plot_initial_distribution(x, y, d, case_name='SPS')


    def run_SPS_with_beta_beat(self, 
                               load_tbt_data=False,
                               save_tune_data=True,
                               Qy_frac=25, 
                               beta_beat=0.05,
                               make_single_Jy_trace=False,
                               use_symmetric_lattice=False,
                               add_non_linear_magnet_errors=False
                               ):
        """
        Default FMA analysis for SPS Pb ions, plot final results and tune diffusion in initial distribution
        
        Parameters:
        ----------
        load_tbt_data : bool 
            whether to load turn-by-turn (TBT) data from tracking is already saved
        save_tune_data : bool 
            flag to store results Qx, Qy, d from FMA
        Qy_frac : int
            fractional vertical tune
        beta_beat : float 
            relative difference in beta functions (Y for SPS)
        make_single_Jy_trace : bool 
            flag to create single trace with unique vertical action
            Jy, with varying action Jx. "Trace" instead of "grid", if uniform beam is used
        use_symmetric_lattice : bool
            flag to use symmetric lattice without QFA and QDA
        add_non_linear_magnet_errors : bool
            whether to add non-linear chromatic errors for SPS
        
        Returns:
        -------
        None
        """
        beamParams = BeamParameters_SPS
        sps_seq = SPS_sequence_maker()
        line0, twiss_sps =  sps_seq.load_xsuite_line_and_twiss(Qy_frac=Qy_frac, use_symmetric_lattice=use_symmetric_lattice,
                                                               add_non_linear_magnet_errors=add_non_linear_magnet_errors)
        
        # Install SC, track particles and observe tune diffusion
        if load_tbt_data:
            try:
                x, y, _, _ = self.load_tracking_data()
            except FileExistsError:
                line_beat = sps_seq.generate_xsuite_seq_with_beta_beat(beta_beat=beta_beat, line=line0)
                line_SC_beat = self.install_SC_and_get_line(line_beat, beamParams)
                particles = self.generate_particles(line_SC_beat, beamParams, make_single_Jy_trace)
                x, y = self.track_particles(particles, line_SC_beat)
        else:
            line_beat = sps_seq.generate_xsuite_seq_with_beta_beat(beta_beat=beta_beat, line=line0)
            line_SC_beat = self.install_SC_and_get_line(line_beat, beamParams)
            particles = self.generate_particles(line_SC_beat, beamParams, make_single_Jy_trace)
            x, y = self.track_particles(particles, line_SC_beat)
  
        # Extract diffusion coefficient from FMA of turn-by-turn data
        if load_tbt_data:
            Qx, Qy, d = self.load_tune_data()
        else:
            Qx, Qy, d = self.run_FMA(x, y, Qmin=self.Q_min_SPS)
            
        if save_tune_data and not load_tbt_data:
            os.makedirs(self.output_folder, exist_ok=True)
            np.save('{}/Qx.npy'.format(self.output_folder), Qx)
            np.save('{}/Qy.npy'.format(self.output_folder), Qy)
            np.save('{}/d.npy'.format(self.output_folder), d)
         
        # Tunes from Twiss
        Qh_set = twiss_sps['qx']
        Qv_set = twiss_sps['qy']
        
        # Add interger tunes to fractional tunes 
        Qx += int(twiss_sps['qx'])
        Qy += int(twiss_sps['qy'])
        
        # Make tune footprint, need plot range
        plot_range  = [[26.0, 26.35], [26.0, 26.35]]
   
        self.plot_FMA(d, Qx, Qy, Qh_set, Qv_set,'SPS', plot_range)
        self.plot_initial_distribution(x, y, d, case_name='SPS_beta_beat')
        
        

    def run_custom_beam_SPS(self, ion_type, m_ion, Q_SPS, Q_PS,
                            qx, qy, Nb, load_tbt_data=False, beta_beat=None 
                            ):
        """
        FMA analysis for SPS custom beams
        
        Parameters:
        -----------
        ion type : str
            which ion (str)
        m_ion : float
            ion mass in atomic units [u]
        Q_SPS : int 
            SPS ion charge state
        Q_PS : int
            PS ion charge state
        qx : float
            horizontal tune
        qy ; float
            vertical tune
        Nb : int
            bunch intensity (default 'None' will keep default Pb parameters)
        load_tbt_data : bool
            flag if tracking is already done
        
        Returns:
        --------
        None
        """
        beamParams = BeamParameters_SPS
        if Nb is not None:
            beamParams.Nb = Nb 
        s = SPS_sequence_maker(qx0=qx, qy0=qy, m_ion=m_ion, Q_SPS=Q_SPS, Q_PS=Q_PS, ion_type=ion_type)
        line0 = s.generate_xsuite_seq()
        twiss_sps = line0.twiss()
        
        # Install SC, track particles and observe tune diffusion
        if load_tbt_data:
            try:
                x, y, _, _ = self.load_tracking_data()
            except FileExistsError:
                line = self.install_SC_and_get_line(line0, beamParams)
                particles = self.generate_particles(line, beamParams)
                x, y = self.track_particles(particles, line)
        else:
            line = self.install_SC_and_get_line(line0, beamParams)
            particles = self.generate_particles(line, beamParams)
            x, y = self.track_particles(particles, line)
  
        # Extract diffusion coefficient from FMA of turn-by-turn data
        Qx, Qy, d = self.run_FMA(x, y)
         
        # Tunes from Twiss
        Qh_set = twiss_sps['qx']
        Qv_set = twiss_sps['qy']
        print('\nSet tune is located at {:.4f}, {:.4f}\n'.format(Qh_set, Qv_set))
        
        # Add interger tunes to fractional tunes 
        Qx += int(twiss_sps['qx'])
        Qy += int(twiss_sps['qy'])
        
        # Make tune footprint
        plot_range  = [[26.0, 26.35], [26.0, 26.35]]
   
        self.plot_FMA(d, Qx, Qy, Qh_set, Qv_set,'SPS', plot_range)
        self.plot_initial_distribution(x, y, d, case_name='SPS')
        
        
    def run_PS(self, load_tbt_data=False, save_tune_data=True):
        """
        Default FMA analysis for SPS Pb ions, plot final results and tune diffusion in initial distribution
        
        Parameters
        ----------
        load_tbt_data : bool
            flag if turn-by-turn data from tracking is already saved
        
        Returns:
        -------
        None
        """
        beamParams = BeamParameters_PS
        ps_seq = PS_sequence_maker()
        
        line0, twiss_ps = ps_seq.load_xsuite_line_and_twiss()
        
        # Install SC, track particles and observe tune diffusion
        if load_tbt_data:
            try:
                x, y, _, _ = self.load_tracking_data()
            except FileExistsError:
                line = self.install_SC_and_get_line(line0, beamParams)
                particles = self.generate_particles(line, beamParams)
                x, y = self.track_particles(particles, line)
        else:
            line = self.install_SC_and_get_line(line0, beamParams)
            particles = self.generate_particles(line, beamParams)
            x, y = self.track_particles(particles, line)
            
        Qx, Qy, d = self.run_FMA(x, y, Qmin=self.Q_min_PS)

        if save_tune_data:
            os.makedirs(self.output_folder, exist_ok=True)
            np.save('{}/Qx.npy'.format(self.output_folder), Qx)
            np.save('{}/Qy.npy'.format(self.output_folder), Qy)
            np.save('{}/d.npy'.format(self.output_folder), d)


        # Tunes from Twiss
        Qh_set = twiss_ps['qx']
        Qv_set = twiss_ps['qy']
       
        # Add interger tunes to fractional tunes 
        Qx += int(twiss_ps['qx'])
        Qy += int(twiss_ps['qy']) 
       
        # Make tune footprint
        plot_range  = [[6.0, 6.4], [6.0, 6.4]]
   
        self.plot_FMA(d, Qx, Qy, Qh_set, Qv_set,'PS', plot_range) 
        self.plot_initial_distribution(x, y, d, case_name='PS')


    def run_PS_with_beta_beat(self, load_tbt_data=False, beta_beat=0.02):
        """
        Default FMA analysis for PS Pb ions, plot final results and tune diffusion in initial distribution
        
        Parameters
        ----------
        load_tbt_data : bool
            flag whether turn-by-turn data from tracking is already saved
        beta_beat : float
            relative difference in beta functions (X for PS)
        
        Returns:
        -------
        None
        """
        beamParams = BeamParameters_PS
        ps_seq = PS_sequence_maker()
        line0, twiss_ps =  ps_seq.load_xsuite_line_and_twiss()
        
        # Install SC, track particles and observe tune diffusion
        if load_tbt_data:
            try:
                x, y, _, _ = self.load_tracking_data()
            except FileExistsError:
                line_beat = ps_seq.generate_xsuite_seq_with_beta_beat(beta_beat=beta_beat, line=line0)
                line_SC_beat = self.install_SC_and_get_line(line_beat, beamParams)
                particles = self.generate_particles(line_SC_beat, beamParams)
                x, y = self.track_particles(particles, line_SC_beat)
        else:
            line_beat = ps_seq.generate_xsuite_seq_with_beta_beat(beta_beat=beta_beat, line=line0)
            line_SC_beat = self.install_SC_and_get_line(line_beat, beamParams)
            particles = self.generate_particles(line_SC_beat, beamParams)
            x, y = self.track_particles(particles, line_SC_beat)
  
        # Extract diffusion coefficient from FMA of turn-by-turn data
        Qx, Qy, d = self.run_FMA(x, y, Qmin=self.Q_min_PS)
         
        # Tunes from Twiss
        Qh_set = twiss_ps['qx']
        Qv_set = twiss_ps['qy']
        
        # Add interger tunes to fractional tunes 
        Qx += int(twiss_ps['qx'])
        Qy += int(twiss_ps['qy'])
        
        # Make tune footprint, need plot range
        plot_range  = [[6.0, 6.4], [6.0, 6.4]]
   
        self.plot_FMA(d, Qx, Qy, Qh_set, Qv_set,'PS', plot_range)
        self.plot_initial_distribution(x, y, d, case_name='PS_beta_beat')


    def run_custom_beam_PS(self, ion_type, m_ion, Q_LEIR, Q_PS,
                            qx, qy, Nb=None, load_tbt_data=False 
                            ):
        """
        FMA analysis for SPS custom beams
        
        Parameters:
        -----------
        ion type : str
            which ion (str)
        m_ion : float
            ion mass in atomic units [u]
        Q_LEIR : int
            LEIR ion charge state
        Q_PS : int
            PS ion charge state
        qx : float
            horizontal tune
        qy ; float
            vertical tune
        Nb : int
            bunch intensity (default 'None' will keep default Pb parameters)
        load_tbt_data : bool
            flag if tracking is already done
        
        Returns:
        --------
        None, but generates plots
        """
        beamParams = BeamParameters_PS
        if Nb is not None:
            beamParams.Nb = Nb 
        s = PS_sequence_maker(qx0=qx, qy0=qy, m_ion=m_ion, Q_LEIR=Q_LEIR, Q_PS=Q_PS, ion_type=ion_type)
        line0 = s.generate_xsuite_seq()
        twiss_ps = line0.twiss()
        
        # Install SC, track particles and observe tune diffusion
        if load_tbt_data:
            try:
                x, y, _, _ = self.load_tracking_data()
            except FileExistsError:
                line = self.install_SC_and_get_line(line0, beamParams)
                particles = self.generate_particles(line, beamParams)
                x, y = self.track_particles(particles, line)
        else:
            line = self.install_SC_and_get_line(line0, beamParams)
            particles = self.generate_particles(line, beamParams)
            x, y = self.track_particles(particles, line)
              
        Qx, Qy, d = self.run_FMA(x, y)
        
        # Tunes from Twiss
        Qh_set = twiss_ps['qx']
        Qv_set = twiss_ps['qy']
        print('\nSet tune is located at {:.4f}, {:.4f}\n'.format(Qh_set, Qv_set))
        
        # Add interger tunes to fractional tunes 
        Qx += int(twiss_ps['qx'])
        Qy += int(twiss_ps['qy'])
        
        # Make tune footprint
        plot_range  = [[6.0, 6.4], [6.0, 6.4]]
   
        self.plot_FMA(d, Qx, Qy, Qh_set, Qv_set,'PS', plot_range)
        self.plot_initial_distribution(x, y, d, case_name='PS')
        
    