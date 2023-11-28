"""q
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

#from statisticalEmittance import statisticalEmittance 

##### Plot settings 
import matplotlib.pyplot as plt
SMALL_SIZE = 18
MEDIUM_SIZE = 21
BIGGER_SIZE = 26
plt.rcParams["font.family"] = "serif"
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)   # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# from statisticalEmittance.statisticalEmittance import statisticalEmittance 
from PySCRDT import resonance_lines
import NAFFlib

from .fma_data_classes import BeamParameters_PS, BeamParameters_SPS, Sequences
from .sequence_classes_ps import PS_sequence_maker
from .sequence_classes_sps import SPS_sequence_maker
#from .statistical_emittance import StatisticalEmittance

@dataclass
class FMA:
    """
    Performs tracking and frequency map analysis
    
    Parameters:
    num_turns - to track in total
    num_spacecharge_interactions - how many interactions per turn
    tol_spacecharge_position - tolerance in placement of SC kicks along line
    delta0 - relative momentum offset dp/p
    mode - space charge model: 'frozen', 'semi-frozen' or 'pic' (frozen recommended)
    n_theta - number of divisions for theta coordinates for particles in normalized coordinates
    n_r - number of divisions for r coordinates for particles in normalized coordinates
    output_folder - where to save data
    plot_order - order to include in resonance diagram
    periodicity - periodicity in tune diagram
    Q_min_SPS, Q_min_PS - min tune to filter out from synchrotron frequency
    """
    num_turns: int = 1200
    num_spacecharge_interactions: int = 160 
    tol_spacecharge_position: float = 1e-2
    delta0: float = 0.0
    n_theta: int = 50
    n_r: int = 100
    mode: str = 'frozen'
    output_folder: str = 'output_fma'
    plot_order: int = 4
    periodicity: int = 16
    Q_min_SPS: float = 0.05
    Q_min_PS: float = 0.015
    
    def install_SC_and_track(self, line, beamParams, save_data=True):
        """
        Install Space Charge (SC) and tracks particles with provided Xsuite line and beam parameters
        
        Parameters:
        ----------
        line - xsuite line to track through
        beamParams - beam parameters (data class containing Nb, sigma_z, exn, eyn)
        
        Returns:
        -------
        x, y- numpy arrays containing turn-by-turn data coordinates
        """
        context = xo.ContextCpu()  # to be upgrade to GPU if needed 
        
        # Initialize longitudinal profile for beams 
        lprofile = xf.LongitudinalProfileQGaussian(
                number_of_particles = beamParams.Nb,
                sigma_z = beamParams.sigma_z,
                z0=0.,
                q_parameter=1.)

        # Install frozen space charge as base 
        xf.install_spacecharge_frozen(line = line,
                           particle_ref = line.particle_ref,
                           longitudinal_profile = lprofile,
                           nemitt_x = beamParams.exn, nemitt_y = beamParams.eyn,
                           sigma_z = beamParams.sigma_z,
                           num_spacecharge_interactions = self.num_spacecharge_interactions,
                           tol_spacecharge_position = self.tol_spacecharge_position)
        
        # Build tracker for line
        line.build_tracker(_context = context)
        twiss_xtrack_with_sc = line.twiss()

        ##### Generate particles #####
        x_norm, y_norm, _, _ = xp.generate_2D_polar_grid(
                                                        theta_range=(0.01, np.pi/2-0.01),
                                                        ntheta = self.n_theta,
                                                        r_range = (0.1, 7),
                                                        nr = self.n_r)
        # Build the particle object
        particles = xp.build_particles(line = line, particle_ref = line.particle_ref,
                                       x_norm=x_norm, y_norm=y_norm, delta=self.delta0,
                                       nemitt_x = beamParams.exn, nemitt_y = beamParams.eyn)
        print('\nBuilt particle object of size {}...'.format(len(particles.x)))
        
        #### TRACKING #### 
        # Track the particles and return turn-by-turn coordinates
        x, y = np.zeros([len(particles.x), self.num_turns]), np.zeros([len(particles.x), self.num_turns])
        
        # If off-momentum particles are used, need to use betatronic matrices to correct for dispersive components
        use_betatronic_matrices = False if self.delta0 == 0.0 else True
    
        
        i = 0
        print('\nParticles are off-momentum: {}'.format(use_betatronic_matrices))
        print('\nStarting tracking...')
        for turn in range(self.num_turns):
            if i % 20 == 0:
                print('Tracking turn {}'.format(i))
        
            #else: 
            x[:, i] = particles.x
            y[:, i] = particles.y

            # Track the particles
            line.track(particles)
            i += 1
        
        # Correct for dispersion if off-momentum, else simple TBT data
        if use_betatronic_matrices:
            x_noDP = x - twiss_xtrack_with_sc['dx'][0]*(x.transpose() - self.delta0).transpose()
            y_noDP = y - twiss_xtrack_with_sc['dy'][0]*(y.transpose() - self.delta0).transpose()
        
        print('Finished tracking.\n')
        print('Average X and Y of tracking: {} and {}'.format(np.mean(x), np.mean(y)))

        if save_data:
            os.makedirs(self.output_folder, exist_ok=True)
            np.save('{}/x.npy'.format(self.output_folder), x)
            np.save('{}/y.npy'.format(self.output_folder), y)
            if use_betatronic_matrices:
                np.save('{}/x_noDP.npy'.format(self.output_folder), x_noDP)
                np.save('{}/y_noDP.npy'.format(self.output_folder), y_noDP)
            print('Saved tracking data.')
        
        if use_betatronic_matrices:
            return x_noDP, y_noDP 
        else:
            return x, y
    
    def load_tracking_data(self):
        """Loads numpy data if tracking has already been made"""
        try:
            if self.delta0 == 0.0:
                print('\nLoading on-momentum data!\n')
                x=np.load('{}/x.npy'.format(self.output_folder))
                y=np.load('{}/y.npy'.format(self.output_folder))
                return x, y
            else:
                print('\nLoading off-momentum data!\n')
                x_noDP = np.load('{}/x_noDP.npy'.format(self.output_folder))
                y_noDP = np.load('{}/y_noDP.npy'.format(self.output_folder))
                return x_noDP, y_noDP
        except FileNotFoundError:
            print('Tracking data does not exist!')
            pass
        
        
    def run_FMA(self, x_tbt_data, y_tbt_data, Qmin=None):
        """
        Run FMA analysis for given turn-by-turn coordinates
        
        Parameters:
        ----------
        x_tbt_data, y_tbt_data - numpy arrays of turn-by-turn data for particles 
        
        Returns: 
        -------
        d - numpy array containing diffusion for all particles
        Qx, Qy - numpy arrays containing final tunes of all particles
        """
        if Qmin is None:
            Qmin = self.Q_min_SPS 
        
        # Initialize empty arrays for tunes of particles, during first and second half of run - at split_ind
        Qx_1, Qy_1 = np.zeros(len(x_tbt_data)), np.zeros(len(x_tbt_data))
        Qx_2, Qy_2 = np.zeros(len(x_tbt_data)), np.zeros(len(x_tbt_data))
        split_ind = int(self.num_turns/2)
        
        # Iterate over particles to find tune
        for i_part in range(len(x_tbt_data)):
            
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
        
        # Find FMA diffusion of tunes
        d = np.log(np.sqrt( (Qx_2 - Qx_1)**2 + (Qy_2 - Qy_1)**2))
    
        return d, Qx_2, Qy_2
       
    def plot_FMA(self, x, y, d, Qx, Qy, Qh_set, Qv_set, case_name, 
                 plot_range, plot_initial_distribution=True):   
        """
        Plots FMA diffusion and possibly initial distribution
        
        x, y, d, Qx, Qy - input data from self.run_FMA
        Qx_set, Qy_set - input data from Twiss
        case_name - name of accelerator
        """
        fig = plt.figure(figsize=(8,6))
        tune_diagram = resonance_lines(plot_range[0],
                    plot_range[1], np.arange(1, self.plot_order+1), self.periodicity)
        tune_diagram.plot_resonance(figure_object = fig, interactive=False)

        plt.scatter(Qx, Qy, 5.5, d, 'o', lw = 0.1, zorder=10, cmap=plt.cm.jet)
        plt.plot(Qh_set, Qv_set, 'ro', markersize=11, label="Set tune")
        plt.xlabel('$Q_{x}$')
        plt.ylabel('$Q_{y}$')
        cbar=plt.colorbar()
        cbar.set_label('d',fontsize='18')
        cbar.ax.tick_params(labelsize='18')
        plt.legend(loc='upper left')
        plt.clim(-20.5,-4.5)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        fig.savefig('{}/FMA_plot_{}.png'.format(self.output_folder, case_name), dpi=250)
        
        # Plot initial distribution, interpolating between discrete points 
        if plot_initial_distribution:
            fig2=plt.figure(figsize=(8,6))
            XX,YY = np.meshgrid(np.unique(x[:,0]), np.unique(y[:,0]))
            Z = griddata((x[:,0], y[:,0]), d, (XX,YY), method='cubic') # linear alternative
            Zm = np.ma.masked_invalid(Z)
            fig2.suptitle('Initial Distribution', fontsize='18')
            plt.pcolormesh(XX,YY,Zm,cmap=plt.cm.jet)
            # plt.scatter(x[:,0],y[:,0],4, d, 'o',lw = 0.1,zorder=10, cmap=plt.cm.jet)  # without interpolation
            plt.tick_params(axis='both', labelsize='18')
            plt.xlabel('x [m]', fontsize='20')
            plt.ylabel('y [m]', fontsize='20')
            plt.clim(-20.5,-4.5)
            cbar=plt.colorbar()
            cbar.set_label('d',fontsize='18')
            cbar.ax.tick_params(labelsize='18')
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            fig2.savefig('{}/{}_Initial_distribution.png'.format(self.output_folder, case_name), dpi=250)
        
        plt.show()
        
        
    def plot_centroid_from_tbt_data(self, x_data=None, y_data=None, load_tbt_data=False):
        """
        Generate centroid plot from turn-by-turn data to observe e.g. synchrotron motion
        
        Parameters:
        ----------
        x_tbt_data, y_tbt_data
        """
        if load_tbt_data:
            x_tbt_data, y_tbt_data = self.load_tracking_data()
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
        plt.show()
        
        
    def run_SPS(self, 
                load_tbt_data=False,
                ):
        """Default FMA analysis for SPS Pb ions"""
        beamParams = BeamParameters_SPS
        line, twiss_sps = Sequences.get_SPS_line_and_twiss()
        
        # Install SC, track particles and observe tune diffusion
        x, y = self.load_tracking_data() if load_tbt_data else self.install_SC_and_track(line, beamParams)
        d, Qx, Qy = self.run_FMA(x, y, Qmin=self.Q_min_SPS)
        
        # Add interger tunes to fractional tunes 
        Qx += beamParams().Q_int
        Qy += beamParams().Q_int
        
        # Tunes from Twiss
        Qh_set = twiss_sps['qx']
        Qv_set = twiss_sps['qy']
        
        # Make tune footprint
        plot_range  = [[26.0, 26.35], [26.0, 26.35]]
   
        self.plot_FMA(x, y, d, Qx, Qy, Qh_set, Qv_set,'SPS', plot_range)


    def run_custom_beam_SPS(self, ion_type, m_ion, Q_SPS, Q_PS,
                            qx, qy, Nb, load_tbt_data=False 
                            ):
        """
        FMA analysis for SPS custom beams
        
        Parameters:
        -----------
        ion type - which ion (str)
        m_ion - ion mass in atomic units [u]
        Q_SPS - SPS ion charge state
        Q_PS - PS ion charge state
        qx - horizontal tune
        qy - vertical tune
        Nb - bunch intensity (default 'None' will keep default Pb parameters)
        load_tbt_data - bool if tracking is already done
        
        Returns:
        --------
        None, but generates plots
        """
        beamParams = BeamParameters_SPS
        if Nb is not None:
            beamParams.Nb = Nb 
        s = SPS_sequence_maker(qx0=qx, qy0=qy, m_ion=m_ion, Q_SPS=Q_SPS, Q_PS=Q_PS, ion_type=ion_type)
        line = s.generate_xsuite_seq()
        twiss_sps = line.twiss()
        
        # Install SC, track particles and observe tune diffusion
        x, y = self.load_tracking_data() if load_tbt_data else self.install_SC_and_track(line, beamParams)
        d, Qx, Qy = self.run_FMA(x, y)
        
        # Add interger tunes to fractional tunes 
        Qx += beamParams().Q_int
        Qy += beamParams().Q_int
        
        # Tunes from Twiss
        Qh_set = twiss_sps['qx']
        Qv_set = twiss_sps['qy']
        
        # Make tune footprint
        plot_range  = [[26.0, 26.35], [26.0, 26.35]]
   
        self.plot_FMA(x, y, d, Qx, Qy, Qh_set, Qv_set,'SPS', plot_range)
        
        
    def run_PS(self, load_tbt_data=False):
        """Default FMA analysis for PS Pb ions"""
        beamParams = BeamParameters_PS
        line, twiss_ps = Sequences.get_PS_line_and_twiss()
        
        # Install SC, track particles and observe tune diffusion
        x, y = self.load_tracking_data() if load_tbt_data else self.install_SC_and_track(line, beamParams)
        d, Qx, Qy = self.run_FMA(x, y, Qmin=self.Q_min_PS)
        
        # Add interger tunes to fractional tunes 
        Qx += beamParams().Q_int
        Qy += beamParams().Q_int
        
        # Tunes from Twiss
        Qh_set = twiss_ps['qx']
        Qv_set = twiss_ps['qy']
        
        # Make tune footprint
        plot_range  = [[6.0, 6.4], [6.0, 6.4]]
   
        self.plot_FMA(x, y, d, Qx, Qy, Qh_set, Qv_set,'PS', plot_range) 


    def run_custom_beam_PS(self, ion_type, m_ion, Q_LEIR, Q_PS,
                            qx, qy, Nb=None, load_tbt_data=False 
                            ):
        """
        FMA analysis for SPS custom beams
        
        Parameters:
        -----------
        ion type - which ion (str)
        m_ion - ion mass in atomic units [u]
        Q_LEIR - LEIR ion charge state
        Q_PS - PS ion charge state
        qx - horizontal tune
        qy - vertical tune
        Nb - bunch intensity (default 'None' will keep default Pb parameters)
        load_tbt_data - bool if tracking is already done
        
        Returns:
        --------
        None, but generates plots
        """
        beamParams = BeamParameters_PS
        if Nb is not None:
            beamParams.Nb = Nb 
        s = PS_sequence_maker(qx0=qx, qy0=qy, m_ion=m_ion, Q_LEIR=Q_LEIR, Q_PS=Q_PS, ion_type=ion_type)
        line = s.generate_xsuite_seq()
        twiss_ps = line.twiss()
        
        # Install SC, track particles and observe tune diffusion
        x, y = self.load_tracking_data() if load_tbt_data else self.install_SC_and_track(line, beamParams)
        d, Qx, Qy = self.run_FMA(x, y)
        
        # Add interger tunes to fractional tunes 
        Qx += beamParams().Q_int
        Qy += beamParams().Q_int
        
        # Tunes from Twiss
        Qh_set = twiss_ps['qx']
        Qv_set = twiss_ps['qy']
        
        # Make tune footprint
        plot_range  = [[6.0, 6.4], [6.0, 6.4]]
   
        self.plot_FMA(x, y, d, Qx, Qy, Qh_set, Qv_set,'PS', plot_range)
    