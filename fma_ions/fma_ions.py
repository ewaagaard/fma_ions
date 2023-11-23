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

from .fma_data_classes import BeamParameters_PS, BeamParameters_SPS, PS_Sequence, SPS_Sequence

@dataclass
class FMA:
    """
    Performs tracking and frequency map analysis
    
    Parameters:
    num_turns - to track in total
    num_spacecharge_interactions - how many interactions per turn
    tol_spacecharge_position - tolerance in placement of SC kicks along line
    mode - space charge model: 'frozen', 'semi-frozen' or 'pic' (frozen recommended)
    n_theta - number of divisions for theta coordinates for particles in normalized coordinates
    n_r - number of divisions for r coordinates for particles in normalized coordinates
    output_folder - where to save data
    """
    num_turns: int = 1200
    num_spacecharge_interactions: int = 160 
    tol_spacecharge_position: float = 1e-2
    n_theta: int = 50
    n_r: int = 100
    mode: str = 'frozen'
    output_folder: str = 'output_fma'
    plot_order: int = 4
    periodicity: int = 16
    tracking_data_exists: bool = False
    

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

        ##### Generate particles #####
        x_norm, y_norm, _, _ = xp.generate_2D_polar_grid(
                                                        theta_range=(0.01, np.pi/2-0.01),
                                                        ntheta = self.n_theta,
                                                        r_range = (0.1, 7),
                                                        nr = self.n_r)
        # Build the particle object
        particles = xp.build_particles(line = line, particle_ref = line.particle_ref,
                                       x_norm=x_norm, y_norm=y_norm, delta=0,
                                       nemitt_x = beamParams.exn, nemitt_y = beamParams.eyn)
        print('\nBuilt particle object of size {}...'.format(len(particles.x)))
        
        # Track the particles and return turn-by-turn coordinates
        print('\nStarting tracking...')
        #line.track(particles, num_turns = self.num_turns, turn_by_turn_monitor=True, with_progress=True)
        x, y = np.zeros([len(particles.x), self.num_turns]), np.zeros([len(particles.x), self.num_turns])
        
        i = 0
        for turn in range(self.num_turns):
            if i % 20 == 0:
                print('Tracking turn {}'.format(i))
        
            # Save the turn-by-turn data
            x[:, i] = particles.x
            y[:, i] = particles.y

            # Track the particles
            line.track(particles)
            i += 1
        
        print('Finished tracking.\n')
        print('Average X and Y of tracking: {} and {}'.format(np.mean(x), np.mean(y)))

        if save_data:
            os.makedirs(self.output_folder, exist_ok=True)
            np.save('{}/x.npy'.format(self.output_folder), x)
            np.save('{}/y.npy'.format(self.output_folder), y)
            self.tracking_data_exists = True
            print('Saved tracking data.')
            
        return x, y
    
    
    def load_tracking_data(self):
        """Loads numpy data if tracking has already been made"""
        try:
            x=np.load('output_fma/x.npy')
            y=np.load('output_fma/y.npy')
            self.tracking_data_exists = True
            return x, y
        except FileNotFoundError:
            print('Tracking data does not exist!')
            self.tracking_data_exists = True
            pass
        
        
    def run_FMA(self, x_tbt_data, y_tbt_data):
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
        # Initialize empty arrays for tunes of particles, during first and second half of run - at split_ind
        Qx_1, Qy_1 = np.zeros(len(x_tbt_data)), np.zeros(len(x_tbt_data))
        Qx_2, Qy_2 = np.zeros(len(x_tbt_data)), np.zeros(len(x_tbt_data))
        split_ind = int(self.num_turns/2)
        
        # Iterate over particles to find tune
        for i_part in range(len(x_tbt_data)):
            
            # Find dominant frequency with NAFFlib - also remember to subtract mean 
            Qx_1[i_part] = NAFFlib.get_tune(x_tbt_data[i_part, :split_ind] \
                                            - np.mean(x_tbt_data[i_part, :split_ind]))  
            Qy_1[i_part] = NAFFlib.get_tune(y_tbt_data[i_part, :split_ind] \
                                            - np.mean(y_tbt_data[i_part, :split_ind]))
            Qx_2[i_part] = NAFFlib.get_tune(x_tbt_data[i_part, split_ind:] \
                                            - np.mean(x_tbt_data[i_part, split_ind:]))  
            Qy_2[i_part] = NAFFlib.get_tune(y_tbt_data[i_part, :split_ind] \
                                            - np.mean(y_tbt_data[i_part, :split_ind]))  
        
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
        
        
    def run_SPS(self, 
                load_tbt_data=False,
                ):
        """Default FMA analysis for SPS Pb ions"""
        beamParams = BeamParameters_SPS
        line = SPS_Sequence.sps_line
        
        # Install SC, track particles and observe tune diffusion
        x, y = self.load_tracking_data() if load_tbt_data else self.install_SC_and_track(line, beamParams)
        d, Qx, Qy = self.run_FMA(x, y)
        
        # Add interger tunes to fractional tunes 
        Qx += beamParams().Q_int
        Qy += beamParams().Q_int
        
        # Tunes from Twiss
        Qh_set = SPS_Sequence.twiss_sps['qx']
        Qv_set = SPS_Sequence.twiss_sps['qy']
        
        # Make tune footprint
        plot_range  = [[26.0, 26.35], [26.0, 26.35]]
   
        self.plot_FMA(x, y, d, Qx, Qy, Qh_set, Qv_set,'SPS', plot_range)
     
        
    def run_PS(self, load_tbt_data=False):
        """Default FMA analysis for PS Pb ions"""
        beamParams = BeamParameters_PS
        line = PS_Sequence.ps_line
        
        # Install SC, track particles and observe tune diffusion
        x, y = self.load_tracking_data() if load_tbt_data else self.install_SC_and_track(line, beamParams)
        d, Qx, Qy = self.run_FMA(x, y)
        
        # Add interger tunes to fractional tunes 
        Qx += beamParams().Q_int
        Qy += beamParams().Q_int
        
        # Tunes from Twiss
        Qh_set = PS_Sequence.twiss_ps['qx']
        Qv_set = PS_Sequence.twiss_ps['qy']
        
        # Make tune footprint
        plot_range  = [[6.0, 6.4], [6.0, 6.4]]
   
        self.plot_FMA(x, y, d, Qx, Qy, Qh_set, Qv_set,'PS', plot_range) 

    