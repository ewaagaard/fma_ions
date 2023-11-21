"""
Main class to perform Frequency Map Analysis
"""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import constants
import xtrack as xt
import xpart as xp
import xfields as xf
import xobjects as xo
import os 

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
plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# from statisticalEmittance.statisticalEmittance import statisticalEmittance 
from PySCRDT import PySCRDT, resonance_lines, tune_footprint_maker
import NAFFlib

from .fma_data_classes import BeamParameters_PS, BeamParameters_SPS, PS_Sequence, SPS_Sequence


class FMA:
    """
    Performs tracking and frequency map analysis
    
    Parameters:
    num_turns - to track in total
    num_spacecharge_interactions - how many interactions per turn
    tol_spacecharge_position - tolerance in placement of SC kicks along line
    """
    def __init__(self, 
                 num_turns = 1200,
                 num_particles = 5000,
                 num_spacecharge_interactions = 160, 
                 tol_spacecharge_position = 1e-2,
                 mode = 'frozen',
                 output_folder = 'output_fma'
                 ):
        self.num_turns = num_turns
        self.num_particles = num_particles
        self.num_spacecharge_interactions = num_spacecharge_interactions
        self.tol_spacecharge_position = tol_spacecharge_position
        self.mode = mode
        self.context = xo.ContextCpu()  # to be upgrade to GPU if needed 
        self.output_folder = output_folder
        self.tracking_data_exists = False

        # Tune footprint details
        self.plot_order = 4
        self.periodicity = 16
    
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
        line.build_tracker(_context = self.context)

        ##### Generate particles #####
        x_norm, y_norm, _, _ = xp.generate_2D_polar_grid(
                                                        theta_range=(0.01, np.pi/2-0.01),
                                                        ntheta = 50,
                                                        r_range = (0.1, 7),
                                                        nr = 80)
        # Build the particle object
        particles = xp.build_particles(line = line, particle_ref = line.particle_ref,
                                       x_norm=x_norm, y_norm=y_norm, delta=0,
                                       nemitt_x = beamParams.exn, nemitt_y = beamParams.eyn)
        print('\nBuilt particle object of size {}...'.format(len(particles.x)))
        
        # Track the particles and return turn-by-turn coordinates
        print('\nStarting tracking...')
        line.track(particles, num_turns = self.num_turns, turn_by_turn_monitor=True, with_progress=True)
        print('Finished tracking.\n')
    
        x = line.record_last_track.x
        y = line.record_last_track.y
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
    

    def run_SPS(self):
        """Default FMA analysis for SPS Pb ions"""
        beamParams = BeamParameters_SPS
        line = SPS_Sequence.sps_line
        
        # Install SC, track particles and observe tune diffusion
        x, y = self.install_SC_and_track(line, beamParams)
        d, Qx, Qy = self.run_FMA(x, y)
        
        # Add interger tunes to fractional tunes 
        Qx += beamParams().Q_int
        Qy += beamParams().Q_int
        
        # Make tune footprint
        plot_range  = [[26.0, 26.35], [26.0, 26.35]]
   
        fig, ax = plt.figure(figsize=(8,6))
        tune_diagram = resonance_lines(plot_range[0],
                    plot_range[1], np.arange(1, self.plot_order+1), self.periodicity)
        tune_diagram.plot_resonance(figure_object = fig, interactive=False)

        plt.scatter(Qx, Qy, 4, d, 'o', lw = 0.1, zorder=10, cmap=plt.cm.jet)
        plt.xlabel('$\mathrm{Q_x}$')
        plt.ylabel('$\mathrm{Q_y}$')
        cbar=plt.colorbar()
        cbar.set_label('d',fontsize='18')
        cbar.ax.tick_params(labelsize='18')
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        fig.savefig('{}/FMA_plot_SPS.png'.format(self.output_folder), dpi=250)
        plt.show()
        
        
    def run_PS(self):
        """Default FMA analysis for PS Pb ions"""
        beamParams = BeamParameters_PS
        line = PS_Sequence.ps_line
        
        # Install SC, track particles and observe tune diffusion
        x, y = self.install_SC_and_track(line, beamParams)
        d, Qx, Qy = self.run_FMA(x, y)
        
        # Add interger tunes to fractional tunes 
        Qx += beamParams().Q_int
        Qy += beamParams().Q_int
        
        # Make tune footprint
        plot_range  = [[6.0, 6.4], [6.0, 6.4]]
   
        fig, ax = plt.figure(figsize=(8,6))
        tune_diagram = resonance_lines(plot_range[0],
                    plot_range[1], np.arange(1, self.plot_order+1), self.periodicity)
        tune_diagram.plot_resonance(figure_object = fig, interactive=False)

        plt.scatter(Qx, Qy, 4, d, 'o', lw = 0.1, zorder=10, cmap=plt.cm.jet)
        plt.xlabel('$\mathrm{Q_x}$')
        plt.ylabel('$\mathrm{Q_y}$')
        cbar=plt.colorbar()
        cbar.set_label('d',fontsize='18')
        cbar.ax.tick_params(labelsize='18')
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        fig.savefig('{}/FMA_plot_PS.png'.format(self.output_folder), dpi=250)
        plt.show()
    