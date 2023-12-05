"""
Main class to investigate Dynamic Aperture (DA) studies in PS and SPS
"""
from dataclasses import dataclass
import numpy as np
import os

import xtrack as xt
import xpart as xp
import xobjects as xo

from .fma_ions import FMA
from .fma_data_classes import BeamParameters_PS, BeamParameters_SPS, Sequences
from .sequence_classes_ps import PS_sequence_maker
from .sequence_classes_sps import SPS_sequence_maker

@dataclass
class DA: 
    """
    Main class to study DA of provided sequence 
    
    Parameters:
    ----------
    use_uniform_beam - if True generate a transverse pencil distribution, otherwise 2D polar grid
    num_turns - to track in total
    delta0 - relative momentum offset dp/p
    z0 - initial longitudinal offset zeta
    n_theta - number of divisions for theta coordinates for particles in normalized coordinates
    n_r - number of divisions for r coordinates for particles in normalized coordinates
    n_linear - default number of points if uniform linear grid for normalized X and Y are used
    n_sigma - max number of beam sizes sigma to generate particles
    output_folder - where to save data
    qx, qy - horizontal and vertical tunes, if customized tune is desired
    """
    use_uniform_beam: bool = True
    num_turns: int = 1000
    delta0: float = 0.0
    z0: float = 0.0
    n_theta: int = 50
    n_r: int = 100
    n_linear: int = 100
    n_sigma: float = 50.0
    output_folder: str = 'output_DA'
    qx: float = None
    qy: float = None
    
    
    def build_tracker_and_generate_particles(self, line, beamParams):
        """
        Install Space Charge (SC) and generate particles with provided Xsuite line and beam parameters
        
        Parameters:
        ----------
        line - xsuite line to track through
        beamParams - beam parameters (data class containing Nb, sigma_z, exn, eyn)
        
        Returns:
        -------
        x, y- numpy arrays containing turn-by-turn data coordinates
        """
        context = xo.ContextCpu()  # to be upgrade to GPU if needed 
        
        # Build tracker for line
        line.build_tracker(_context = context)
        line.optimize_for_tracking()
        twiss = line.twiss()

        ##### Generate particles #####
        print('\nGenerating particles with delta = {:.2e} and z = {:.2e}'.format(
            0, self.z0))
        if self.use_uniform_beam:     
            print('Making UNIFORM distribution...')
            # Generate arrays of normalized coordinates 
            x_values = np.linspace(0.1, self.n_sigma, num=self.n_linear)  
            y_values = np.linspace(0.1, self.n_sigma, num=self.n_linear)  

            # Create a meshgrid for the uniform beam distribution
            X, Y = np.meshgrid(x_values, y_values)
            x_norm, y_norm = X.flatten(), Y.flatten()
            
        else:
            print('Making POLAR distribution...')
            x_norm, y_norm, _, _ = xp.generate_2D_polar_grid(
                                                            theta_range=(0.01, np.pi/2-0.01),
                                                            ntheta = self.n_theta,
                                                            r_range = (0.1, 7),
                                                            nr = self.n_r)
        # Store normalized coordinates
        self._x_norm, self._y_norm = x_norm, y_norm
            
        # Build the particle object
        particles = xp.build_particles(line = line, particle_ref = line.particle_ref,
                                       x_norm=x_norm, y_norm=y_norm, delta=self.delta0, zeta=self.z0,
                                       nemitt_x = beamParams.exn, nemitt_y = beamParams.eyn)
        
        print('\nBuilt particle object of size {}...'.format(len(particles.x)))
        
        return line, particles
    
    
    def track_particles(self, particles, line, save_tbt_data=True):
        """
        Track particles through lattice with space charge elments installed
        
        Parameters:
        ----------
        particles - particles object from xpart
        line - xsuite line to track through, where space charge has been installed 
        
        Returns:
        -------
        x, y - numpy arrays containing turn-by-turn data coordinates
        """          
        #### TRACKING #### 
        # Track the particles and return turn-by-turn coordinates
        state = np.zeros([len(particles.state), self.num_turns])
        
        print('\nStarting tracking...')
        i = 0
        for turn in range(self.num_turns):
            if i % 20 == 0:
                print('Tracking turn {}'.format(i))
        
            state[:, i] = particles.state
        
            # Track the particles
            line.track(particles)
            i += 1
        
        print('Finished tracking.\n')
        print('{} out of {} particles survived'.format(sum(particles.state > 0), len(particles.state)))
        
        if save_tbt_data:
            os.makedirs(self.output_folder, exist_ok=True)
            np.save('{}/state.npy'.format(self.output_folder), state)
            np.save('{}/x0_norm.npy'.format(self.output_folder), self._x_norm)
            np.save('{}/y0_norm.npy'.format(self.output_folder), self._y_norm)
            print('Saved tracking data.')
    
        return particles.state
    

    def run_SPS(self, 
                load_tbt_data=False,
                use_default_tunes=True
                ):
        """Default FMA analysis for SPS Pb ions"""
        
        beamParams = BeamParameters_SPS
        
        # Load SPS lattice with default tunes, or custom tunes
        if use_default_tunes:
            line, twiss_sps = Sequences.get_SPS_line_and_twiss()
        else:
            s = SPS_sequence_maker(qx0=self.qx, qy0=self.qy)
            line = s.generate_xsuite_seq()
            twiss_sps = line.twiss()
        
        # Install SC, track particles and observe tune diffusion
        if load_tbt_data:
            try:
                state = self.load_tracking_data()
            except FileExistsError:
                print('\nCannot load data!\n')
        else:
            line, particles = self.build_tracker_and_generate_particles(line, beamParams)
            state = self.track_particles(particles, line)