from xpart.longitudinal import generate_longitudinal_coordinates
from xpart import build_particles
import numpy as np

def generate_parabolic_distribution(num_particles, nemitt_x, nemitt_y, sigma_z, Nb, line, _context=None):
    """
    Function to generate a transversely Gaussian and longitudinally parabolic particle distribution.
    
    Parameters:
    -----------
    num_particles : int
        number of macroparticles
    nemitt_x : float
        normalized X emittance
    nemitt_y : float
        normalized Y emittance
    sigma_z : float
        RMS bunch length
    Nb : float
        total bunch intensity
    line : xtrack.Line
        line object used for tracking
    _context : xojects.context
        custom context for tracking that can be provided. Default will be used if 'None' is given
        
    Returns:
    -------
    particles : xp.Particles
    """
    # Generate longitudinal coordinates s
    zeta, delta = generate_longitudinal_coordinates(line=line, distribution='parabolic',
    num_particles=num_particles,
    engine='single-rf-harmonic', sigma_z=sigma_z,
    particle_ref=line.particle_ref, return_matcher=False)
    
    # Initiate normalized coordinates
    x_norm = np.random.normal(size=num_particles)
    px_norm = np.random.normal(size=num_particles)
    y_norm = np.random.normal(size=num_particles)
    py_norm = np.random.normal(size=num_particles)
    
    particles = build_particles(_context=None, particle_ref=line.particle_ref,
    zeta=zeta, delta=delta,
    x_norm=x_norm, px_norm=px_norm,
    y_norm=y_norm, py_norm=py_norm,
    scale_with_transverse_norm_emitt=(nemitt_x, nemitt_y),
    weight=Nb/num_particles, line=line)
    
    return particles
