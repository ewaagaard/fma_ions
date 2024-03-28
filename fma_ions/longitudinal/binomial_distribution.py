from xpart.longitudinal import generate_longitudinal_coordinates
from xpart import build_particles
import numpy as np

def generate_binomial_distribution_from_PS_extr(_context=None, 
							num_particles=None,
                			nemitt_x=None, 
                			nemitt_y=None, 
                			sigma_z=None,
                			particle_ref=None, 
                			total_intensity_particles=None,
                			tracker=None,
                			line=None,
                			return_matcher=False,
                            m=None
                			):

	"""
	Function to generate a parabolic longitudinal distribution from PS extraction
	intended for SPS injection
	"""
	
	if m is None:
		m = 4.7 # typical value for ions at PS extraction

	# Import PS line
	

	# Generate longitudinal coordinates from PS extraction
	zeta, delta, matcher = generate_longitudinal_coordinates(line=line, distribution='binomial', 
							num_particles=num_particles, 
							engine='single-rf-harmonic', sigma_z=sigma_z,
							particle_ref=particle_ref, return_matcher=True, m=m)
	
	# Initiate normalized coordinates 
	x_norm = np.random.normal(size=num_particles)
	px_norm = np.random.normal(size=num_particles)
	y_norm = np.random.normal(size=num_particles)
	py_norm = np.random.normal(size=num_particles)

	# If not provided, use number of particles as intensity 
	if total_intensity_particles is None:   
		total_intensity_particles = num_particles

	particles = build_particles(_context=None, particle_ref=particle_ref,
				zeta=zeta, delta=delta, 
				x_norm=x_norm, px_norm=px_norm,
				y_norm=y_norm, py_norm=py_norm,
				nemitt_x=nemitt_x, nemitt_y=nemitt_y,
				weight=total_intensity_particles/num_particles, line=line)

	if return_matcher:
		return particles, matcher
	else:
		return particles


def generate_binomial_distribution(_context=None, 
							num_particles=None,
                			nemitt_x=None, 
                			nemitt_y=None, 
                			sigma_z=None,
                			particle_ref=None, 
                			total_intensity_particles=None,
                			line=None,
                			return_matcher=False,
                            m=None
                			):

	"""
	Function to generate a parabolic longitudinal distribution for xtrack.Line
	"""
	
	if m is None:
		m = 4.7 # typical value for ions at PS extraction

	# Generate longitudinal coordinates s
	zeta, delta, matcher = generate_longitudinal_coordinates(line=line, distribution='binomial', 
							num_particles=num_particles, 
							engine='single-rf-harmonic', sigma_z=sigma_z,
							particle_ref=particle_ref, return_matcher=True, m=m)
	
	# Initiate normalized coordinates 
	x_norm = np.random.normal(size=num_particles)
	px_norm = np.random.normal(size=num_particles)
	y_norm = np.random.normal(size=num_particles)
	py_norm = np.random.normal(size=num_particles)

	# If not provided, use number of particles as intensity 
	if total_intensity_particles is None:   
		total_intensity_particles = num_particles

	particles = build_particles(_context=_context, particle_ref=particle_ref,
				zeta=zeta, delta=delta, 
				x_norm=x_norm, px_norm=px_norm,
				y_norm=y_norm, py_norm=py_norm,
				nemitt_x=nemitt_x, nemitt_y=nemitt_y,
				weight=total_intensity_particles/num_particles, line=line)

	if return_matcher:
		return particles, matcher
	else:
		return particles
