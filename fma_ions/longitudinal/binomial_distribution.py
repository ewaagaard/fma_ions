from xpart.longitudinal import generate_longitudinal_coordinates
from xpart import build_particles
import xpart as xp
from fma_ions.sequences import PS_sequence_maker
import numpy as np


def generate_binomial_distribution_from_PS_extr(_context=None, 
                                                num_particles=None,
                                                nemitt_x=None, 
                                                nemitt_y=None, 
                                                sigma_z=None,
                                                total_intensity_particles=None,
                                                line=None,
                                                return_separatrix_coord=False,
                                                m=None):
    """
    Function to generate a parabolic longitudinal distribution from PS extraction
    intended for SPS injection
    
    Parameters:
    ----------
    num_particles : int
        number of macroparticles
    nemitt_x : float
        normalized horizontal emittance
    nemitt_y : float
        normalized vertical emittance
    sigma_z : float
        bunch length in meters
    total_intensity_particles : float
        number of ions per bunch
    line : xt.Line
        line object to which generate particle objects
    return_separatrix_coord : bool
        whether to return zeta/delta coordinates of separatrix
    m : float
        binomial parameter for fit
    """
    if m is None:
        m = 4.7  # typical value for ions at PS extraction

    # Import PS line
    ps = PS_sequence_maker()
    ps_line, _ = ps.load_xsuite_line_and_twiss(at_injection_energy=False)

    # Generate longitudinal coordinates from PS extraction
    zeta, delta, _ = generate_longitudinal_coordinates(line=ps_line, distribution='binomial', 
                                                             num_particles=num_particles, 
                                                             engine='single-rf-harmonic', sigma_z=sigma_z,
                                                             particle_ref=ps_line.particle_ref, return_matcher=True, m=m)
    
    # Get a separate matcher for SPS
    _, _, matcher = generate_longitudinal_coordinates(line=line, distribution='binomial', 
                                                             num_particles=num_particles, 
                                                             engine='single-rf-harmonic', sigma_z=sigma_z,
                                                             particle_ref=line.particle_ref, return_matcher=True, m=m)
    
    # Get separatrix coordinates from matcher
    if return_separatrix_coord:
        ufp = matcher.get_unstable_fixed_point()
        xx = np.linspace(-ufp, ufp, 1000)
        yy = np.sqrt(2*matcher.A/matcher.C) * np.cos(matcher.B/2.*xx)
        
        zeta_separatrix = np.array(line.particle_ref._xobject.beta0[0]) * np.array(xx)  # zeta
        temp_particles = xp.Particles(p0c=line.particle_ref._xobject.p0c[0],
                                   zeta=zeta_separatrix, ptau=yy)
        delta_separatrix = np.array(temp_particles.delta)

    # Initiate normalized coordinates 
    x_norm = np.random.normal(size=num_particles)
    px_norm = np.random.normal(size=num_particles)
    y_norm = np.random.normal(size=num_particles)
    py_norm = np.random.normal(size=num_particles)

    # If not provided, use number of particles as intensity 
    if total_intensity_particles is None:   
        total_intensity_particles = num_particles

    particles = build_particles(_context=_context, particle_ref=line.particle_ref,
                                zeta=zeta, delta=delta, 
                                x_norm=x_norm, px_norm=px_norm,
                                y_norm=y_norm, py_norm=py_norm,
                                nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                                weight=total_intensity_particles/num_particles, line=line)

    print('Generated binomially longitudinal particles with m = {} and sigma_z = {:.3f} m'.format(m, np.std(particles.zeta)))

    if return_separatrix_coord:
        return particles, zeta_separatrix, delta_separatrix
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
