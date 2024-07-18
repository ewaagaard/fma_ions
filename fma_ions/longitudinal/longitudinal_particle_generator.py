"""
Container for particle generator class with different distributions
"""
from xpart.longitudinal import generate_longitudinal_coordinates
from xpart import build_particles
import xpart as xp
import xobjects as xo
from fma_ions.sequences import PS_sequence_maker
import numpy as np


def generate_particles_transverse_gaussian(beamParams, line, longitudinal_distribution_type, num_part, _context=None,
                                           matched_for_PS_extraction=False)->xp.Particles:
    
    """
    Function to generate a transverse Gaussian distribution with desired longitudinal distribution,
    either matched to PS extraction or SPS injection
    
    Parameters:
    ----------
    beamParams : dataclass
        container of exn, eyn, Nb and sigma_z. Default 'None' will load nominal SPS beam parameters 
    line : xt.Line
        line object to which generate particle objects
    longitudinal_distribution_type : str
        'gaussian', 'qgaussian', 'binomial' or 'parabolic'
    num_particles : int
        number of macroparticles
    context : xo.context
    matched_for_PS_extraction : bool
        whether to match particles for injection of line (SPS) or extraction of PS

    Returns:
    --------
    particles : xp.Particles
    """
    if _context is None:
        _context = xo.ContextCpu(omp_num_threads='auto')
        print('CPU context generated')
    
    if longitudinal_distribution_type == 'gaussian':
        
        particles = xp.generate_matched_gaussian_bunch(_context=_context,
            num_particles=num_part, 
            total_intensity_particles=beamParams.Nb,
            nemitt_x=beamParams.exn, 
            nemitt_y=beamParams.eyn, 
            sigma_z= beamParams.sigma_z,
            particle_ref=line.particle_ref, 
            line=line)

    elif longitudinal_distribution_type in ['qgaussian', 'binomial', 'parabolic']:
        
        # Select reference line: from PS extraction or not
        if matched_for_PS_extraction:
            ps = PS_sequence_maker()
            line0, _ = ps.load_xsuite_line_and_twiss(at_injection_energy=False)
        else:
            line0 = line
        
        # Generate longitudinal coordinates matched for either PS extraction or SPS injection
        zeta, delta = generate_longitudinal_coordinates(line=line0, distribution=longitudinal_distribution_type, 
                                                        num_particles=num_part, 
                                                        engine='single-rf-harmonic', sigma_z=beamParams.sigma_z,
                                                        particle_ref=line0.particle_ref, return_matcher=False, m=beamParams.m, q=beamParams.q)
        
        # Initiate normalized coordinates 
        x_norm = np.random.normal(size=num_part)
        px_norm = np.random.normal(size=num_part)
        y_norm = np.random.normal(size=num_part)
        py_norm = np.random.normal(size=num_part)
        
        # Build particles with reference particle and emittance matched to desired sequence
        particles = build_particles(_context=_context, particle_ref=line.particle_ref,
                                    zeta=zeta, delta=delta, 
                                    x_norm=x_norm, px_norm=px_norm,
                                    y_norm=y_norm, py_norm=py_norm,
                                    nemitt_x=beamParams.exn, nemitt_y=beamParams.eyn,
                                    weight=beamParams.Nb/num_part, line=line)
    else:
        raise ValueError("'longitudinal_distribution_type' has to be 'gaussian', 'qgaussian', 'binomial' or 'parabolic'!")
    
    
    return particles
    



def build_particles_linear_in_zeta(beamParams, line, num_particles_linear_in_zeta=5, 
                                   xy_norm_default=0.1, scale_factor_Qs=None, _context=None):
    """
    

    Parameters
    ----------
    beamParams : dataclass
        container of exn, eyn, Nb and sigma_z. Default 'None' will load nominal SPS beam parameters 
    line : xt.Line
        line object to which generate particle objects
    num_particles_linear_in_zeta : int
        number of equally spaced macroparticles linear in zeta
    xy_norm_default : float
        if building particles linear in zeta, what is the default normalized transverse coordinates (exact center not ideal for 
        if we want to study space charge and resonances)
    scale_factor_Qs : float
        if not None, factor by which we scale Qs (V_RF, h) and divide sigma_z and Nb for similar space charge effects
    context : xo.context
   
    Returns
    -------
    particles : xp.Particles
    """
    # Find suitable zeta range - make linear spacing between close to center of RF bucket and to separatrix
    factor = scale_factor_Qs if scale_factor_Qs is not None else 1.0
    zetas = np.linspace(0.05, 0.7 / factor, num=num_particles_linear_in_zeta)

    # Build the particle object
    particles = xp.build_particles(line = line, particle_ref = line.particle_ref,
                                x_norm=xy_norm_default, y_norm=xy_norm_default, delta=0.0, zeta=zetas,
                                nemitt_x = beamParams.exn, nemitt_y = beamParams.eyn, _context=_context)
    
    return particles



def return_separatrix_coordinates(beamParams, line, longitudinal_distribution_type, num_part=1000):
    """
    Generate zeta, delta coordinates of RF bucket separatrix

    Parameters
    ----------
    beamParams : dataclass
        container of exn, eyn, Nb and sigma_z. Default 'None' will load nominal SPS beam parameters 
    line : xt.Line
        line object to which generate separatrix coordinates
        DESCRIPTION.
    longitudinal_distribution_type : str
        'gaussian', 'qgaussian', 'binomial' or 'parabolic'
    num_part : int
        Number of dummy macroparticles. The default is 1000.

    Returns
    -------
    zeta_separatrix : np.ndarray
    delta_separatrix : np.ndarray
    """
    
    # Get separatrix coordinates from matcher
    _, _, matcher = generate_longitudinal_coordinates(line=line, distribution=longitudinal_distribution_type, 
                                                             num_particles=num_part, 
                                                             engine='single-rf-harmonic', sigma_z=beamParams.sigma_z,
                                                             particle_ref=line.particle_ref, return_matcher=True, 
                                                             m=beamParams.m, q=beamParams.q)
    

    ufp = matcher.get_unstable_fixed_point()
    xx = np.linspace(-ufp, ufp, 1000)
    yy = np.sqrt(2*matcher.A/matcher.C) * np.cos(matcher.B/2.*xx)
    
    zeta_separatrix = np.array(line.particle_ref._xobject.beta0[0]) * np.array(xx)  # zeta
    temp_particles = xp.Particles(p0c=line.particle_ref._xobject.p0c[0],
                               zeta=zeta_separatrix, ptau=yy)
    delta_separatrix = np.array(temp_particles.delta)
    
    return zeta_separatrix, delta_separatrix