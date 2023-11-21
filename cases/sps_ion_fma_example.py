"""
Frequency Map Analysis (FMA) example with SPS Pb ions 
- with space charge installed, investigate tune diffusion
"""
import time
import numpy as np
import xobjects as xo
import xpart as xp
import xtrack as xt
import xfields as xf

# Import the Xsuite sequence
fname_line = ('sequences/sps/SPS_2021_Pb_ions_matched_with_RF.json')
# fname_line = ('sequences/sps/sps_injection_ions.json')

# Set beam parameters and SC interactions
bunch_intensity = 3.5e8 #
sigma_z = 0.15 #22.5e-2  # in m, with 4 ns bunch length (4 rms) from table 38.3 in LHC design report (2004)
nemitt_x = 1.3e-6
nemitt_y = 0.9e-6
n_part = 5000
num_turns = 2#20

num_spacecharge_interactions = 540
tol_spacecharge_position = 1e-2
mode = 'frozen'

# Select context and load the Xsuite line
context = xo.ContextCpu()
line = xt.Line.from_json(fname_line)
particle_ref = line.particle_ref

# Install spacecharge interactions (frozen) #
lprofile = xf.LongitudinalProfileQGaussian(
        number_of_particles=bunch_intensity,
        sigma_z=sigma_z,
        z0=0.,
        q_parameter=1.)


xf.install_spacecharge_frozen(line=line,
                   particle_ref=particle_ref,
                   longitudinal_profile=lprofile,
                   nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                   sigma_z=sigma_z,
                   num_spacecharge_interactions=num_spacecharge_interactions,
                   tol_spacecharge_position=tol_spacecharge_position)

                   
# Build tracker
line.build_tracker(_context=context)
line.optimize_for_tracking()
tw = line.twiss()
line_sc_off = line.filter_elements(exclude_types_starting_with='SpaceCh')

# Generate matched Gaussian beam
#particles = xp.generate_matched_gaussian_bunch(_context=context,
#         num_particles=n_part, total_intensity_particles=bunch_intensity,
#         nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
#         particle_ref=particle_ref, line=line_sc_off)

x_norm, y_norm, _, _ = xp.generate_2D_polar_grid(
    theta_range=(0.01, np.pi/2-0.01),
    ntheta = 20,
    r_range = (0.1, 7),
    nr = 30)

particles = xp.build_particles(line=line, particle_ref=line.particle_ref,
                               x_norm=x_norm, y_norm=y_norm, delta=0,
                               nemitt_x=nemitt_x, nemitt_y=nemitt_y)

# Track the particles
i = 0
for turn in range(num_turns):
    time0 = time.time()
    print('Tracking turn {}'.format(i))
    i += 1

    # Track the particles
    line.track(particles)