import numpy as np
import xtrack as xt
import xpart as xp
import xobjects as xo
import xfields as xf

####################
# Choose a context #
####################

context = xo.ContextCpu()

print(context)

# Load the PS line, define beam parameters 
fname_line = ('../../sequences/ps/PS_2022_Pb_ions_matched_with_RF.json')
line = xt.Line.from_json(fname_line)

nemitt_x=0.8e-6
nemitt_y=0.5e-6
bunch_intensity=8.1e10
sigma_z=4.74

# from space charge example
num_turns=12 #00 
num_spacecharge_interactions = 160 
tol_spacecharge_position = 1e-2 

# Available modes: frozen/quasi-frozen/pic
mode = 'frozen'

#############################################
# Install spacecharge interactions (frozen) #
#############################################

lprofile = xf.LongitudinalProfileQGaussian(
        number_of_particles=bunch_intensity,
        sigma_z=sigma_z,
        z0=0.,
        q_parameter=1.)

xf.install_spacecharge_frozen(line=line,
                   particle_ref=line.particle_ref,
                   longitudinal_profile=lprofile,
                   nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                   sigma_z=sigma_z,
                   num_spacecharge_interactions=num_spacecharge_interactions,
                   tol_spacecharge_position=tol_spacecharge_position)


#################
# Build Tracker #
#################

line.build_tracker(_context=context)
line_sc_off = line.filter_elements(exclude_types_starting_with='SpaceCh')

######################
# Generate particles #
######################

x_norm, y_norm, _, _ = xp.generate_2D_polar_grid(
    theta_range=(0.01, np.pi/2-0.01),
    ntheta = 20,
    r_range = (0.1, 7),
    nr = 30)


#particles = xp.build_particles(line=line, particle_ref=line.particle_ref,
#                               x_norm=x_norm, y_norm=y_norm, delta=0,
#                               nemitt_x=nemitt_x, nemitt_y=nemitt_y)


# Generate matched Gaussian beam
n_part = 5000
particles = xp.generate_matched_gaussian_bunch(_context=context,
         num_particles=n_part, total_intensity_particles=bunch_intensity,
         nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
         particle_ref=line.particle_ref, line=line_sc_off)

# Track the particles 
#line.track(particles, num_turns=num_turns, turn_by_turn_monitor=True )
i = 0
for turn in range(num_turns):
    print('Tracking turn {}'.format(i))
    i += 1

    # Track the particles
    line.track(particles)