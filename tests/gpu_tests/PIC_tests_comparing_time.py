"""
Track 1000 turns with 5000 particles - how fast is GPU vs CPU?
"""
import numpy as np
import fma_ions
import xpart as xp
import xobjects as xo
import xtrack as xt
import xfields as xf
import time

pic_solver = 'FFTSolver2p5DAveraged'

# Initial parameters
n_part = 5_000
n_part2 = 5_000
n_turns = 150

# Switch context if needed
context = xo.ContextCupy()
context2 = xo.ContextCupy()

# Import SPS line
sps = fma_ions.SPS_sequence_maker()
line, twiss = sps.load_xsuite_line_and_twiss()
line.discard_tracker()
line2 = line.copy()

line.build_tracker(_context=context2, compile=False)
line2.build_tracker(_context=context2, compile=False)

#############################################
# Install spacecharge interactions (frozen) #
#############################################

lprofile = xf.LongitudinalProfileQGaussian(
        number_of_particles=fma_ions.BeamParameters_SPS.Nb,
        sigma_z=fma_ions.BeamParameters_SPS.sigma_z,
        z0=0.,
        q_parameter=1.)

xf.install_spacecharge_frozen(line=line,
                   longitudinal_profile=lprofile,
                   nemitt_x=fma_ions.BeamParameters_SPS.exn, nemitt_y=fma_ions.BeamParameters_SPS.eyn,
                   sigma_z=fma_ions.BeamParameters_SPS.sigma_z,
                   num_spacecharge_interactions=1080,
                   )

xf.install_spacecharge_frozen(line=line2,
                   longitudinal_profile=lprofile,
                   nemitt_x=fma_ions.BeamParameters_SPS.exn, nemitt_y=fma_ions.BeamParameters_SPS.eyn,
                   sigma_z=fma_ions.BeamParameters_SPS.sigma_z,
                   num_spacecharge_interactions=1080,
                   )

# Replace frozen SC with PIC for line2
pic_collection, all_pics = xf.replace_spacecharge_with_PIC(
    line=line2,
    n_sigmas_range_pic_x=8,
    n_sigmas_range_pic_y=8,
    nx_grid=256, ny_grid=256, nz_grid=100,
    n_lims_x=7, n_lims_y=3,
    z_range=(-3*fma_ions.BeamParameters_SPS.sigma_z, 3*fma_ions.BeamParameters_SPS.sigma_z),
    solver=pic_solver)

# Build trackers
line.build_tracker(_context=context)
line2.build_tracker(_context=context2)


particles = xp.generate_matched_gaussian_bunch(_context=context,
        num_particles=n_part, total_intensity_particles=fma_ions.BeamParameters_SPS.Nb,
        nemitt_x=fma_ions.BeamParameters_SPS.exn, nemitt_y=fma_ions.BeamParameters_SPS.eyn, 
        sigma_z= fma_ions.BeamParameters_SPS.sigma_z,
        particle_ref=line2.particle_ref, line=line2)

particles2 = xp.generate_matched_gaussian_bunch(_context=context2,
        num_particles=n_part2, total_intensity_particles=fma_ions.BeamParameters_SPS.Nb,
        nemitt_x=fma_ions.BeamParameters_SPS.exn, nemitt_y=fma_ions.BeamParameters_SPS.eyn, 
        sigma_z= fma_ions.BeamParameters_SPS.sigma_z,
        particle_ref=line2.particle_ref, line=line2)

# Particles are allocated on the context chosen for the line.
# Also test loading numpy arrays
x = np.zeros([n_part, n_turns])
y = np.zeros([n_part, n_turns])
x2 = np.zeros([n_part, n_turns])
y2 = np.zeros([n_part, n_turns])
x3 = np.zeros([n_part2, n_turns])
y3 = np.zeros([n_part2, n_turns])

#### GPU tracking - remember to get it from CPU context ####
time10 = time.time()
for turn in range(n_turns):
   line.track(particles)
   x2[:, turn] = particles.x.get()
   y2[:, turn] = particles.y.get()
   if turn % 5 == 0:
       print('Tracking turn {}'.format(turn))
time11 = time.time()
dt1 = time11-time10

#### GPU tracking with 15 000 particles - remember to get it from CPU context ####
time20 = time.time()
for turn in range(n_turns):
   line2.track(particles2)
   x3[:, turn] = particles2.x.get()
   y3[:, turn] = particles2.y.get()
   if turn % 5 == 0:
       print('Tracking turn {}'.format(turn))
time21 = time.time()
dt2 = time21-time20

print('\nGPU tracking {} particles with frozen: {} s\nGPU PIC tracking with {} particles: {} s'.format(n_part, dt1, n_part2, dt2))
