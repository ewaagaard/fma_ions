"""
Small script to test tracking time when adding longitudinal SC kicks
"""
import numpy as np
import time
import xtrack as xt
import xpart as xp
import xobjects as xo
import xfields as xf

import fma_ions

use_gpu = True

# Beam parameters, will be used for space charge
if use_gpu:
    context = xo.ContextCupy()
else:
    context = xo.ContextCpu(omp_num_threads='auto')

num_turns = 1000
num_part = 1000
Nb = 2.46e8 # bunch_intensity measured 2.46e8 Pb ions per bunch on 2023-10-16
sigma_z =  0.225
nemitt_x = 1.3e-6
nemitt_y = 0.9e-6
num_spacecharge_interactions = 1080

# Load line, install space charge
sps = fma_ions.SPS_sequence_maker()
line, twiss = sps.load_xsuite_line_and_twiss()

# Build Gaussian beam
particles = xp.generate_matched_gaussian_bunch(_context=context,
                num_particles=num_part, 
                total_intensity_particles=Nb,
                nemitt_x=nemitt_x, 
                nemitt_y=nemitt_y, 
                sigma_z= sigma_z,
                particle_ref=line.particle_ref, 
                line=line)
particles2 = particles.copy()

# Install frozen space charge, emulating a Gaussian bunch
lprofile = xf.LongitudinalProfileQGaussian(
        number_of_particles = Nb,
        sigma_z = sigma_z,
        z0=0.,
        q_parameter=1.0)

# Install frozen space charge as base 
xf.install_spacecharge_frozen(line = line,
                   particle_ref = line.particle_ref,
                   longitudinal_profile = lprofile,
                   nemitt_x = nemitt_x, nemitt_y = nemitt_y,
                   sigma_z = sigma_z,
                   num_spacecharge_interactions = num_spacecharge_interactions)

line.build_tracker(_context = context)

# Track particles with no Z kick
time00 = time.time()
for turn in range(1, num_turns):
    if turn % 20 == 0:
        print(f'Round 1: tracking turn {turn}')
    line.track(particles)
time01 = time.time()
dt0 = time01-time00

# Make a copy of line, with longitudinal Z kicks
tt = line.get_table()
tt_sc = tt.rows[tt.element_type=='SpaceChargeBiGaussian']
for nn in tt_sc.name:
    line[nn].z_kick_num_integ_per_sigma = 5

# Track particles with Z kick
time10 = time.time()
for turn in range(1, num_turns):
    if turn % 10 == 0:
        print(f'Round 2: tracking turn {turn}')
    line.track(particles2)
time11 = time.time()
dt1 = time11-time10

print('\nTracking time no Z kick: {:.1f} s = {:.1f} min'.format(dt0, dt0/60))
print('Tracking time with Z kick: {:.1f} s = {:.1f} min'.format(dt1, dt1/60))
print('Using gpu: {}'.format(use_gpu))