"""
Small example to install SPS beam profile monitor
Following example in https://xsuite.readthedocs.io/en/latest/track.html#beam-profile-monitor
"""
import time
import matplotlib.pyplot as plt

import fma_ions
import xtrack as xt
import xobjects as xo
import xpart as xp

# Turns for tracking and collecting particle data
num_turns = 300
nbins = 200
num_part = 10_000
nturns_profile_accumulation = 100
context = xo.ContextCpu(omp_num_threads='auto')

# Beam parameters, will be used for space charge
Nb = 2.46e8 # bunch_intensity measured 2.46e8 Pb ions per bunch on 2023-10-16
sigma_z =  0.225
exn = 1.3e-6
eyn = 0.9e-6

# instantiate sps sequence object and load pre-made files
sps = fma_ions.SPS_sequence_maker(26.30, 26.19)
line, twiss = sps.load_xsuite_line_and_twiss()
line.discard_tracker()

# Create horizontal beam monitor
monitorH = xt.BeamProfileMonitor(
    start_at_turn=nturns_profile_accumulation/2, stop_at_turn=num_turns,
    frev=1,
    sampling_frequency=1/nturns_profile_accumulation,
    n=nbins,
    x_range=0.05,
    y_range=0.05)
line.insert_element(index='bwsrc.51637', element=monitorH, name='monitorH')

# Create vertical beam monitor
monitorV = xt.BeamProfileMonitor(
    start_at_turn=nturns_profile_accumulation/2, stop_at_turn=num_turns,
    frev=1,
    sampling_frequency=1/nturns_profile_accumulation,
    n=nbins,
    x_range=0.05,
    y_range=0.05)
line.insert_element(index='bwsrc.41677', element=monitorV, name='monitorV')

line.build_tracker(_context = context)

# Generate Gaussian particle bunch
particles = xp.generate_matched_gaussian_bunch(_context=context,
    num_particles=num_part, 
    total_intensity_particles=Nb,
    nemitt_x=exn, 
    nemitt_y=eyn, 
    sigma_z=sigma_z,
    particle_ref=line.particle_ref, 
    line=line)

# Track particles
time00 = time.time()
for turn in range(1, num_turns):
    if turn % 10 == 0:
        print(f'Tracking turn {turn}')
    line.track(particles)
time01 = time.time()
dt0 = time01-time00
print('\nTracking time: {:.1f} s = {:.1f} min'.format(dt0, dt0/60))

# Plot profile of particles
fig, ax = plt.subplots(1, 1, figsize = (8, 6))
for i in range(len(monitorH.x_intensity)):
    ax.plot(monitorH.x_grid, monitorH.x_intensity[i], label='Turn aggreation {}'.format(nturns_profile_accumulation*(i+1)))
ax.set_xlabel('x [m]')
ax.set_ylabel('Counts')
ax.legend()
plt.tight_layout()

# Plot profile of particles
fig2, ax2 = plt.subplots(1, 1, figsize = (8, 6))
for i in range(len(monitorV.y_intensity)):
    ax2.plot(monitorV.y_grid, monitorV.y_intensity[i], label='Turn aggreation {}'.format(nturns_profile_accumulation*(i+1)))
ax2.set_ylabel('Counts')
ax2.set_xlabel('y [m]')
ax2.legend()
plt.tight_layout()

plt.show()