"""
First test of xtrack SC simulations on GPU context with new Legion - does it work?
"""
import numpy as np
import fma_ions
import xpart as xp
import xobjects as xo
import xtrack as xt
import xfields as xf


# Initial parameters
n_part = 100
n_turns = 100
gpu_device = 0

# Switch context if needed
if gpu_device is None:
   context = xo.ContextCpu()
   print('Attempting CPUs')
else:
   context = xo.ContextCupy(device=gpu_device)
   print('Attempting GPUs')

# Import SPS line
sps = fma_ions.SPS_sequence_maker()
line, twiss = sps.load_xsuite_line_and_twiss()

# Choose context - discard tracker and move context to GPU
context = xo.ContextCupy(device=gpu_device)
line.discard_tracker()
line.build_tracker(_context=context)

## Build particle object on context
particles = xp.generate_matched_gaussian_bunch(_context=context,
        num_particles=n_part, total_intensity_particles=fma_ions.BeamParameters_SPS.Nb,
        nemitt_x=fma_ions.BeamParameters_SPS.exn, nemitt_y=fma_ions.BeamParameters_SPS.eyn, 
        sigma_z= fma_ions.BeamParameters_SPS.sigma_z,
        particle_ref=line.particle_ref, line=line)
# Reference mass, charge, energy are taken from the reference particle.
# Particles are allocated on the context chosen for the line.

for turn in range(n_turns):
   line.track(particles)
   print('Tracking turn {}'.format(turn))


