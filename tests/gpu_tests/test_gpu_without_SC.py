"""
Test whether tracking particles on GPU without installing space charge works
"""
import numpy as np
import fma_ions
import xpart as xp
import xobjects as xo
import xtrack as xt
import xfields as xf
import time

# Initial parameters
n_part = 10000
n_turns = 100

# Switch context if needed
context = xo.ContextCupy()

# Import SPS line
sps = fma_ions.SPS_sequence_maker()
line, twiss = sps.load_xsuite_line_and_twiss()
line.discard_tracker()
line.build_tracker(_context=context)

## Build particle object on both contexts
particles = xp.generate_matched_gaussian_bunch(_context=context,
        num_particles=n_part, total_intensity_particles=fma_ions.BeamParameters_SPS.Nb,
        nemitt_x=fma_ions.BeamParameters_SPS.exn, nemitt_y=fma_ions.BeamParameters_SPS.eyn, 
        sigma_z= fma_ions.BeamParameters_SPS.sigma_z,
        particle_ref=line.particle_ref, line=line)

#### GPU tracking ####
time00 = time.time()
for turn in range(n_turns):
   line.track(particles)
   if turn % 5 == 0:
       print('Tracking turn {}'.format(turn))
time01 = time.time()
dt0 = time01-time00
print(f'Finished in {dt0} seconds')