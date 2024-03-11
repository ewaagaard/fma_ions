"""
Small tester script to check from where horizontal losses originate
"""
import numpy as np
import fma_ions
import xpart as xp
import xobjects as xo
import xtrack as xt
import xfields as xf

context = xo.ContextCpu()
n_part = 5000
n_turns = 30

# Import SPS line
sps = fma_ions.SPS_sequence_maker()
line, twiss = sps.load_xsuite_line_and_twiss()
line2, twiss2 = sps.load_xsuite_line_and_twiss(add_aperture=True)
line3, twiss3 = sps.load_xsuite_line_and_twiss(beta_beat=0.1)
line.build_tracker(_context=context)
line2.build_tracker(_context=context)
line3.build_tracker(_context=context)

## Build particle object on both contexts
particles = xp.generate_matched_gaussian_bunch(_context=context,
        num_particles=n_part, total_intensity_particles=fma_ions.BeamParameters_SPS.Nb,
        nemitt_x=fma_ions.BeamParameters_SPS.exn, nemitt_y=fma_ions.BeamParameters_SPS.eyn, 
        sigma_z= fma_ions.BeamParameters_SPS.sigma_z,
        particle_ref=line.particle_ref, line=line)
particles2 = particles.copy()
particles3 = particles.copy()


#### CPU tracking ####
for turn in range(n_turns):
       line.track(particles)
       line2.track(particles2)
       line3.track(particles3)
       if turn % 5 == 0:
           print('Tracking turn {}'.format(turn))

# Print status of killed particles
print('1: SPS ideal lattice, no aperture or error\n')
print(particles.state[particles.state <= 0])
print('2: SPS ideal lattice, with aperture\n')
print(particles2.state[particles2.state <= 0])
print('3: SPS with beta-beat, no aperture\n')
print(particles3.state[particles3.state <= 0])