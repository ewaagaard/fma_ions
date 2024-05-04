"""
Tester script to generate SPS sequence with protons, tracking one turn
"""
import fma_ions
import xtrack as xt
import xpart as xp

sps = fma_ions.SPS_sequence_maker(ion_type='proton')
line, twiss = sps.load_xsuite_line_and_twiss()

# Test one-turn tracking of dummy particles
particles = xp.Particles(line=line)
line.track(particles, num_turns=1)

# Also test generate proton sequence with beta-beat
line2, twiss = sps.load_xsuite_line_and_twiss(beta_beat=0.1)