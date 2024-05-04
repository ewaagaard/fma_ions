"""
Tester script to load SPS sequence, changing RF voltage
"""
import fma_ions
import xtrack as xt
import xpart as xp

sps = fma_ions.SPS_sequence_maker()
line, twiss = sps.load_xsuite_line_and_twiss(voltage=2e6)

# Test one-turn tracking of dummy particles
particles = xp.Particles(line=line)
line.track(particles, num_turns=1)
