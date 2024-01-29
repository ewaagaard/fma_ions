"""
Tester script to generate SPS sequence - replacing all QFA magnets with QF to make lattice fully symmetric
"""
import fma_ions

# First generate sequence for Qy = 26.25
sps = fma_ions.SPS_sequence_maker(qy0=26.25)
sps.generate_symmetric_SPS_lattice()

# Then generate sequence for Qy = 26.19
sps2 = fma_ions.SPS_sequence_maker(qy0=26.19)
sps2.generate_symmetric_SPS_lattice()