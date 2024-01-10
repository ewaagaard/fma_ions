"""
Tester script to generate SPS sequence - replacing all QFA magnets with QF to make lattice fully symmetric
"""
import fma_ions

sps = fma_ions.SPS_sequence_maker()
sps.generate_symmetric_SPS_lattice()
