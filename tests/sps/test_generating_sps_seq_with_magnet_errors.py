"""
Tester script to generate SPS sequence with non-linear magnet errors 
"""
import fma_ions

sps = fma_ions.SPS_sequence_maker()
line = sps.generate_xsuite_seq(add_non_linear_magnet_errors=True)
twiss = line.twiss()
print('\nGenerated sequence with qx = {}, qy = {} - with errors!\n'.format(twiss['qx'], twiss['qy']))