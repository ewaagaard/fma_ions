"""
Tester script to generate SPS sequence with non-linear magnet errors 
"""
import fma_ions
import numpy as np


sps = fma_ions.SPS_sequence_maker()
#line = sps.generate_xsuite_seq(add_non_linear_magnet_errors=True)
#twiss = line.twiss()
#print('\nGenerated sequence with qx = {}, qy = {} - with errors!\n'.format(twiss['qx'], twiss['qy']))


# Then test adding beta-beat to SPS line 
line1 = sps.generate_xsuite_seq_with_beta_beat(add_non_linear_magnet_errors=False)
twiss1 = line1.twiss()
print('\nGenerated sequence with qx = {}, qy = {} - with beta-beat!\n'.format(twiss1['qx'], twiss1['qy']))

'''
# Then test adding beta-beat to SPS line with non-linear chromatic errors
line2 = sps.generate_xsuite_seq_with_beta_beat(add_non_linear_magnet_errors=True)
twiss2 = line2.twiss()
print('\nGenerated sequence with qx = {}, qy = {} - with magnet errors and beta-beat!\n'.format(twiss2['qx'], twiss2['qy']))
'''
