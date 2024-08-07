"""
Tester script to generate SPS sequence with different tunes
"""
import fma_ions

sps = fma_ions.SPS_sequence_maker(26.30, 26.19)
line = sps.generate_xsuite_seq(save_xsuite_seq=True)
twiss = line.twiss()
print('\nGenerated sequence with qx = {}, qy = {}\n'.format(twiss['qx'], twiss['qy']))