"""
Tester script to generate SPS sequence with oxygen beams
"""
import fma_ions

sps = fma_ions.SPS_sequence_maker(26.30, 26.19, ion_type='O', Q_PS=4., Q_SPS=8., m_ion=15.9949) 
line = sps.generate_xsuite_seq()
twiss = line.twiss()
print('\nGenerated sequence with qx = {}, qy = {}\n'.format(twiss['qx'], twiss['qy']))

# Then trying to load the generated sequence