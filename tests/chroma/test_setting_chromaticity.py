"""
Test script to ensure that SPS Q26 proton chroma measurements from Ingrid are properly included 
- QPH = 0.234 - 0.06 = 0.174
- QPV = 0.216 - 0.13 = 0.086
"""
import fma_ions

# Generate sequence, check twiss values
sps = fma_ions.SPS_sequence_maker(qx0=26.30, qy0=26.19, ion_type='proton')
line = sps.generate_xsuite_seq() 
tw = line.twiss()
print('Qx = {:.3f}, Qy = {:.3f}'.format(tw['qx'], tw['qy']))
print('dqx = {:.3f}, dqy = {:.3f}'.format(tw['dqx'], tw['dqy']))

