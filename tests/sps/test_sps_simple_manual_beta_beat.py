"""
Tester script to investigate beta beat in SPS sequence 
- introduce one QD error and observe effect 
"""
from sequence_maker import SPS_sequence_maker
import numpy as np

sps = SPS_sequence_maker(26.30, 26.19)
line = sps.generate_xsuite_seq()
twiss = line.twiss()
print('\nGenerated sequence with qx = {}, qy = {}\n'.format(twiss['qx'], twiss['qy']))

# Make a copy, add a QD error and observe effect on Twiss 
dQD = 1.2  # relative change in QD strength for one quadrupole
line2 = line.copy()
line2['qd.63510..1'].knl[1] = dQD * line['qd.63510..1'].knl[1]
twiss2 = line2.twiss()

# Compare difference in Twiss 
print('Twiss max betx difference: {:.3f} vs {:.3f} with QD error'.format(np.max(twiss['betx']),
                                                                                np.max(twiss2['betx'])))
print('Twiss max bety difference: {:.3f} vs {:.3f} with QD error'.format(np.max(twiss['bety']),
                                                                                np.max(twiss2['bety'])))

# Show beta-beat 
print('\nX beta-beat: {:.4f}'.format( (np.max(twiss2['betx']) - np.max(twiss['betx']))/np.max(twiss['betx']) ))
print('\nY beta-beat: {:.4f}'.format( (np.max(twiss2['bety']) - np.max(twiss['bety']))/np.max(twiss['bety']) ))