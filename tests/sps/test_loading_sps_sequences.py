"""
Test loading pre-made SPS sequences 
"""
import numpy as np
from sequence_maker import SPS_sequence_maker

# instantiate sps sequence object and load pre-made files
sps = SPS_sequence_maker(26.30, 26.19)
line00, twiss00 = sps.load_xsuite_line_and_twiss()


line01, twiss01 = sps.load_xsuite_line_and_twiss(beta_beat=0.05)
line02, twiss02 = sps.load_xsuite_line_and_twiss(beta_beat=0.15)


# Try with different fractional tune 
line10, twiss10 = sps.load_xsuite_line_and_twiss(Qy_frac=25)
line11, twiss11 = sps.load_xsuite_line_and_twiss(Qy_frac=25, beta_beat=0.05)
line12, twiss12 = sps.load_xsuite_line_and_twiss(Qy_frac=25, beta_beat=0.15)

# Test line that doesn't yet exist
line2, twiss2 = sps.load_xsuite_line_and_twiss(Qy_frac=19, beta_beat=0.25)


# Compare beta-beat against default line
print('\nCheck fractional tune Qy = 0.19, predicted beta-beat 0.05\n')
print('Twiss max betx difference: {:.3f} vs {:.3f} with QD error'.format(np.max(twiss00['betx']),
                                                                                np.max(twiss01['betx'])))
print('Twiss max bety difference: {:.3f} vs {:.3f} with QD error'.format(np.max(twiss00['bety']),
                                                                                np.max(twiss01['bety'])))

# Show beta-beat 
print('\nX beta-beat: {:.4f}'.format( (np.max(twiss01['betx']) - np.max(twiss00['betx']))/np.max(twiss00['betx']) ))
print('\nY beta-beat: {:.4f}'.format( (np.max(twiss01['bety']) - np.max(twiss00['bety']))/np.max(twiss00['bety']) ))

print('\nCheck fractional tune Qy = 0.25, predicted beta-beat 0.05\n')
print('Twiss max betx difference: {:.3f} vs {:.3f} with QD error'.format(np.max(twiss10['betx']),
                                                                                np.max(twiss11['betx'])))
print('Twiss max bety difference: {:.3f} vs {:.3f} with QD error'.format(np.max(twiss10['bety']),
                                                                                np.max(twiss11['bety'])))

# Show beta-beat 
print('\nX beta-beat: {:.4f}'.format( (np.max(twiss11['betx']) - np.max(twiss10['betx']))/np.max(twiss10['betx']) ))
print('\nY beta-beat: {:.4f}'.format( (np.max(twiss11['bety']) - np.max(twiss10['bety']))/np.max(twiss10['bety']) ))