"""
Test loading pre-made SPS sequence, then change synchrotron tune
"""
import numpy as np
from fma_ions import SPS_sequence_maker

# instantiate sps sequence object and load pre-made files
sps = SPS_sequence_maker(26.30, 26.19)
line, twiss = sps.load_xsuite_line_and_twiss()

# Update synchrotron tune
factor = 2
line, sigma_z, Nb = sps.change_synchrotron_tune_by_factor(factor, line)
twiss2 = line.twiss()

print('Scaling with factor {}: new 1/Qs = {:.6f}, old 1/Qs = {:.6f}'.format(factor, 1/twiss2.qs, 1/twiss.qs))