"""
Tester script to generate SPS sequence with beta-beat in both planes
"""
import fma_ions
import numpy as np


sps = fma_ions.SPS_sequence_maker()
line = sps.generate_xsuite_seq_with_beta_beat(beta_beat=0.1, plane='both')
twiss = line.twiss()

