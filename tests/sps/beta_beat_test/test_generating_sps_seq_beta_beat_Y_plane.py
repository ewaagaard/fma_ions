"""
Tester script to generate SPS sequence with beta-beat in Y
"""
import fma_ions
import numpy as np

# Test both planes, then loading
sps = fma_ions.SPS_sequence_maker()
line = sps.generate_xsuite_seq_with_beta_beat(beta_beat=0.1, plane='Y')
twiss = line.twiss()

line2 = sps.load_xsuite_line_and_twiss(beta_beat=0.1, plane='Y')