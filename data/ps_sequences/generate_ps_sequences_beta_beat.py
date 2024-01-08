"""
Generate standard sequences with beta beat in PS sequence 
"""
from fma_ions import PS_sequence_maker
import numpy as np

# Create beta-beat vectors 
beat = np.array([0.01, 0.02, 0.05, 0.1])

# Instantiate SPS sequence makers with different tunes 
ps0 = PS_sequence_maker(6.15, 6.245, seq_folder='qx_dot15')
ps1 = PS_sequence_maker(6.21, 6.245, seq_folder='qx_dot21')

for b in beat:
    line0 = ps0.generate_xsuite_seq_with_beta_beat(b, save_xsuite_seq=True)
    line1 = ps1.generate_xsuite_seq_with_beta_beat(b, save_xsuite_seq=True)

