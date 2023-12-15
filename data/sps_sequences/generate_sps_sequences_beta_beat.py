"""
Generate standard sequences with beta beat in SPS sequence 
"""
from fma_ions import SPS_sequence_maker
import numpy as np

# Create beta-beat vectors 
beat = np.array([0.02, 0.05, 0.10, 0.15])

# Instantiate SPS sequence makers with different tunes 
sps0 = SPS_sequence_maker(26.30, 26.25, seq_folder='qy_dot25')
sps1 = SPS_sequence_maker(26.30, 26.19, seq_folder='qy_dot19')

for b in beat:
    line0 = sps0.generate_xsuite_seq_with_beta_beat(b, save_xsuite_seq=True)
    line1 = sps1.generate_xsuite_seq_with_beta_beat(b, save_xsuite_seq=True)

