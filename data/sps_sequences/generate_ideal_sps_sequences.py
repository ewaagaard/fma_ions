"""
Generate standard sequences SPS sequence for different tunes
"""
from fma_ions import SPS_sequence_maker


# Instantiate SPS sequence makers with different tunes 
sps0 = SPS_sequence_maker(26.30, 26.25, seq_folder='qy_dot25')
sps1 = SPS_sequence_maker(26.30, 26.19, seq_folder='qy_dot19')

# Generate Xsuite sequences
line = sps0.generate_xsuite_seq(save_xsuite_seq=True)
line2 = sps1.generate_xsuite_seq(save_xsuite_seq=True)