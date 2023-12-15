"""
Generate standard SPS sequence for MADX
"""
from fma_ions import SPS_sequence_maker


# Instantiate SPS sequence makers with different tunes 
sps0 = SPS_sequence_maker(26.30, 26.25, seq_folder='madx')

# Generate Xsuite sequences
line = sps0.generate_xsuite_seq(save_madx_seq=True)