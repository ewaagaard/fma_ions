"""
Generate standard symmetric SPS sequence for different tunes
- QFAs and QDAs replaced by normal QFs and QDs
"""
from fma_ions import SPS_sequence_maker


# Instantiate SPS sequence makers with different tunes 
sps0 = SPS_sequence_maker(26.30, 26.25, seq_folder='qy_dot25')
sps1 = SPS_sequence_maker(26.30, 26.19, seq_folder='qy_dot19')

# Generate Xsuite sequences
line = sps0.generate_symmetric_SPS_lattice(save_madx_seq=True, save_xsuite_seq=True)
line2 = sps1.generate_symmetric_SPS_lattice(save_madx_seq=True, save_xsuite_seq=True)