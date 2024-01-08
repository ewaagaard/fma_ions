"""
Generate ideal sequences for PS sequence 
"""
from fma_ions import PS_sequence_maker

# Instantiate SPS sequence makers with different tunes 
ps0 = PS_sequence_maker(6.15, 6.245, seq_folder='qx_dot15')
ps1 = PS_sequence_maker(6.21, 6.245, seq_folder='qx_dot21')


line0 = ps0.generate_xsuite_seq(save_xsuite_seq=True)
line1 = ps1.generate_xsuite_seq(save_xsuite_seq=True)

