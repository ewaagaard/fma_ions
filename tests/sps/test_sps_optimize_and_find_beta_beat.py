"""
Tester script to investigate beta beat in SPS sequence 
- introduce one QD error and observe effect 
"""
from sequence_maker import SPS_sequence_maker
import numpy as np

sps = SPS_sequence_maker(26.30, 26.19)
result = sps.generate_xsuite_seq_with_beta_beat()
print(result)