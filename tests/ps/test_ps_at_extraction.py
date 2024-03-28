"""
Test script to check whether correct PS extraction sequence is used
"""
from fma_ions import PS_sequence_maker

# Instantiate sequence maker objects - Pb default beam, Pb custom beam and O beam
s0 = PS_sequence_maker()
line = s0.generate_xsuite_seq(save_xsuite_seq=True, save_madx_seq=True, at_injection_energy=False)
line2 = s0.load_xsuite_line_and_twiss(at_injection_energy=False) # try to load the same line

# Check cases 
print('\nGenerated {} beam with gamma = {:.5f}\n'.format(s0.ion_type, s0.particle_sample.gamma0[0]))
nn = 'pa.c10.11' 
print(f'Lag: {line[nn].lag}')
print(f'Voltage: {line[nn].voltage} V')
print(f'Frequency: {line[nn].frequency} Hz') 

line.twiss()