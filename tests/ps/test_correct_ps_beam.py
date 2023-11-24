"""
Test script to check whether correct PS injection energy is handled
"""
from fma_ions import PS_sequence_maker

# Instantiate sequence maker objects - Pb default beam, Pb custom beam and O beam
s0 = PS_sequence_maker()
s1 = PS_sequence_maker(m_ion=208., Q_LEIR=54., Q_PS=54.)
s2 = PS_sequence_maker(ion_type='O', m_ion=15.99, Q_LEIR=4., Q_PS=4.)

s0.generate_xsuite_seq(save_xsuite_seq=False)
s1.generate_xsuite_seq(save_xsuite_seq=False)
s2.generate_xsuite_seq(save_xsuite_seq=False)

# Check cases 
print('\nCase 1: Generated {} beam with gamma = {:.5f}\n'.format(s0.ion_type, s0.particle_sample.gamma0[0]))
print('\nCase 2: Generated {} beam with gamma = {:.5f}\n'.format(s1.ion_type, s1.particle_sample.gamma0[0]))
print('\nCase 3: Generated {} beam with gamma = {:.5f}\n'.format(s2.ion_type, s2.particle_sample.gamma0[0]))