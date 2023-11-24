"""
Test script to check whether correct SPS injection energy is handled
"""
from fma_ions import SPS_sequence_maker

# Instantiate sequence maker objects - Pb default beam, Pb custom beam and O beam
s0 = SPS_sequence_maker()
s1 = SPS_sequence_maker(m_ion=208., Q_PS=54., Q_SPS=82.)
s2 = SPS_sequence_maker(ion_type='O', m_ion=15.99, Q_PS=4., Q_SPS=8.)

s0.generate_xsuite_seq(save_xsuite_seq=False)
s1.generate_xsuite_seq(save_xsuite_seq=False)
s2.generate_xsuite_seq(save_xsuite_seq=False)

# Check cases 
print('\nCase 1: Generated {} beam with gamma = {:.3f}\n'.format(s0.ion_type, s0.particle_sample.gamma0[0]))
print('\nCase 2: Generated {} beam with gamma = {:.3f}\n'.format(s1.ion_type, s1.particle_sample.gamma0[0]))
print('\nCase 3: Generated {} beam with gamma = {:.3f}\n'.format(s2.ion_type, s2.particle_sample.gamma0[0]))