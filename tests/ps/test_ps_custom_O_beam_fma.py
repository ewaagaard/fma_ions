"""
Generate PS FMA plot - with O ions at different tune
"""
import fma_ions

# Initialize FMA object
fma_ps = fma_ions.FMA(num_turns=120, n_theta=30, n_r=30)

# Test Twiss and tune adjustments of PS 
ps = fma_ions.PS_sequence_maker(ion_type='O', m_ion=15.99, Q_LEIR=4., Q_PS=4., qx0=6.19, qy0=6.14)
line = ps.generate_xsuite_seq()
twiss = line.twiss()
print('\nPS O beam: new tunes Qx = {:.4f}, Qy = {:.4f}\n'.format(twiss['qx'], twiss['qy']))
print('Ref particle:')
print(line.particle_ref.show())

# Run the quick test FMA analysis
fma_ps.run_custom_beam_PS(ion_type='O', m_ion=15.99, Q_LEIR=4., Q_PS=4., qx=6.19, qy=6.14, Nb=110e8)
