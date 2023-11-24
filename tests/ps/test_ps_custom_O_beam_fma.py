"""
Generate PS FMA plot - with O ions at different tune
"""
import fma_ions

fma_ps = fma_ions.FMA(n_theta=30, n_r=50)
fma_ps.run_custom_beam_PS(ion_type='O', m_ion=15.99, Q_LEIR=4., Q_PS=4., qx=6.19, qy=6.14, Nb=110e8)