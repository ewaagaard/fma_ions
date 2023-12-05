"""
Generate PS Pb ion FMA plot - ideal lattice, with momentum offset
Tunes set to qx=6.15, qy=6.245, which were observed as stable during PS 2023 MDs
"""
import fma_ions

fma_ps = fma_ions.FMA(output_folder='output_Pb_off_momentum_ideal_lattice_Qx_Qy_stable_z0_0dot3', z0=0.3, n_linear=200)
fma_ps.run_custom_beam_PS(ion_type='Pb', m_ion=207.98, Q_LEIR=54., Q_PS=54., qx=6.15, qy=6.245)
