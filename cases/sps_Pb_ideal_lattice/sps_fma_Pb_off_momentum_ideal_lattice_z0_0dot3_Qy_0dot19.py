"""
Generate SPS Pb ion FMA plot - ideal lattice, with momentum offset
"""
import fma_ions

fma_sps = fma_ions.FMA(output_folder='output_Pb_off_momentum_ideal_lattice_z0_0dot3_Qy_0dot19', z0=0.3, n_linear=200)
fma_sps.run_custom_beam_SPS(ion_type='Pb', m_ion=207.98, Q_SPS=82., Q_PS=54., qx=26.30, qy=26.19, Nb=2.2e8)
