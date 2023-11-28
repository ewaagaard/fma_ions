"""
Generate SPS Pb ion FMA plot - ideal lattice, with momentum offset
"""
import fma_ions

fma_sps = fma_ions.FMA(output_folder='output_Pb_off_momentum_ideal_lattice', delta0=1e-3, n_r=200, n_theta=80)
fma_sps.run_SPS()
