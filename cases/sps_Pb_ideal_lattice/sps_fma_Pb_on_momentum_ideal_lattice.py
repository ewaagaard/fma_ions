"""
Generate SPS Pb ion FMA plot - ideal lattice, no momentum offset
"""
import fma_ions

fma_sps = fma_ions.FMA(output_folder='output_Pb_on_momentum_ideal_lattice', delta0=0., n_r=200, n_theta=80)
fma_sps.run_SPS()
