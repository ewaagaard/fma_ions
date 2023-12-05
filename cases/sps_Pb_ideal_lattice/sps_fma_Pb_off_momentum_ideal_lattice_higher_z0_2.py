"""
Generate SPS Pb ion FMA plot - ideal lattice, with momentum offset
"""
import fma_ions

fma_sps = fma_ions.FMA(output_folder='output_Pb_off_momentum_ideal_lattice_z0_0dot15', z0=0.15, n_linear=200)
fma_sps.run_SPS()
