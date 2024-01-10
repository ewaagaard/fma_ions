"""
Generate SPS Pb ion FMA plot - ideal lattice, with momentum offset
"""
import fma_ions

fma_sps = fma_ions.FMA(output_folder='output_Pb_on_momentum_ideal_lattice_Qy_0dot19', z0=0.0, n_linear=200)
fma_sps.run_SPS(Qy_frac=19, load_tbt_data=True)
