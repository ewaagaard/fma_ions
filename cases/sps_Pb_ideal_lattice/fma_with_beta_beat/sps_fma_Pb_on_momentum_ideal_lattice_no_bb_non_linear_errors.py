"""
Generate SPS Pb ion FMA plot - ideal lattice, no momentum offset
- no beta-beat and non-linear chromatic errors
"""
import fma_ions

fma_sps = fma_ions.FMA(output_folder='output_Pb_on_momentum_ideal_lattice_no_bb_non_linear_chroma_errors', z0=0., n_linear=200)
fma_sps.run_SPS(load_tbt_data=False, add_non_linear_magnet_errors=True)
