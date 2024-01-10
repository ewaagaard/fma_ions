"""
Generate SPS Pb ion FMA plot - symmetric lattice without QFA and QDA, no momentum offset
"""
import fma_ions

fma_sps = fma_ions.FMA(output_folder='output_Pb_on_momentum_symmetric_lattice', z0=0., n_linear=200)
fma_sps.run_SPS(load_tbt_data=False, use_symmetric_lattice=True)
