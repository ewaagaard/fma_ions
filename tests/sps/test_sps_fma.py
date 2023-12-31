"""
Generate SPS Pb ion FMA plot
"""
import fma_ions

fma_sps = fma_ions.FMA(num_turns=60, n_linear=25, output_folder='Output_test')
fma_sps.run_SPS(load_tbt_data=False)
