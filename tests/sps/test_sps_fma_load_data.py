"""
Generate SPS Pb ion FMA plot with loaded data
"""
import fma_ions

fma_sps = fma_ions.FMA(num_turns=60, n_linear=250, output_folder='Output_test_many_particles')
fma_sps.run_SPS(load_tbt_data=True)
