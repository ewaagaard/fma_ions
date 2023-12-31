"""
Generate SPS Pb ion FMA plot
"""
import fma_ions

fma_sps = fma_ions.FMA(num_turns=50, n_linear=20, z0=0.2, output_folder='Output_test_off_momentum')
fma_sps.run_SPS(load_tbt_data=False)
