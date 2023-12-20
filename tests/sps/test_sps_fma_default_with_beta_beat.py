"""
Generate SPS Pb ion FMA plot
"""
import fma_ions

fma_sps = fma_ions.FMA(num_turns=120, n_linear=25, output_folder='Output_test')
fma_sps.run_SPS_with_beta_beat(load_tbt_data=False, beta_beat=0.05)
