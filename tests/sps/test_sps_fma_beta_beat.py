"""
Generate SPS Pb ion FMA plot - with beta-beat and space charge
"""
import fma_ions

fma_sps = fma_ions.FMA(num_turns=120, n_linear=25, output_folder='Output_sc_test_beta_beat')
fma_sps.run_SPS_with_beta_beat(load_tbt_data=False, Qy_frac=19, beta_beat=0.05)
