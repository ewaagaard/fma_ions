"""
Generate PS Pb ion FMA plot - with beta-beat and space charge
"""
import fma_ions

fma_ps = fma_ions.FMA(num_turns=120, n_linear=25, output_folder='Output_sc_test_beta_beat')
fma_ps.run_PS_with_beta_beat(load_tbt_data=False, beta_beat=0.02)
