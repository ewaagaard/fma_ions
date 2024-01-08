"""
Generate PS Pb ion FMA plot
"""
import fma_ions

fma_ps = fma_ions.FMA(num_turns=800, n_linear=25, output_folder='Output_test')
fma_ps.run_PS()
