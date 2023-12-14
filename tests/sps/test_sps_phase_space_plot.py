"""
Generate SPS phase space plot from generated turn-by-turn data
"""
import fma_ions

fma_sps = fma_ions.FMA(output_folder='Output_test')

# Load Twiss and plot normalized phase space
sps_seq = fma_ions.SPS_sequence_maker()
sps_line, twiss_sps = sps_seq.load_xsuite_line_and_twiss()

fma_sps.plot_normalized_phase_space(twiss_sps)
