"""
Plot turn-by-turn data from SPS flat bottom tracking class
"""
import fma_ions

# Check that data is loaded correctly and plots are made
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=100)
sps.load_tbt_data_and_plot()