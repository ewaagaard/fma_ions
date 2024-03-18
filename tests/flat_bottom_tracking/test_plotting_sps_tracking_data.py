"""
Plot turn-by-turn data from SPS flat bottom tracking class
"""
import fma_ions

tracking_has_been_done = True

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=100, turn_print_interval=10, num_part=100)

if not tracking_has_been_done:
    tbt = sps.track_SPS(which_context='cpu')
    tbt.to_parquet('tbt.parquet')

# First try in turns, with / without measurements
sps.load_tbt_data_and_plot(x_unit_in_turns=True, show_plot=True)
sps.load_tbt_data_and_plot(include_emittance_measurements=True, x_unit_in_turns=True, show_plot=True)

# Then use time units, with / without measurements
sps.load_tbt_data_and_plot(x_unit_in_turns=False, show_plot=True)
sps.load_tbt_data_and_plot(include_emittance_measurements=True, x_unit_in_turns=False, show_plot=True)