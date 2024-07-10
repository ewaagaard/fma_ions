"""
Test script to test speed of frozen space charge with Z kicks
"""

import fma_ions
output_dir = './'

n_turns = 1_000
num_part = 5_000

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns, num_part=num_part, turn_print_interval=20)
tbt = sps.track_SPS(which_context='cpu', install_SC_on_line=True, beta_beat=0.1, 
                    add_non_linear_magnet_errors=True)
tbt.to_json(output_dir)

# Then plot
sps_plot = fma_ions.SPS_Plotting()
sps_plot.plot_tracking_data()
sps_plot.plot_longitudinal_monitor_data()