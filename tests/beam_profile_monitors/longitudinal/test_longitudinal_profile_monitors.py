"""
Simple test script to launch SPS tracking with longitudinal beam monitor
"""
import fma_ions
output_dir = './'

n_turns = 200
num_part = 10_000

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns, num_part=num_part, turn_print_interval=20)
tbt = sps.track_SPS(which_context='cpu', install_SC_on_line=False, add_aperture=False)
tbt.to_json(output_dir)


# Load data and plot
sps_plot = fma_ions.SPS_Plotting()
sps_plot.plot_longitudinal_monitor_data()
sps_plot.plot_WS_profile_monitor_data()