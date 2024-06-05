"""
Simple test script to launch SPS tracking with longitudinal beam monitor - on GPU context
"""
import fma_ions
output_dir = './'

n_turns = 200
num_part = 10_000

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns, num_part=num_part, turn_print_interval=20)
zeta_monitor = sps.track_SPS(which_context='gpu', install_SC_on_line=False, add_aperture=False)

sps_plot = fma_ions.SPS_Plotting()
sps_plot.plot_longitudinal_monitor_data(zeta_monitor)