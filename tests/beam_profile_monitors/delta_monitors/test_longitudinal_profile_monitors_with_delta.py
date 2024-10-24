"""
Simple test script to launch SPS tracking with longitudinal beam monitor
"""
import fma_ions
output_dir = './'

n_turns = 1200
num_part = 3000

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns, num_part=num_part, turn_print_interval=20)
tbt = sps.track_SPS(which_context='cpu', install_SC_on_line=False, add_aperture=True, also_keep_delta_profiles=True,
                    nturns_profile_accumulation_interval=300)
tbt.to_json(output_dir)

# Load data and plot
sps_plot = fma_ions.SPS_Plotting()
sps_plot.plot_delta_monitor_data()
