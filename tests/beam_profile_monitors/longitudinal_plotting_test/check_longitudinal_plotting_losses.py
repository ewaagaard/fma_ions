"""
Simple test script to launch SPS tracking - check that longitudinal losses are plotted correctly

Want to ensure that particles that are lost are not included in plots
"""
import fma_ions

n_turns = 100
num_part = 10_000

# Test default tracking with space charge on CPU context - then test plotting
#sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns, num_part=num_part, turn_print_interval=20)
#tbt = sps.track_SPS(which_context='cpu', install_SC_on_line=False, apply_kinetic_IBS_kicks=True,
#                    distribution_type='binomial', nturns_profile_accumulation_interval=25)
#tbt.to_json()

# Load data and plot
sps_plot = fma_ions.SPS_Plotting()
#sps_plot.plot_tracking_data()
sps_plot.plot_longitudinal_monitor_data()
#sps_plot.plot_WS_profile_monitor_data()