"""
Test tracking with kinetic IBS kicks applied
"""
import fma_ions

# Test default tracking with space charge on GPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=5, turn_print_interval=1)
tbt = sps.track_SPS(which_context='cpu', distribution_type='binomial')

sps_plot = fma_ions.SPS_Plotting()
sps_plot.plot_tracking_data(tbt.to_dict(convert_to_numpy=True))
sps_plot.plot_longitudinal_monitor_data(tbt.to_dict(convert_to_numpy=True))

