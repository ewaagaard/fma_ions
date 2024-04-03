"""
Test tracking oxygen with kinetic IBS kicks applied
"""
import fma_ions

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_part=100, num_turns=100, turn_print_interval=10)
tbt = sps.track_SPS(ion_type='O', which_context='cpu', add_aperture=True, apply_kinetic_IBS_kicks=True)
sps.plot_tracking_data(tbt, show_plot=True)
