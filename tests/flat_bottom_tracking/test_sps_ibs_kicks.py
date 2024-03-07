"""
Test tracking with kinetic IBS kicks applied
"""
import fma_ions

# Test default tracking with space charge on GPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=100, turn_print_interval=10)
tbt = sps.track_SPS(which_context='cpu', add_aperture=False, apply_kinetic_IBS_kicks=True)

