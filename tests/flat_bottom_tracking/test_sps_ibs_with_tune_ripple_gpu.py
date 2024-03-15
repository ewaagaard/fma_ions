"""
Test tracking with kinetic IBS kicks applied and tune ripple
"""
import fma_ions

# Test default tracking with space charge on GPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=100, turn_print_interval=10)
tbt = sps.track_SPS(Qy_frac=19, which_context='cpu', add_aperture=True, apply_kinetic_IBS_kicks=True, add_tune_ripple=True)

