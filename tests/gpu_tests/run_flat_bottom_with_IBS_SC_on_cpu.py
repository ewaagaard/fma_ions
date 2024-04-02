"""
IBS kinetic kicks on CPU context for SPS flat bottom tracking class
"""
import fma_ions

# Test default tracking with space charge on GPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=10)
tbt = sps.track_SPS(which_context='cpu', add_aperture=True, apply_kinetic_IBS_kicks=True, beta_beat=0.1,
                    add_non_linear_magnet_errors=True)
