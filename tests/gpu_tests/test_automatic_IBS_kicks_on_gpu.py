"""
IBS kinetic kicks on GPU context for SPS flat bottom tracking class - automatically updating growth rates
"""
import fma_ions

# Test default tracking with space charge on GPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=100)
tbt = sps.track_SPS(which_context='gpu', add_aperture=True, apply_kinetic_IBS_kicks=True, auto_recompute_ibs_coefficients=True)
