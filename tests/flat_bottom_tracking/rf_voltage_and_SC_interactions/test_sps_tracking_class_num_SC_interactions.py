"""
First tests of SPS flat bottom tracking class with different number of SC interactions
"""
import fma_ions

# Test default tracking with space charge
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=10)
tbt = sps.track_SPS(num_spacecharge_interactions=100, which_context='cpu', add_aperture=False)
