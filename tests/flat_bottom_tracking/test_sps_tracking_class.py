"""
First tests of SPS flat bottom tracking class
"""
import fma_ions

# Test default tracking with CPU - test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=100)
tbt = sps.track_SPS(which_context='cpu')