"""
First tests of SPS flat bottom tracking class
"""
import fma_ions

# Test default tracking with GPUs
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=100)
sps.track_SPS()