"""
First tests of SPS flat bottom tracking class with different rf voltage
"""
import fma_ions

# Test default tracking with space charge
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=10)
tbt = sps.track_SPS(voltage=2e6, which_context='cpu')
