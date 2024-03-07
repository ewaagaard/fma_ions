"""
First tests of SPS flat bottom tracking class
"""
import fma_ions

# Test default tracking with space charge on GPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=100)
tbt = sps.track_SPS(which_context='gpu', add_aperture=False)

# Test loading the data correctly again
tbt2 = sps.load_tbt_data()