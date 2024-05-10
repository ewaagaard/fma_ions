"""
First tests of SPS flat bottom tracking class with protons
"""
import fma_ions

# Test default tracking with space charge on GPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=10)
tbt = sps.track_SPS(ion_type='proton', which_context='cpu')

# Test loading the data correctly again
# tbt2 = sps.load_tbt_data()