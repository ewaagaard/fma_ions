"""
SPS lattice with beta-beat and space charge - CPU for HTCondor submission
"""
import fma_ions

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=100)
tbt = sps.track_SPS(which_context='cpu', beta_beat=0.1, add_non_linear_magnet_errors=True)
