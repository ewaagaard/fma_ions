"""
SPS lattice with BB and space charge
"""
import fma_ions

# Test default tracking with space charge on GPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=10_000, _output_folder='output_BB')
tbt = sps.track_SPS(which_context='gpu', add_aperture=True, beta_beat=0.1)
