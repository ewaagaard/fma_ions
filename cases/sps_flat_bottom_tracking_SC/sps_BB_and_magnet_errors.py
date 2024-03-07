"""
SPS lattice with BB, non-linear magnet errors and space charge
"""
import fma_ions

# Test default tracking with space charge on GPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=10_000, _output_folder='output_BB_and_magnet_errors')
tbt = sps.track_SPS(which_context='gpu', add_aperture=True, beta_beat=0.1, add_non_linear_magnet_errors=True)
