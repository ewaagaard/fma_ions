"""
SPS ideal lattice with space charge
"""

import fma_ions

# Test default tracking with space charge on GPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=10_000, _output_folder='output_ideal_lattice')
tbt = sps.track_SPS(which_context='gpu', add_aperture=True)
