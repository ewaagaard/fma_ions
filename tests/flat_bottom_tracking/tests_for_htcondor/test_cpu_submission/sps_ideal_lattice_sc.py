"""
SPS ideal lattice with space charge - CPU for HTCondor submission
"""

import fma_ions

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=100, output_folder='output_ideal_lattice')
tbt = sps.track_SPS(which_context='cpu')
