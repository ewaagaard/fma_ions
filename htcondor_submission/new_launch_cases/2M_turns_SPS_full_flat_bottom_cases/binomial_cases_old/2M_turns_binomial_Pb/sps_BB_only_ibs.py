"""
SPS lattice with Pb ions - GPU
"""
import fma_ions
import pandas as pd
output_dir = './'

n_turns = 2_000_000
num_part = 10_000

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns, num_part=num_part)
tbt = sps.track_SPS(which_context='gpu', install_SC_on_line=False, beta_beat=0.05,  
                    add_non_linear_magnet_errors=True, apply_kinetic_IBS_kicks=True, distribution_type='binomial')
tbt.to_json(output_dir)
