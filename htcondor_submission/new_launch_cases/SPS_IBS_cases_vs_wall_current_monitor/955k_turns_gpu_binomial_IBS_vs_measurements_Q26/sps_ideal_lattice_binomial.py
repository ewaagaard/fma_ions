"""
SPS lattice with ideal lattice with nothing else - with GPUs
"""
import fma_ions
import pandas as pd
output_dir = './'

n_turns = 955_000  # corresponds to 22 s
num_part = 10_000

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns, num_part=num_part)
tbt = sps.track_SPS(which_context='gpu', install_SC_on_line=False, 
                    apply_kinetic_IBS_kicks=False, distribution_type='binomial')
tbt.to_json(output_dir)
