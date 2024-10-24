"""
SPS lattice with ideal lattice and only IBS (no SC) - with GPUs
"""
import fma_ions
import pandas as pd
output_dir = './'

n_turns = 2_000_000
num_part = 10_000

sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns, num_part=num_part)
tbt = sps.track_SPS(which_context='gpu', install_SC_on_line=False, apply_kinetic_IBS_kicks=True, distribution_type='binomial')
tbt.to_json(output_dir)
