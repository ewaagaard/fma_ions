"""
SPS ideal lattice with space charge - with GPUs for 800 000 turns
"""
import fma_ions
import pandas as pd
output_dir = './'

n_turns = 800_000

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns)
tbt = sps.track_SPS(which_context='gpu')
tbt.to_parquet(f'{output_dir}/tbt.parquet')