"""
SPS lattice with beta-beat and space charge and IBS - GPU for HTCondor submission
"""
import fma_ions
import pandas as pd
output_dir = './'

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=100)
tbt = sps.track_SPS(which_context='gpu', beta_beat=0.1, add_non_linear_magnet_errors=True, apply_kinetic_IBS_kicks=True)
tbt.to_parquet(f'{output_dir}/tbt.parquet')