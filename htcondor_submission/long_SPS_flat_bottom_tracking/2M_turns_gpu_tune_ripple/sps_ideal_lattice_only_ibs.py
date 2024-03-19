"""
SPS ideal lattice and only IBS (no SC) and tune ripple  - with GPUs for 2M turns
"""
import fma_ions
import pandas as pd
output_dir = './'

n_turns = 2_000_000

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns)
tbt = sps.track_SPS(which_context='gpu', install_SC_on_line=False, apply_kinetic_IBS_kicks=True, ibs_step = 300, add_tune_ripple=True)
tbt.to_parquet(f'{output_dir}/tbt.parquet')