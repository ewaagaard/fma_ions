"""
SPS ideal lattice with space charge - CPU for HTCondor submission
"""
import fma_ions
import pandas as pd
output_dir = './'

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=100)
tbt = sps.track_SPS(which_context='cpu')
tbt.to_parquet(f'{output_dir}/tbt.parquet')