"""
SPS ideal lattice with space charge - with GPUs for 50 000 turns
"""
import fma_ions
import pandas as pd
output_dir = './'

# Instantiate SPS Flat Bottom Tracker and then track on GPUs
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=50_000)
tbt = sps.track_SPS(which_context='gpu', Qy_frac=19)
tbt.to_parquet(f'{output_dir}/tbt.parquet')