"""
SPS lattice with beta-beat and PIC space charge - with GPUs
"""
import fma_ions
import pandas as pd
output_dir = './'

num_turns = 20_000
num_part = 10_000

# Instantiate SPS Flat Bottom Tracker and then track on GPUs
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=num_turns, num_part=num_part)
tbt = sps.track_SPS(ion_type='O', which_context='gpu', Qy_frac=25, SC_mode='frozen', beta_beat=0.1, add_non_linear_magnet_errors=True)
tbt.to_parquet(f'{output_dir}/tbt.parquet')