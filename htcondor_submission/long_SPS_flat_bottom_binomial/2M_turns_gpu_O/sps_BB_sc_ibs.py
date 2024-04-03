"""
SPS lattice with beta-beat and space charge and IBS - with GPUs for 2M turns
"""
import fma_ions
import pandas as pd
output_dir = './'

n_turns = 2_000_000

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns)
tbt = sps.track_SPS(ion_type='O', which_context='gpu', beta_beat=0.1, add_non_linear_magnet_errors=True, 
                    distribution_type='binomial', apply_kinetic_IBS_kicks=True, ibs_step = 5000)
tbt.to_parquet(f'{output_dir}/tbt.parquet')