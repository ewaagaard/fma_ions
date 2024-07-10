"""
SPS lattice with beta-beat and PIC space charge and IBS - with GPUs for 50 000 turns
"""
import fma_ions
import pandas as pd
output_dir = './'

# Instantiate SPS Flat Bottom Tracker and then track on GPUs
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=1_000, num_part=50_000)
tbt = sps.track_SPS(which_context='gpu', Qy_frac=19, beta_beat=0.1, SC_mode='PIC',
                    add_non_linear_magnet_errors=True, apply_kinetic_IBS_kicks=True)
tbt.to_parquet(f'{output_dir}/tbt.parquet')