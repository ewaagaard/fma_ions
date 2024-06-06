"""
SPS lattice with beta-beat and only IBS (no SC) - with GPUs
"""
import fma_ions
import pandas as pd
output_dir = './'

n_turns = 1_000_000
num_part = 50_000

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns, num_part=num_part)
tbt = sps.track_SPS(which_context='gpu', install_SC_on_line=False, beta_beat=None, add_non_linear_magnet_errors=False, 
                    apply_kinetic_IBS_kicks=True, ibs_step = 5000, save_full_particle_data=True, full_particle_data_interval=int(1e5))
tbt.to_json(f'{output_dir}/tbt.json')