"""
SPS lattice with beta-beat and IBS and tune ripple - with GPUs for oxygen
"""
import fma_ions
import pandas as pd
output_dir = './'

n_turns = 100_000

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns)
tbt = sps.track_SPS(ion_type='O', Qy_frac=25, which_context='gpu', install_SC_on_line=False, beta_beat=0.1, 
                    add_non_linear_magnet_errors=True, apply_kinetic_IBS_kicks=True)
tbt.to_parquet(f'{output_dir}/tbt.parquet')