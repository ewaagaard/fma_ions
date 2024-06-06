"""
SPS ideal lattice with space charge - change RF voltage to 1MV
"""
import fma_ions
import pandas as pd
output_dir = './'

n_turns = 500_000

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns)
tbt = sps.track_SPS(which_context='gpu', voltage=1e6, save_full_particle_data=True, full_particle_data_interval=1, 
                    distribution_type='linear_in_zeta')
tbt.to_json(f'{output_dir}/tbt.json')