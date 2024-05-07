"""
SPS ideal lattice with space charge - with GPUs
"""
import fma_ions
import pandas as pd
output_dir = './'

n_turns = 10_000

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns)
tbt = sps.track_SPS(ion_type='Pb', which_context='gpu', save_full_particle_data=True, full_particle_data_interval=1, 
                    install_SC_on_line=False, apply_kinetic_IBS_kicks=True, beta_beat=0.1, add_non_linear_magnet_errors=True,
                    distribution_type='linear_in_zeta')
tbt.to_json(f'{output_dir}/tbt.json')