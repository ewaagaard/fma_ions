"""
Script to check longitudinal phase space with space charge, to see if particles drift outwards
"""
import fma_ions
import os

output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=20, turn_print_interval=2)
#tbt = sps.track_SPS(which_context='cpu', beta_beat=0.1, add_non_linear_magnet_errors=True, 
#                    save_full_particle_data=True, full_particle_data_interval=1, distribution_type='linear_in_zeta')
#tbt.to_json(f'{output_dir}/tbt.json')
sps.plot_longitudinal_phase_space_trajectories(output_folder=output_dir, include_sps_separatrix=False)