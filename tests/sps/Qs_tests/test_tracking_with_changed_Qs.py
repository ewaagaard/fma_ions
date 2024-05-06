"""
SPS Pb ions - track particles with and without synchrotron tune modification
"""
import fma_ions
import os

os.makedirs('run1', exist_ok=True)
os.makedirs('run2', exist_ok=True)

# Tracking without changing synchrotron tune
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=10, output_folder='run1')
tbt = sps.track_SPS(which_context='cpu', save_full_particle_data=True, full_particle_data_interval=1, 
                    distribution_type='linear_in_zeta')
tbt.to_json('run1/tbt.json')
sps.plot_longitudinal_phase_space_trajectories(output_folder='run1')

# Then track again, updating synchrotron tune to be twice as fast
tbt2 = sps.track_SPS(which_context='cpu', save_full_particle_data=True, full_particle_data_interval=1, 
                    distribution_type='linear_in_zeta', scale_factor_Qs=2)
tbt2.to_json('run2/tbt.json')
sps.plot_longitudinal_phase_space_trajectories(output_folder='run2', extra_plt_str='_scale_factor2', scale_factor_Qs=2)