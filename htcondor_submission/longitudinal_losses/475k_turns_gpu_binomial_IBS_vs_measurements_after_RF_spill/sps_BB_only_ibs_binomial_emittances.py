"""
SPS lattice with beta-beat and only IBS (no SC) - with GPUs. Only save emittances and global quantities
"""
import fma_ions
import pandas as pd
output_dir = './'

n_turns = 475_000  # corresponds to 22 s
num_part = 10_000

# Instantiate beam parameters, custom made to compare with 2016 measurements
beamParams = fma_ions.BeamParameters_SPS()
beamParams.Nb = 3.536e8 * 0.95  # loss factor from first turn observed with wall current monitor
beamParams.exn = 1.3e-6 # in m
beamParams.eyn = 0.8e-6 # in m
beamParams.sigma_z_binomial = 0.215 # what we measure after initial losses out of the RF bucket
beamParams.m = 2.8 # meausred for binomial after SPS injection
Qy_frac = 25 # old fractional tune

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns, num_part=num_part)
tbt = sps.track_SPS(which_context='gpu', Qy_frac=Qy_frac, beamParams=beamParams, install_SC_on_line=False, beta_beat=0.1, 
                    add_non_linear_magnet_errors=True, apply_kinetic_IBS_kicks=True, ibs_step = 5000, save_full_particle_data=False,
                    update_particles_and_sc_for_binomial=False, distribution_type='binomial')
tbt.to_parquet(f'{output_dir}/tbt.parquet')  # parquet if we only need 1D arrays with ensemble values