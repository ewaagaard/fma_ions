"""
Script to check correct bunch length of binomial distribution - after RF spill
"""
import fma_ions
import pandas as pd
output_dir = './'

tracking_has_been_done = True

n_turns = 500  
num_part = 5_000

# Instantiate beam parameters, custom made to compare with 2016 measurements
beamParams = fma_ions.BeamParameters_SPS()
beamParams.Nb = 3.536e8 * 0.95  # loss factor from first turn observed with wall current monitor
beamParams.exn = 1.3e-6 # in m
beamParams.eyn = 0.8e-6 # in m
beamParams.sigma_z_binomial = 0.225 # what we measure after initial losses out of the RF bucket
beamParams.m = 2.8 # meausred for binomial after SPS injection
Qy_frac = 25 # old fractional tune

if not tracking_has_been_done:
    sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns, num_part=num_part, turn_print_interval=10)
    tbt = sps.track_SPS(which_context='cpu', Qy_frac=Qy_frac, beamParams=beamParams, install_SC_on_line=False,
                        apply_kinetic_IBS_kicks=True, distribution_type='binomial', nturns_profile_accumulation_interval=10)
    tbt.to_json(output_dir)


sps_plot = fma_ions.SPS_Plotting()
sps_plot.plot_longitudinal_monitor_data(inj_profile_is_after_RF_spill=True, also_compare_with_profile_data=False)
sps_plot.plot_tracking_data()