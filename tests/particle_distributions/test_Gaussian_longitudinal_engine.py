"""
Test script of two different Gaussian longitudinal RF matching engines
"""
import fma_ions
import xobjects as xo
import os
import pandas as pd

tracking_is_done = True

# Define cupy context
context = xo.ContextCupy()

# Test default tracking with space charge on GPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_part=10_000, num_turns=10_000, turn_print_interval=100)

beamParams2 = fma_ions.BeamParameters_SPS()
beamParams2.sigma_z = 0.19 

if not tracking_is_done:
    # Test with single-RF-engine
    df = sps.track_SPS(Qy_frac=19, ibs_step=5000,
                beamParams=beamParams2, which_context='gpu', 
                engine='single-rf-harmonic', save_tbt_data=True)

    # Test with single-RF-engine
    df2 = sps.track_SPS(Qy_frac=19, which_context='gpu', ibs_step=5000,
                beamParams=beamParams2, 
                engine=None, save_tbt_data=True)


    # Make output directory for data
    os.makedirs('Output', exist_ok=True)
    df.to_parquet('Output/tbt_single_rf_harmonic_engine.parquet')
    df2.to_parquet('Output/tbt_normal_engine.parquet')

df = pd.read_parquet('Output/tbt_single_rf_harmonic_engine.parquet')
df2 = pd.read_parquet('Output/tbt_normal_engine.parquet')

print('Final Nb with single-rf-harmonic engine: {:.5e}'.format(df['Nb'].iloc[-1]))
print('Final Nb with default engine: {:.5e}'.format(df2['Nb'].iloc[-1]))