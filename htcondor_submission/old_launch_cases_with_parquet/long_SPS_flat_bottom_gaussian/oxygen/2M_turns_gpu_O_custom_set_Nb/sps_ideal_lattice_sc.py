"""
SPS ideal lattice with space charge - with GPUs for 2M turns
"""
import fma_ions
import pandas as pd
output_dir = './'

n_turns = 2_000_000

# Update to new bunch intensity
beamParams = fma_ions.BeamParameters_SPS()
beamParams.Nb = 2.5e9

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns)
tbt = sps.track_SPS(ion_type='O', which_context='gpu', beamParams=beamParams)
tbt.to_parquet(f'{output_dir}/tbt.parquet')