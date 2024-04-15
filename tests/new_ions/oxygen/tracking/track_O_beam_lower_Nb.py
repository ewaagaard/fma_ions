"""
SPS lattice with beta-beat and space charge - with GPUs for 2M turns - lower the Nb
"""
import fma_ions
import pandas as pd
output_dir = './'

n_turns = 1_000

beamParams = fma_ions.BeamParameters_SPS()
beamParams.Nb = 1e9

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns, output_folder='output_lower_Nb')
tbt = sps.track_SPS(ion_type='O', which_context='gpu', beta_beat=0.1, add_non_linear_magnet_errors=True, beamParams=beamParams)
sps.plot_tracking_data(tbt)
