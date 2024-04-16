"""
SPS lattice with beta-beat and space charge
"""
import fma_ions
import pandas as pd
output_dir = './'

n_turns = 2_000

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns, output_folder='output_isabelle_params_Qydot20')
tbt = sps.track_SPS(ion_type='O', Qy_frac=20, which_context='gpu', beta_beat=0.1, add_non_linear_magnet_errors=True)
sps.plot_tracking_data(tbt)
