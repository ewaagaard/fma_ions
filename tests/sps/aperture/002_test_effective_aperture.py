## Test script to launch SPS tracker with effective aperture
## Try tests close to the half-integer tune

import fma_ions
output_dir = './'

n_turns = 500
num_part = 2000

# Tracking on GPU context
sps = fma_ions.SPS_Flat_Bottom_Tracker(qx0=26.31, qy0=26.10, num_turns=n_turns, num_part=num_part, turn_print_interval=50)
tbt = sps.track_SPS(which_context='gpu', distribution_type='qgaussian', install_SC_on_line=True, add_beta_beat=True,
                add_non_linear_magnet_errors=True, apply_kinetic_IBS_kicks=True, ibs_step = 2000, add_tune_ripple=True, 
                SC_adaptive_interval_during_tracking=100, x_max_at_WS=0.025, y_max_at_WS=0.013)
tbt.to_json(output_dir)