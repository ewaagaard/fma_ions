"""
SPS lattice with beta-beat and SC and IBS - with GPUs
"""
import fma_ions
output_dir = './'

n_turns = 955_000  # corresponds to 22 s
num_part = 10_000

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns, num_part=num_part)
tbt = sps.track_SPS(which_context='gpu', install_SC_on_line=True, beta_beat=0.05, plane_for_beta_beat='both',
                    add_non_linear_magnet_errors=True, apply_kinetic_IBS_kicks=True, distribution_type='qgaussian')
tbt.to_json(output_dir)
