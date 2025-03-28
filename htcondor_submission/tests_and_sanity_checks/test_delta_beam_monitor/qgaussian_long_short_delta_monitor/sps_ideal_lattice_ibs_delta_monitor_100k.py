"""
SPS lattice with ideal lattice - with GPUs
"""
import fma_ions
output_dir = './'

n_turns = 100_000
num_part = 10_000

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns, num_part=num_part)
tbt = sps.track_SPS(which_context='gpu', install_SC_on_line=False, apply_kinetic_IBS_kicks=True, 
                    distribution_type='qgaussian', also_keep_delta_profiles=True)
tbt.to_json(output_dir)
