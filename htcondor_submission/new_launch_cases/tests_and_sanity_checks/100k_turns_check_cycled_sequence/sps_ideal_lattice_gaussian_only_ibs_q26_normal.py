"""
SPS lattice with beta-beat and SC - with GPUs
"""
import fma_ions
output_dir = './'

n_turns = 100_000 
num_part = 10_000

beamParams = fma_ions.BeamParameters_SPS_Binomial_2016()
beamParams.Nb = 3.59e9 # take intensity times 10 to see effect more strongly

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns, num_part=num_part)
tbt = sps.track_SPS(which_context='gpu', beamParams=beamParams, install_SC_on_line=False, apply_kinetic_IBS_kicks=True, distribution_type='gaussian', cycle_sequence_to_minimum_dpx=False)
tbt.to_json(output_dir)
