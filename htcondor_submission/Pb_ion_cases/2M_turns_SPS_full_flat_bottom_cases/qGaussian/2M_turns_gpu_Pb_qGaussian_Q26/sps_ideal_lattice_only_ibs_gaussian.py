"""
SPS lattice with ideal lattice and only IBS (no SC) - with GPUs
"""
import fma_ions
output_dir = './'

n_turns = 2_000_000  # corresponds to 45 s
num_part = 10_000

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns, num_part=num_part, qy0=26.19)
tbt = sps.track_SPS(which_context='gpu', beamParams=fma_ions.BeamParameters_SPS(), install_SC_on_line=False, apply_kinetic_IBS_kicks=True, 
                    distribution_type='gaussian')
tbt.to_json(output_dir)
