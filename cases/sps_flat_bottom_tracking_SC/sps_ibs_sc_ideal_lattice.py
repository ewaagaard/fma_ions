"""
SPS ideal lattice with space charge and IBS
"""
import fma_ions

# Test default tracking with space charge and IBS on GPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=10_000, output_folder='output_ideal_lattice_ibs')
tbt = sps.track_SPS(which_context='gpu', add_aperture=True, apply_kinetic_IBS_kicks=True, ibs_step=100)
