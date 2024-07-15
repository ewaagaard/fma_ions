"""
Test tracking with kinetic IBS kicks from xfields
"""
import fma_ions

# Test default tracking with space charge on GPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_part=10_000, num_turns=100, turn_print_interval=10)
tbt = sps.track_SPS(install_SC_on_line=False, add_aperture=False, apply_kinetic_IBS_kicks=True, ibs_step=10)
