"""
Tester script to ensure tune ripple is on in SPS Flat Bottom Tracker class
"""
import fma_ions

# Instantiate flat bottom tracker object
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_part=100, num_turns=50, turn_print_interval=5)
sps.track_SPS(install_SC_on_line=False, add_tune_ripple=True, dq=0.1, install_beam_monitors=True)