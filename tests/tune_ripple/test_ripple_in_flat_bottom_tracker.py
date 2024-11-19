"""
Tester script to ensure tune ripple is on in SPS Flat Bottom Tracker class
- exaggerate tune ripple amplitude, with space charge and magnet errors
"""
import fma_ions

# Instantiate flat bottom tracker object
sps = fma_ions.SPS_Flat_Bottom_Tracker(qx0=26.30, qy0=26.13, num_part=200, num_turns=200, turn_print_interval=1)
tbt = sps.track_SPS(install_SC_on_line=True, add_tune_ripple=True, dq=0.01, add_aperture=True, beta_beat=0.15,
              add_non_linear_magnet_errors=True, add_sextupolar_errors=True, add_octupolar_errors=True)