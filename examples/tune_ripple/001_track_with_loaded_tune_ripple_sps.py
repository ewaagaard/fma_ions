"""
Small test script to track a single SPS particle for 100 turns, with a
loaded kqf and kqd ripple from the uncompensated case
"""
import fma_ions

sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=100, num_part=1, turn_print_interval=5)
tbt = sps.track_SPS(which_context='cpu', distribution_type='single', install_SC_on_line=False, add_tune_ripple=True,
                    load_full_spectrum=True, apply_50_Hz_comp=False)

