"""
Test SPS flat bottom tracking with GPUs and PIC space charge
"""

import fma_ions
import pandas as pd
output_dir = './'

# Instantiate SPS Flat Bottom Tracker and then track on GPUs
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=20)
tbt = sps.track_SPS(which_context='gpu', Qy_frac=19, beta_beat=0.1, SC_mode='PIC',
                    add_non_linear_magnet_errors=True, apply_kinetic_IBS_kicks=True)