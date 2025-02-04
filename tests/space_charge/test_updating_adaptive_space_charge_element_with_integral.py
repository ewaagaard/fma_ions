"""
Small test to investigate how much the beam profiles will differ from the Gaussian fit for the space charge element, and then update accordingly
"""
import fma_ions
import numpy as np
import matplotlib.pyplot as plt

n_turns = 500
num_part = 400

# Tracking on GPU context
sps = fma_ions.SPS_Flat_Bottom_Tracker(qx0=26.31, qy0=26.10, num_turns=n_turns, num_part=num_part, turn_print_interval=100)
tbt = sps.track_SPS(which_context='cpu', distribution_type='gaussian', install_SC_on_line=True, add_beta_beat=True,
                SC_adaptive_interval_during_tracking=100, adjust_integral_for_SC_adaptive_interval_during_tracking=True)