"""
Small example to kick beam and plot TBT data for 200 macroparticles
"""
import fma_ions
import numpy as np
import matplotlib.pyplot as plt

# Try single particle
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=500, num_part=200, turn_print_interval=20)
tbt = sps.track_SPS(which_context='cpu', distribution_type='single', install_SC_on_line=False, 
                    add_tune_ripple=True, kick_beam=True)
tbt_dict = tbt.to_dict()

# plot turn-by-turn data
fig, ax = plt.subplots(2, 1, sharex=True, constrained_layout=True)
ax[0].plot(tbt_dict['X_data'], color='b')
ax[1].plot(tbt_dict['Y_data'], color='darkorange')
ax[0].set_ylabel('X [m]')
ax[1].set_ylabel('Y [m]')
ax[1].set_xlabel('Turns')
plt.show()