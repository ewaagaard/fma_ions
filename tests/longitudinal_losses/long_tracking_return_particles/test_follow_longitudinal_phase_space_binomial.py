"""
Script to track Pb particles in SPS, study full phase space of all particles 
"""
import fma_ions
import matplotlib.pyplot as plt
import os
import json
import xobjects as xo
import xpart as xp

distribution_type = 'binomial'

# Track with 10% BB and space charge 
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_part=100, num_turns=100, turn_print_interval=10)
tbt_dict = sps.track_SPS(which_context='cpu', beta_beat=0.1, add_non_linear_magnet_errors=True,  distribution_type=distribution_type,
                         save_full_particle_data=True, update_particles_and_sc_for_binomial=True)

# Output directory
os.makedirs('output_plots', exist_ok=True)

# Final dead and alive indices
alive_ind_final = tbt_dict['state'][:, -1] > 0
dead_ind_final = tbt_dict['state'][:, -1] < 1

# Plot longitudinal phase space, initial and final state
fig, ax = plt.subplots(2, 1, figsize = (10, 8), sharey=True)

# Plot initial particles
ax[0].plot(tbt_dict['zeta'][alive_ind_final, 0], tbt_dict['delta'][alive_ind_final, 0]*1000, '.', 
        color='blue', markersize=3.6, label='Alive')
ax[0].plot(tbt_dict['zeta'][dead_ind_final, 0], tbt_dict['delta'][dead_ind_final, 0]*1000, '.', 
        color='darkred', markersize=3.6, label='Dead')
ax[0].plot(sps._zeta_separatrix, sps._delta_separatrix * 1e3, '-', color='red', alpha=0.7, label='SPS RF separatrix')
ax[0].plot(sps._zeta_separatrix, -sps._delta_separatrix * 1e3, '-', color='red', alpha=0.7, label=None)

# Plot final particles
ax[1].plot(tbt_dict['zeta'][alive_ind_final, -1], tbt_dict['delta'][alive_ind_final, -1]*1000, '.', 
        color='blue', markersize=3.6, label='Alive')
ax[1].plot(tbt_dict['zeta'][dead_ind_final, -1], tbt_dict['delta'][dead_ind_final, -1]*1000, '.', 
        color='darkred', markersize=3.6, label='Dead')
ax[1].plot(sps._zeta_separatrix, sps._delta_separatrix * 1e3, '-', color='red', alpha=0.7, label='SPS RF separatrix')
ax[1].plot(sps._zeta_separatrix, -sps._delta_separatrix * 1e3, '-', color='red', alpha=0.7, label=None)
ax[1].legend(loc='upper right', fontsize=13)

#ax.set_ylim(-1.4, 1.4)
#ax.set_xlim(-1.0, 1.0)
ax[1].set_xlabel(r'$\zeta$ [m]')
ax[0].set_ylabel(r'$\delta$ [1e-3]')
ax[1].set_ylabel(r'$\delta$ [1e-3]')
plt.tight_layout()
fig.savefig('output_plots/SPS_Pb_longitudinal_turn_{}.png'.format(distribution_type), dpi=250)
