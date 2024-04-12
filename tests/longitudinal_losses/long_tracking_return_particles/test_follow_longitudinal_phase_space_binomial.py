"""
Script to track Pb particles in SPS, study full phase space of all particles 
"""
import fma_ions
import matplotlib.pyplot as plt
import os

distribution_type = 'binomial'

# Track with 10% BB and space charge 
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_part=10_000, num_turns=100, turn_print_interval=10)
tbt_dict, particles, particles0 = sps.track_SPS(which_context='gpu', beta_beat=0.1, add_non_linear_magnet_errors=True,  distribution_type=distribution_type,
                         save_final_particles=True)

# Output directory
os.makedirs('output_plots', exist_ok=True)

# Final dead and alive indices
alive_ind_final = particles.state > 0
dead_ind_final = particles.state < 1

# Plot longitudinal phase space, initial and final state
fig, ax = plt.subplots(2, 1, figsize = (10, 8), sharey=True)

# Plot initial particles
ax[0].plot(particles0.zeta[alive_ind_final], particles0.delta[alive_ind_final]*1000, '.', 
        color='blue', markersize=3.6, label='Alive')
ax[0].plot(particles0.zeta[dead_ind_final], particles0.delta[dead_ind_final]*1000, '.', 
        color='darkred', markersize=3.6, label='Dead')
ax[0].plot(sps._zeta_separatrix, sps._delta_separatrix * 1e3, '-', color='red', label='SPS RF separatrix')
ax[0].plot(sps._zeta_separatrix, -sps._delta_separatrix * 1e3, '-', color='red', label=None)

# Plot final particles
ax[1].plot(particles.zeta[alive_ind_final], particles.delta[alive_ind_final]*1000, '.', 
        color='blue', markersize=3.6, label='Alive')
ax[1].plot(particles.zeta[dead_ind_final], particles.delta[dead_ind_final]*1000, '.', 
        color='darkred', markersize=3.6, label='Dead')
ax[1].plot(sps._zeta_separatrix, sps._delta_separatrix * 1e3, '-', color='red', label='SPS RF separatrix')
ax[1].plot(sps._zeta_separatrix, -sps._delta_separatrix * 1e3, '-', color='red', label=None)

#ax.set_ylim(-1.4, 1.4)
#ax.set_xlim(-1.0, 1.0)
ax[1].set_xlabel(r'$\zeta$ [m]')
ax[0].set_ylabel(r'$\delta$ [1e-3]')
ax[1].set_ylabel(r'$\delta$ [1e-3]')
plt.tight_layout()
fig.savefig('output_plots/SPS_Pb_longitudinal_turn_{}.png'.format(distribution_type), dpi=250)
