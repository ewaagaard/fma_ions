"""
Script to track O particles in SPS, study full phase space of all particles 
"""
import fma_ions
import matplotlib.pyplot as plt
import os
import numpy as np

distribution_type = 'gaussian'

# Max zeta for particles
max_zeta = 0.74269329

# Track with 10% BB, space charge and IBS
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_part=10_000, num_turns=50, turn_print_interval=10)
tbt_dict = sps.track_SPS(which_context='gpu', ion_type='O', beta_beat=0.1, add_non_linear_magnet_errors=True, apply_kinetic_IBS_kicks=False,
                save_full_particle_data=True, distribution_type=distribution_type, full_particle_data_interval=1)

# Output directory
os.makedirs('output_plots', exist_ok=True)
os.makedirs('output_plots/phase_space_plots_Y', exist_ok=True)

# Final dead and alive indices
alive_ind_final = tbt_dict.state[:, -1] > 0
dead_ind_final = tbt_dict.state[:, -1] < 1

##### Horizontal phase space #####
fig0, ax0 = plt.subplots(2, 1, figsize = (10, 8), sharey=True)

# Plot initial particles
ax0[0].plot(tbt_dict.x[alive_ind_final, 0], tbt_dict.px[alive_ind_final, 0]*1000, '.', 
        color='blue', markersize=3.6, label='Alive')
ax0[0].plot(tbt_dict.x[dead_ind_final, 0], tbt_dict.px[dead_ind_final, 0]*1000, '.', 
        color='darkred', markersize=3.6, label='Dead')

# Plot final particles
ax0[1].plot(tbt_dict.x[alive_ind_final, -1], tbt_dict.px[alive_ind_final, -1]*1000, '.', 
        color='blue', markersize=3.6, label='Alive')
ax0[1].plot(tbt_dict.x[dead_ind_final, -1], tbt_dict.px[dead_ind_final, -1]*1000, '.', 
        color='darkred', markersize=3.6, label='Dead')
ax0[1].legend(loc='upper right', fontsize=13)
ax0[1].set_xlabel(r'$x$ [m]')
ax0[0].set_ylabel(r'$p_{{x}}$ [1e-3]')
ax0[1].set_ylabel(r'$p_{{x}}$ [1e-3]')
plt.tight_layout()
fig0.savefig('output_plots/SPS_O_phase_space_X.png', dpi=250)


##### Vertical phase space #####
fig01, ax01 = plt.subplots(2, 1, figsize = (10, 8), sharey=True)

# Plot initial particles
ax01[0].plot(tbt_dict.y[alive_ind_final, 0], tbt_dict.py[alive_ind_final, 0]*1000, '.', 
        color='blue', markersize=3.6, label='Alive')
ax01[0].plot(tbt_dict.y[dead_ind_final, 0], tbt_dict.py[dead_ind_final, 0]*1000, '.', 
        color='darkred', markersize=3.6, label='Dead')

# Plot final particles
ax01[1].plot(tbt_dict.y[alive_ind_final, -1], tbt_dict.py[alive_ind_final, -1]*1000, '.', 
        color='blue', markersize=3.6, label='Alive')
ax01[1].plot(tbt_dict.y[dead_ind_final, -1], tbt_dict.py[dead_ind_final, -1]*1000, '.', 
        color='darkred', markersize=3.6, label='Dead')
ax01[1].legend(loc='upper right', fontsize=13)
ax01[1].set_xlabel(r'$y$ [m]')
ax01[0].set_ylabel(r'$p_{{y}}$ [1e-3]')
ax01[1].set_ylabel(r'$p_{{y}}$ [1e-3]')
plt.tight_layout()
fig01.savefig('output_plots/SPS_O_phase_space_Y.png', dpi=250)

# Plot longitudinal phase space, initial and final state
fig, ax = plt.subplots(2, 1, figsize = (10, 8), sharey=True)

# Plot initial particles
ax[0].plot(tbt_dict.zeta[alive_ind_final, 0], tbt_dict.delta[alive_ind_final, 0]*1000, '.', 
        color='blue', markersize=3.6, label='Alive')
ax[0].plot(tbt_dict.zeta[dead_ind_final, 0], tbt_dict.delta[dead_ind_final, 0]*1000, '.', 
        color='darkred', markersize=3.6, label='Dead')

# Plot final particles
ax[1].plot(tbt_dict.zeta[alive_ind_final, -1], tbt_dict.delta[alive_ind_final, -1]*1000, '.', 
        color='blue', markersize=3.6, label='Alive')
ax[1].plot(tbt_dict.zeta[dead_ind_final, -1], tbt_dict.delta[dead_ind_final, -1]*1000, '.', 
        color='darkred', markersize=3.6, label='Dead')
ax[1].legend(loc='upper right', fontsize=13)
ax[1].set_xlabel(r'$\zeta$ [m]')
ax[0].set_ylabel(r'$\delta$ [1e-3]')
ax[1].set_ylabel(r'$\delta$ [1e-3]')
plt.tight_layout()
fig.savefig('output_plots/SPS_O_phase_space_Z.png', dpi=250)


# Also check phase space evolution

for i in range(sps.num_turns):
        plt.close()
        print(f'Plot turn {i+1}')
        fig, ax = plt.subplots(1, 1, figsize = (10,5))
        plt.suptitle(f'Turn {i+1}')
        ax.plot(tbt_dict.y[:, i], tbt_dict.py[:, i], '.', color='blue', markersize=3.6, label='Y')
        ax.set_ylabel(r'$p_{{y}}$ [1e-3]')
        ax.set_ylabel(r'$p_{{y}}$ [1e-3]')
        plt.tight_layout()
        ax.set_ylim(-4e-4, 4e-4)
        ax.set_xlim(-6e-3, 6e-3)
        fig.savefig('output_plots/phase_space_plots_Y/SPS_Pb_longitudinal_check_turn_{}.png'.format(i+1), dpi=250)