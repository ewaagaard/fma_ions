"""
Test updating binomial space charge 
"""
import fma_ions
import matplotlib.pyplot as plt

# Test default tracking with space charge on GPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_part=1000, num_turns=100, turn_print_interval=10)
tbt_dict = sps.track_SPS(which_context='gpu', distribution_type='binomial', add_aperture=True, 
                    save_full_particle_data=True, update_particles_and_sc_for_binomial=True)
#sps.plot_tracking_data(tbt_dict, show_plot=True)

# Final dead and alive indices
alive_ind_final = tbt_dict['state'][:, -1] > 0
dead_ind_final = tbt_dict['state'][:, -1] < 1


# Save all plots, or only first and last
for i in [0, sps.num_turns-1]:
    plt.close()
    print(f'Plot turn {i+1}')
    fig, ax = plt.subplots(1, 1, figsize = (10,5))
    plt.suptitle(f'Turn {i+1}')
    ax.plot(tbt_dict['zeta'][alive_ind_final, i], tbt_dict['delta'][alive_ind_final, i]*1000, '.', 
            color='blue', markersize=3.6, label='Alive')
    ax.plot(tbt_dict['zeta'][dead_ind_final, i], tbt_dict['delta'][dead_ind_final, i]*1000, '.', 
            color='darkred', markersize=3.6, label='Dead')
    ax.set_ylim(-1.4, 1.4)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel(r'$\zeta$ [m]')
    ax.set_ylabel(r'$\delta$ [1e-3]')
    plt.tight_layout()
    fig.savefig('SPS_Pb_longitudinal_turn_{}.png'.format(i+1), dpi=250)