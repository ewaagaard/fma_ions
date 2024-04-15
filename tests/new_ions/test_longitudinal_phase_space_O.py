"""
Script to track O particles in SPS, study full phase space of all particles 
"""
import fma_ions
import matplotlib.pyplot as plt
import os

distribution_types = ['gaussian']#, 'binomial']
plot_only_first_and_last_turn = False

# Max zeta for particles
max_zeta = 0.74269329

# Track with 10% BB, space charge and IBS
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_part=10_000, num_turns=100, turn_print_interval=10)

for distribution_type in distribution_types:
    tbt_dict = sps.track_SPS(which_context='gpu', ion_type='O', beta_beat=0.1, add_non_linear_magnet_errors=True, apply_kinetic_IBS_kicks=False,
                        save_full_particle_data=True, distribution_type=distribution_type)

    # Output directory
    os.makedirs('output_plots_{}'.format(distribution_type), exist_ok=True)
    os.makedirs('output_plots_f_{}'.format(distribution_type), exist_ok=True)

    # Final dead and alive indices
    alive_ind_final = tbt_dict.state[:, -1] > 0
    dead_ind_final = tbt_dict.state[:, -1] < 1

    # Plot longitudinal phase space, turn by turn

    # Save all plots, or only first and last
    if plot_only_first_and_last_turn: 
        for i in [0, sps.num_turns-1]:
            plt.close()
            print(f'Plot turn {i+1}')
            fig, ax = plt.subplots(1, 1, figsize = (10,5))
            plt.suptitle(f'Turn {i+1}')
            ax.plot(tbt_dict.zeta[alive_ind_final, i], tbt_dict.delta[alive_ind_final, i]*1000, '.', 
                    color='blue', markersize=3.6, label='Alive')
            ax.plot(tbt_dict.zeta[dead_ind_final, i], tbt_dict.delta[dead_ind_final, i]*1000, '.', 
                    color='darkred', markersize=3.6, label='Dead')
            ax.axvline(x=max_zeta, color='orange', linestyle='dashed')
            ax.axvline(x=-max_zeta, color='orange', linestyle='dashed')
            ax.set_ylim(-1.4, 1.4)
            ax.set_xlim(-1.0, 1.0)
            ax.set_xlabel(r'$\zeta$ [m]')
            ax.set_ylabel(r'$\delta$ [1e-3]')
            plt.tight_layout()
            fig.savefig('output_plots_f_{}/SPS_O_longitudinal_turn_{}.png'.format(distribution_type, i+1), dpi=250)
    else:
        for i in range(sps.num_turns):

            plt.close()
            print(f'Plot turn {i+1}')
            fig, ax = plt.subplots(1, 1, figsize = (10,5))
            plt.suptitle(f'Turn {i+1}')
            ax.plot(tbt_dict.zeta[alive_ind_final, i], tbt_dict.delta[alive_ind_final, i]*1000, '.', 
                    color='blue', markersize=3.6, label='Alive')
            ax.plot(tbt_dict.zeta[dead_ind_final, i], tbt_dict.delta[dead_ind_final, i]*1000, '.', 
                    color='darkred', markersize=3.6, label='Dead')
            ax.axvline(x=max_zeta, color='orange', linestyle='dashed')
            ax.axvline(x=-max_zeta, color='orange', linestyle='dashed')
            ax.set_ylim(-1.4, 1.4)
            ax.set_xlim(-1.0, 1.0)
            ax.set_xlabel(r'$\zeta$ [m]')
            ax.set_ylabel(r'$\delta$ [1e-3]')
            plt.tight_layout()
            fig.savefig('output_plots_{}/SPS_O_longitudinal_turn_{}.png'.format(distribution_type, i+1), dpi=250)

