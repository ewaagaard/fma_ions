## Test script to launch SPS tracker with effective aperture
## Try tests close to the half-integer tune

import fma_ions
import numpy as np
import matplotlib.pyplot as plt

output_dir = './'

n_turns = 500
num_part = 400

# Tracking on GPU context
sps = fma_ions.SPS_Flat_Bottom_Tracker(qx0=26.31, qy0=26.10, num_turns=n_turns, num_part=num_part, turn_print_interval=50)
tbt = sps.track_SPS(which_context='cpu', distribution_type='qgaussian', install_SC_on_line=True, add_beta_beat=True,
                add_non_linear_magnet_errors=True, apply_kinetic_IBS_kicks=True, ibs_step = 2000, add_tune_ripple=True, 
                SC_adaptive_interval_during_tracking=100, x_max_at_WS=0.003, y_max_at_WS=0.003)
tbt.to_json(output_dir)

# Plot the resulting profiles
sps_plot = fma_ions.SPS_Plotting()
tbt_dict = sps_plot.load_records_dict_from_json()
fig, ax = plt.subplots(1, 1, figsize = (8, 6), constrained_layout=True)
fig2, ax2 = plt.subplots(1, 1, figsize = (8, 6), constrained_layout=True)

index_to_plot = [0, -1] #[-1] #
plot_str = ['Simulated first 100 turns', 'Simulated last 100 turns'] #['Simulated, last 100 turns']
colors = ['blue', 'orange']

for j, ind in enumerate(index_to_plot):
    # Normalize bin heights
    x_bin_heights_sorted = np.array(sorted(tbt_dict['monitorH_x_intensity'][ind], reverse=True))
    x_height_max_avg = np.mean(x_bin_heights_sorted[:3]) # take average of top 3 values
    X_pos_data = tbt_dict['monitorH_x_grid']
    if ind == 0:
        X0_profile_data = tbt_dict['monitorH_x_intensity'][ind] / x_height_max_avg
        ax.plot(X_pos_data, X0_profile_data, label=plot_str[j], color=colors[j])
    else:
        X_profile_data = tbt_dict['monitorH_x_intensity'][ind] / x_height_max_avg
        ax.plot(X_pos_data, X_profile_data, label=plot_str[j], color=colors[j])

ax.set_xlabel('x [m]')
ax.set_ylabel('Normalized counts')
ax.set_ylim(0, 1.1)

# Plot profile of particles
for j, ind in enumerate(index_to_plot):
    # Normalize bin heights
    y_bin_heights_sorted = np.array(sorted(tbt_dict['monitorV_y_intensity'][ind], reverse=True))
    y_height_max_avg = np.mean(y_bin_heights_sorted[:3]) # take average of top 3 values
    Y_pos_data = tbt_dict['monitorV_y_grid']
    if ind == 0:
        Y0_profile_data = tbt_dict['monitorV_y_intensity'][ind] / y_height_max_avg ### if changes fast, take particle histogram instead
        ax2.plot(Y_pos_data, Y0_profile_data, label=plot_str[j], color=colors[j])
        #particles_i = tbt_dict['particles_i']
    else:
        Y_profile_data = tbt_dict['monitorV_y_intensity'][ind] / y_height_max_avg
        ax2.plot(Y_pos_data, Y_profile_data, label=plot_str[j], color=colors[j])

ax2.set_ylabel('Normalized counts')
ax2.set_xlabel('y [m]')
ax2.set_ylim(0, 1.1)
plt.show()