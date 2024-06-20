"""
Find stable bunch length of generated binomial distribution
"""
import fma_ions
import matplotlib.pyplot as plt

n_turns = 100 
num_part = 10_000

# Run binomial distribution for 100 turns, check
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns, num_part=num_part, turn_print_interval=5)
tbt = sps.track_SPS(which_context='cpu', install_SC_on_line=False, distribution_type='binomial')
tbt_dict = tbt.to_dict(convert_to_numpy=True)

# Load injection data
sps_plot = fma_ions.SPS_Plotting()
_, _, sigma_RMS_Gaussian_in_m, sigma_RMS_Binomial_in_m, ctime = sps_plot.load_bunch_length_data()

# Compare at which bunch length the profile stabilizes
time_units = tbt_dict['Seconds']
f3, ax22 = plt.subplots(1, 1, figsize = (8,6))
ax22.plot(time_units, tbt_dict['bunch_length'], color='turquoise', alpha=0.7, lw=1.5, label='Simulated')
ax22.plot(ctime, sigma_RMS_Binomial_in_m, color='darkorange', label='Measured RMS Binomial')
ax22.set_ylabel(r'$\sigma_{z}$ [m]')
ax22.set_xlabel('Time [s]')
ax22.legend()
#ax22.set_xlim(-0.05 * max(tbt_dict['Seconds']), 1.1 * max(tbt_dict['Seconds']))
f3.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()