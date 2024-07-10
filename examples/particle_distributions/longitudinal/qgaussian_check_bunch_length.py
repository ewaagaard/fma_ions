"""
Find stable bunch length of generated binomial distribution
"""
import fma_ions
import numpy as np
import matplotlib.pyplot as plt

# Beam parameters
n_turns = 300 
num_part = 10_000

# Initiate fit function
fits = fma_ions.Fit_Functions()

# First generate qGaussian particles with default parameters, and more macroparticles for better plotting
sps_seq = fma_ions.SPS_sequence_maker()
line, twiss = sps_seq.load_xsuite_line_and_twiss()
sps0 = fma_ions.SPS_Flat_Bottom_Tracker(num_part=1_000_000)
particles = sps0.generate_particles(line=line, distribution_type='qgaussian', beamParams=fma_ions.BeamParameters_SPS_Binomial_2016())

# Make a histogram
bin_heights, bin_borders = np.histogram(particles.zeta, bins=160)
bin_widths = np.diff(bin_borders)
bin_centers = bin_borders[:-1] + bin_widths / 2
bin_heights = bin_heights / np.max(bin_heights) #normalize height

popt_Q, pcov_Q = fits.fit_Q_Gaussian(bin_centers, bin_heights, q0=0.59)
q = popt_Q[1]
q_error = np.sqrt(np.diag(pcov_Q))[1] # error from covarance_matrix
sigma_q_gaussian = fits.get_sigma_RMS_from_qGaussian_fit(popt_Q)
print('Initial particle q-Gaussian fit q={:.3f} +/- {:.2f}, sigma_RMS = {:.3f} m'.format(q, q_error, sigma_q_gaussian))

fig2, ax2 = plt.subplots(1, 1, figsize = (6,5))
ax2.bar(bin_centers, bin_heights, width=bin_widths, label='Generated particles')
ax2.plot(bin_centers, fits.Q_Gaussian(bin_centers, *popt_Q), color='lime', lw=2.8, 
         label='Q-Gaussian fit\n q={:.3f}, sigma_RMS = {:.3f} m'.format(q, sigma_q_gaussian))  
ax2.set_ylabel('Counts')
ax2.set_xlabel(r'$\zeta$ [m]')
ax2.legend()
fig2.tight_layout()

# Then track for a few hundred turns, see what happens
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns, num_part=num_part, turn_print_interval=5)
tbt = sps.track_SPS(which_context='cpu', install_SC_on_line=False, distribution_type='qgaussian')
tbt_dict = tbt.to_dict(convert_to_numpy=True)

# Load injection data
sps_plot = fma_ions.SPS_Plotting()
_, _, sigma_RMS_Gaussian_in_m, sigma_RMS_Binomial_in_m, ctime = sps_plot.load_bunch_length_data()
sps_plot.plot_longitudinal_monitor_data(tbt_dict=tbt_dict)

# Fit q-Gaussian profiles - Find total number of stacked profiles and turns per profiles
n_profiles = len(tbt_dict['z_bin_heights'][0]) 
nturns_per_profile = tbt_dict['nturns_profile_accumulation_interval']
sigmas = np.zeros(n_profiles)
sigmas_binomial = np.zeros(n_profiles)
sigmas_q_gaussian = np.zeros(n_profiles)
m = np.zeros(n_profiles)
m_error = np.zeros(n_profiles)
q_vals = np.zeros(n_profiles)
q_errors = np.zeros(n_profiles)

# Create time array with
turns_per_s = tbt_dict['Turns'][-1] / tbt_dict['Seconds'][-1]
turn_array = np.arange(0, tbt_dict['Turns'][-1], step=nturns_per_profile)
time_array = turn_array.copy() / turns_per_s

# Fit binomials to 
for i in range(n_profiles):

    z_bin_heights_sorted = np.array(sorted(tbt_dict['z_bin_heights'][:, i], reverse=True))
    z_height_max_avg = np.mean(z_bin_heights_sorted[:5]) # take average of top 5 values
    xdata, ydata = tbt_dict['z_bin_centers'], tbt_dict['z_bin_heights'][:, i] / z_height_max_avg
    
    # Fit both q-Gaussian and binomial
    popt_Q, pcov_Q = fits.fit_Q_Gaussian(xdata, ydata)
    q_vals[i] = popt_Q[1]
    q_errors[i] = np.sqrt(np.diag(pcov_Q))[1] # error from covarance_matrix
    sigmas_q_gaussian[i] = fits.get_sigma_RMS_from_qGaussian_fit(popt_Q)
    print('Profile {}: q-Gaussian fit q={:.3f} +/- {:.2f}, sigma_RMS = {:.3f} m'.format(i, q_vals[i], q_errors[i], 
                                                                                                 sigmas_q_gaussian[i]))
    popt_B, pcov_B = fits.fit_Binomial(xdata, ydata)
    sigmas_binomial[i], sigmas_error = fits.get_sigma_RMS_from_binomial_fit(popt_B, pcov_B)
    m[i] = popt_B[1]
    m_error[i] = np.sqrt(np.diag(pcov_B))[1]
    print('Profile {}: binomial fit m={:.3f} +/- {:.2f}, sigma_RMS = {:.3f} +/- {:.2f}\n'.format(i, m[i], m_error[i], 
                                                                                                 sigmas_binomial[i], sigmas_error))

# Compare at which bunch length the profile stabilizes
time_units = tbt_dict['Seconds']
f3, ax22 = plt.subplots(1, 1, figsize = (8,6))
#ax22.plot(time_units, tbt_dict['bunch_length'], color='turquoise', alpha=0.7, lw=1.5, label='Simulated')
ax22.plot(time_array, sigmas_binomial, color='cyan', ls='--', alpha=0.95, label='Simulated profiles')
ax22.plot(ctime, sigma_RMS_Binomial_in_m, color='darkorange', label='Measured profiles')
ax22.set_ylabel(r'$\sigma_{z, RMS}$ [m] fitted binomial')
ax22.set_xlabel('Time [s]')
#ax22.set_xlim(time_array[0] * 0.8, time_array[-1] * 1.2)
ax22.legend()

# Insert extra box with fitted q-value of profiles
ax23 = ax22.inset_axes([0.7,0.5,0.25,0.25])
ax23.errorbar(time_array, q_vals, yerr=q_errors, color='green', alpha=0.55, markerfacecolor='lime', 
              ls='None', marker='o', ms=5.1, label='Fitted q of simulated profiles')
ax23.set_ylabel('Fitted $q$-value', fontsize=13.5, color='green')
ax23.tick_params(axis="both", labelsize=12)
ax23.tick_params(colors='green', axis='y')
ax23.set_ylim(min(q_vals)-0.2, max(q_vals)+0.2)
ax23.set_xlabel('Time [s]', fontsize=13.5)
f3.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# Then compare longitudinal profile monitor with initially generated particle - how much does distribution change?
fig4, ax4 = plt.subplots(1, 1, figsize = (8,6))
ax4.bar(bin_centers, bin_heights, width=bin_widths, label='Generated particles\nturn 0')
ax4.plot(xdata, ydata, color='r', label='Zeta profile\nturn {}'.format(nturns_per_profile * (i+1)))
ax4.set_ylabel('Counts')
ax4.set_xlabel(r'$\zeta$ [m]')
ax4.legend()
fig4.tight_layout()

plt.show()
