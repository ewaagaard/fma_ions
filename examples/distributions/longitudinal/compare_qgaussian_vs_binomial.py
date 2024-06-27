"""
Script to compare longitudinal Q-gaussian distribution vs binomial, to ensure correct frozen SC distribution
"""
import fma_ions
import numpy as np
from xfields import LongitudinalProfileQGaussian
import matplotlib.pyplot as plt
import scipy.constants as constants

also_plot_long_profile = False
x_axis_in_time_units = True
gamma = 7.33
beta = np.sqrt(1 - 1/gamma**2)

# Fit binomial and Q-Gaussian to data
fits = fma_ions.Fit_Functions()

# Load injection data
sps_plot = fma_ions.SPS_Plotting()
zeta_SPS_inj, zeta_SPS_final, zeta_PS_BSM, data_SPS_inj, data_SPS_final, data_PS_BSM = sps_plot.load_longitudinal_profile_data()
zeta_SPS_inj_after_RF_spill, data_SPS_inj_after_RF_spill = sps_plot.load_longitudinal_profile_after_SPS_injection_RF_spill()

# First cut data approprietly, to avoid the artificial ringing
ind_cut_1 = np.where((zeta_SPS_inj > -.85) & (zeta_SPS_inj < .37))
zeta_SPS_inj_cut = zeta_SPS_inj[ind_cut_1]
data_SPS_inj_cut = data_SPS_inj[ind_cut_1]

ind_cut_2 = np.where((zeta_SPS_inj_after_RF_spill > -0.6) & (zeta_SPS_inj_after_RF_spill < 0.35))
#ind_cut_2 = np.where((zeta_SPS_inj_after_RF_spill > -0.9) & (zeta_SPS_inj_after_RF_spill < 0.9))
zeta_SPS_inj_after_RF_spill_cut = zeta_SPS_inj_after_RF_spill[ind_cut_2]
data_SPS_inj_after_RF_spill_cut = data_SPS_inj_after_RF_spill[ind_cut_2]

# Also get coordinates with cut
ind_cut_2_plot = np.where((zeta_SPS_inj_after_RF_spill > -0.6) & (zeta_SPS_inj_after_RF_spill < 0.35))
zeta_SPS_inj_after_RF_spill_cut_plot = zeta_SPS_inj_after_RF_spill[ind_cut_2_plot]
data_SPS_inj_after_RF_spill_cut_plot = data_SPS_inj_after_RF_spill[ind_cut_2_plot]

# Fit binomial and q-Gaussian to before and after RF spill at injection
popt_Q_before_spill, pcov_Q_before_spill = fits.fit_Q_Gaussian(zeta_SPS_inj_cut, data_SPS_inj_cut)
sigma_RMS_Q_before_spill = fits.get_sigma_RMS_from_qGaussian_fit(popt_Q_before_spill)
popt_B_before_spill, pcov_B_before_spill = fits.fit_Binomial(zeta_SPS_inj_cut, data_SPS_inj_cut)
sigma_RMS_B_before_spill, error_sigma_RMS_B_before_spill = fits.get_sigma_RMS_from_binomial_fit(popt_B_before_spill, pcov_B_before_spill)
print('\nBefore RF spill')
print('Q-Gaussian: q={:.3f} +/- {:.3f}, sigma_RMS = {:.3f}'.format(popt_Q_before_spill[1], np.sqrt(np.diag(pcov_Q_before_spill))[1], 
                                                                   sigma_RMS_Q_before_spill))
print('Binomial: m={:.3f}, sigma_RMS = {:.3f}\n'.format(popt_B_before_spill[1], sigma_RMS_B_before_spill))

popt_G_after_spill, pcov_G_after_spill = fits.fit_Gaussian(zeta_SPS_inj_after_RF_spill_cut, data_SPS_inj_after_RF_spill_cut)
popt_Q_after_spill, pcov_Q_after_spill = fits.fit_Q_Gaussian(zeta_SPS_inj_after_RF_spill_cut, data_SPS_inj_after_RF_spill_cut)
sigma_RMS_Q_after_spill = fits.get_sigma_RMS_from_qGaussian_fit(popt_Q_after_spill)
popt_B_after_spill, pcov_B_after_spill = fits.fit_Binomial(zeta_SPS_inj_after_RF_spill_cut, data_SPS_inj_after_RF_spill_cut)
sigma_RMS_B_after_spill, error_sigma_RMS_B_after_spill = fits.get_sigma_RMS_from_binomial_fit(popt_B_after_spill, pcov_B_after_spill)
print('\nAfter RF spill')
print('Gaussian: sigma_RMS = {:.3f} +/- {:.3f} m'.format(popt_G_after_spill[2],  np.sqrt(np.diag(pcov_G_after_spill))[2]))
print('Q-Gaussian: q={:.3f}, sigma_RMS = {:.3f} m'.format(popt_Q_after_spill[1], sigma_RMS_Q_after_spill))
print('Binomial: m={:.3f}, sigma_RMS = {:.3f} m\n'.format(popt_B_after_spill[1], sigma_RMS_B_after_spill))


# Generate Q-Gaussian longitudinal profile in xfields
z0 = 0.0
npart = int(1e4)
z = np.linspace(-.75, .75, npart)

lprofile0 = LongitudinalProfileQGaussian(
        number_of_particles=npart,
        sigma_z=sigma_RMS_Q_after_spill,
        z0=z0,
        q_parameter=popt_Q_after_spill[1])
lden0 = lprofile0.line_density(z)
lden0_normalized = lden0 / max(lden0)


# Plot longitudinal phase space, initial and final state
fig = plt.figure(figsize = (8, 7.5))
gs = fig.add_gridspec(2, hspace=0, height_ratios= [1, 1])
ax = gs.subplots(sharex=True, sharey=False)

# Convert length units to time in ns if desired
unit_factor = 1e-9 * constants.c * beta if x_axis_in_time_units else 1.

ax[0].plot(zeta_SPS_inj / unit_factor, data_SPS_inj, color='blue', marker='v', ms=5.8, linestyle='None', label='SPS WCM\n2016 data')  
ax[0].plot(zeta_PS_BSM / unit_factor, data_PS_BSM, color='k', markerfacecolor='gold', marker='*', ms=14.8, linestyle='None', alpha=0.8, label='PS BSM data \n2023, at extr.')
ax[0].plot(zeta_SPS_inj_cut / unit_factor, fits.Q_Gaussian(zeta_SPS_inj_cut, *popt_Q_before_spill), color='lime', lw=2.8, label='Q-Gaussian fit')  
ax[0].plot(zeta_SPS_inj_cut /unit_factor, fits.Binomial(zeta_SPS_inj_cut, *popt_B_before_spill), color='red', ls='--', lw=2.8, label='Binomial fit')  

ax[1].plot(zeta_SPS_inj_after_RF_spill / unit_factor, data_SPS_inj_after_RF_spill, color='blue', marker='v', ms=5.8, linestyle='None', label='SPS WCM\n2016 data')  
ax[1].plot(zeta_SPS_inj_after_RF_spill_cut_plot / unit_factor, fits.Q_Gaussian(zeta_SPS_inj_after_RF_spill_cut_plot, *popt_Q_after_spill), color='lime', lw=2.8, label='Q-Gaussian fit')  
ax[1].plot(zeta_SPS_inj_after_RF_spill_cut_plot / unit_factor, fits.Binomial(zeta_SPS_inj_after_RF_spill_cut_plot, *popt_B_after_spill), color='red', ls='--', lw=2.8, label='Binomial fit')  
#ax[1].plot(zeta_SPS_inj_after_RF_spill_cut_plot / unit_factor, fits.Gaussian(zeta_SPS_inj_after_RF_spill_cut_plot, *popt_G_after_spill), color='orange', alpha=0.75, lw=2.8, label='Gaussian fit')  
if also_plot_long_profile:
    ax[1].plot(z / unit_factor, lden0_normalized, color='orange', alpha=0.6, label='xfields Q-Gaussian q={:.1f}'.format(popt_Q_after_spill[1]))

ax[0].legend(loc='upper left', fontsize=12)
ax[1].legend(loc='upper left', fontsize=12)
xlim = 3. if x_axis_in_time_units else 0.95 
ax[0].set_xlim(-xlim, xlim)
ax[1].set_xlim(-xlim, xlim)
ax[1].set_xlabel('Time [ns]' if x_axis_in_time_units else r'$\zeta$ [m]')
ax[0].set_ylabel('Amplitude [a.u.]')
ax[1].set_ylabel('Amplitude [a.u.]')

ax[0].text(0.6, 0.91, 'At injection, before RF spill', fontsize=13, transform=ax[0].transAxes)
ax[1].text(0.6, 0.91, 'At injection, after RF spill', fontsize=13, transform=ax[1].transAxes)
fig.tight_layout()
fig.savefig('SPS_longitudinal_profile_before_vs_after_RF_spill.png', dpi=250)
plt.show()

