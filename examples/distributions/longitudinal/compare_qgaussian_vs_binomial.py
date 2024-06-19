"""
Script to compare longitudinal Q-gaussian distribution vs binomial, to ensure correct frozen SC distribution
"""
import fma_ions
import numpy as np
from xfields import LongitudinalProfileQGaussian
from xpart.longitudinal import generate_longitudinal_coordinates
import matplotlib.pyplot as plt
from scipy.special import gamma as Gamma


# Fit binomial and Q-Gaussian to data
fits = fma_ions.Fit_Functions()

# Load injection data
sps_plot = fma_ions.SPS_Plotting()
zeta_SPS_inj, zeta_SPS_final, zeta_PS_BSM, data_SPS_inj, data_SPS_final, data_PS_BSM = sps_plot.load_longitudinal_profile_data()
zeta_SPS_inj_after_RF_spill, data_SPS_inj_after_RF_spill = sps_plot.load_longitudinal_profile_after_SPS_injection_RF_spill()

# First cut data approprietly, to avoid the artificial ringing
ind_cut_1 = np.where((zeta_SPS_inj > -0.75) & (zeta_SPS_inj < 0.35))
zeta_SPS_inj_cut = zeta_SPS_inj[ind_cut_1]
data_SPS_inj_cut = data_SPS_inj[ind_cut_1]

ind_cut_2 = np.where((zeta_SPS_inj_after_RF_spill > -0.65) & (zeta_SPS_inj_after_RF_spill < 0.3))
zeta_SPS_inj_after_RF_spill_cut = zeta_SPS_inj_after_RF_spill[ind_cut_2]
data_SPS_inj_after_RF_spill_cut = data_SPS_inj_after_RF_spill[ind_cut_2]

# Fit binomial and q-Gaussian to before and after RF spill at injection
popt_Q_before_spill = fits.fit_Q_Gaussian(zeta_SPS_inj_cut, data_SPS_inj_cut)
popt_B_before_spill = fits.fit_Binomial(zeta_SPS_inj_cut, data_SPS_inj_cut)

popt_Q_after_spill = fits.fit_Q_Gaussian(zeta_SPS_inj_after_RF_spill_cut, data_SPS_inj_after_RF_spill_cut)
popt_B_after_spill = fits.fit_Binomial(zeta_SPS_inj_after_RF_spill_cut, data_SPS_inj_after_RF_spill_cut)

# Plot longitudinal phase space, initial and final state
fig, ax = plt.subplots(2, 1, figsize = (8, 10), sharex=True)
ax[0].plot(zeta_SPS_inj, data_SPS_inj, label='SPS wall current\nmonitor data\nbefore RF spill')  
ax[0].plot(zeta_PS_BSM, data_PS_BSM, label='PS BSM data \nat extraction')
ax[0].plot(zeta_SPS_inj_cut, fits.Q_Gaussian(data_SPS_inj_cut, *popt_Q_before_spill), label='Q-Gaussian fit')  
ax[0].plot(zeta_SPS_inj_cut, fits.Binomial(data_SPS_inj_cut, *popt_B_before_spill), label='Binomial fit')  
ax[1].plot(zeta_SPS_inj_after_RF_spill, data_SPS_inj_after_RF_spill, label='SPS wall current\nmonitor data\nafter RF spill')  
ax[0].legend(loc='upper right', fontsize=13)
ax[1].legend(loc='upper right', fontsize=13)

# Adjust axis limits and plot turn
ax[0].set_xlim(-0.85, 0.85)
ax[1].set_xlim(-0.85, 0.85)
ax[1].set_xlabel(r'$\zeta$ [m]')
ax[0].set_ylabel('Normalized count')
ax[1].set_ylabel('Normalized count')
plt.show()

'''


# Longitudinal example parameters
z0 = 0.0
sigma_z = 0.225
npart = int(1e6)
m = 5.3
z = np.linspace(-.75, .75, npart)

# Generate Gaussian profile
lprofile0 = LongitudinalProfileQGaussian(
        number_of_particles=npart,
        sigma_z=sigma_z,
        z0=z0,
        q_parameter=1.0)
lden0 = lprofile0.line_density(z)
lden0_normalized = lden0 / max(lden0)

# Generate longitudinal Q-Gaussian
qq = 0.8
lprofile = LongitudinalProfileQGaussian(
        number_of_particles=npart,
        sigma_z=sigma_z,
        z0=z0,
        q_parameter=qq)
lden = lprofile.line_density(z)
lden_normalized = lden / max(lden)

# Parameters for binomial
A = 1.0
x_max = 0.75
z_binomial = np.linspace(-x_max, x_max, npart)
x0 = 0.0
args = (A, m, x_max, x0)

# Generate figure of longitudinal data
fig, ax = plt.subplots(1, 1, figsize=(8,6))
ax.plot(z, lden0_normalized, label='Gaussian')
ax.plot(z, lden_normalized, label='Q-Gaussian q={:.1f}'.format(qq))
ax.plot(z_binomial, Binomial(z_binomial, *args), ls='--', color='lime', lw=2, label='Binomial m={:.1f}'.format(m))

ax.set_xlabel('Zeta [m]')
ax.legend(loc='lower right')
plt.show()
'''