"""
Generate xpart binomial particle distribution and compare with data
"""
import fma_ions
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 19,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 15,
        "figure.titlesize": 20,
    }
)


# Load data
sps_plot = fma_ions.SPS_Plotting()
zeta_SPS_inj, zeta_SPS_final, zeta_PS_BSM, data_SPS_inj, data_SPS_final, data_PS_BSM = sps_plot.load_longitudinal_profile_data()

# SPS line and Pb nominal parameters
sps = fma_ions.SPS_sequence_maker()
line, twiss = sps.load_xsuite_line_and_twiss()
max_zeta = 0.74269 # assuming h=4653

# Beam parameters
beamParams = fma_ions.BeamParameters_SPS_Binomial_2016_before_RF_capture()
longitudinal_distribution_type = 'qgaussian'
num_part = 80_000#_000

# Generate particles and SPS separatrix coordinates
particles  = fma_ions.generate_particles_transverse_gaussian(beamParams, line, longitudinal_distribution_type, num_part,
                                                             matched_for_PS_extraction=True)
zeta_separatrix, delta_separatrix  = fma_ions.return_separatrix_coordinates(beamParams, line, longitudinal_distribution_type)


# Make histograms in all planes to inspect distribution
bin_heights, bin_borders = np.histogram(particles.zeta, bins=60)
bin_widths = np.diff(bin_borders)
bin_centers = bin_borders[:-1] + bin_widths / 2
bin_heights = bin_heights/np.max(bin_heights) # normalize bin heights


# Plot particle distribution vs data
fig, ax = plt.subplots(1, 1, figsize=(8,6))
ax.plot(zeta_PS_BSM, data_PS_BSM, '*', ms=9, label='Data')
ax.bar(bin_centers, bin_heights, width=bin_widths, alpha=0.8, color='cyan', label='Particles')
ax.axvline(x=max_zeta, color='r', linestyle='dashed', label='Max SPS $\zeta$ for RF bucket')
ax.axvline(x=-max_zeta, color='r', linestyle='dashed', label=None)
ax.set_ylabel('Amplitude [a.u.]')
ax.set_xlabel('$\zeta$ [m]')
ax.set_xlim(-max_zeta-0.15, max_zeta+0.15)
ax.legend(fontsize=13.5)
plt.tight_layout()


# Plot longitudinal phase space
fig2, ax2 = plt.subplots(1, 1, figsize=(8,6))
ax2.plot(particles.zeta, particles.delta*1000, '.', markersize=1)
ax2.axvline(x=max_zeta, color='r', linestyle='dashed', label='Max SPS $\zeta$ for RF bucket')
ax2.axvline(x=-max_zeta, color='r', linestyle='dashed', label=None)
ax2.set_xlabel(r'$\zeta$ [m]')
ax2.set_ylabel(r'$\delta$ [1e-3]')
plt.tight_layout()

# Plot particle distribution and longitudinal phase space together 
#fig3, ax3 = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
fig3 = plt.figure(figsize = (6, 5.5))
gs = fig3.add_gridspec(2, hspace=0, height_ratios= [1, .8])
ax3 = gs.subplots(sharex=True, sharey=False)

lns1 = ax3[0].plot(zeta_PS_BSM, data_PS_BSM, '*', ms=9, label='PS BSM Data')
lns2 = ax3[0].bar(bin_centers, bin_heights, width=bin_widths, alpha=0.8, color='cyan', label='Reconstructed particles')
ax3[0].axvline(x=max_zeta, color='r', linestyle='dashed', label=None) #'Max $\zeta$ for SPS RF bucket')
ax3[0].axvline(x=-max_zeta, color='r', linestyle='dashed', label=None)
ax3[0].set_ylabel('Amplitude [a.u.]')

# ax3[1].plot(particles.zeta, particles.delta*1000, '.', color='cyan', markersize=1)

# Plot particles sorted by density
x, y = particles.zeta, particles.delta*1000
xy = np.vstack([x,y]) # Calculate the point density
z = gaussian_kde(xy)(xy)
idx = z.argsort()  # Sort the points by density, so that the densest points are plotted last
x, y, z = x[idx], y[idx], z[idx]

ax3[1].scatter(x, y, c=z, cmap='cool', s=1)
lns3 = ax3[1].plot(zeta_separatrix, delta_separatrix * 1e3, '-', color='red', linewidth=3, label='SPS RF separatrix')
ax3[1].plot(zeta_separatrix, -delta_separatrix * 1e3, '-', color='red', linewidth=3, label=None)
ax3[1].axvline(x=max_zeta, color='r', linestyle='dashed', label='Max SPS $\zeta$ for RF bucket')
ax3[1].axvline(x=-max_zeta, color='r', linestyle='dashed', label=None)
ax3[1].set_xlabel(r'$\zeta$ [m]')
ax3[1].set_ylabel(r'$\delta$ [1e-3]')
lns = [lns1[0], lns2, lns3[0]]
labs = [l.get_label() for l in lns]
ax3[0].legend(lns, labs, fontsize=12.5, loc=6)
ax3[0].set_xlim(-max_zeta-0.15, max_zeta+0.15)
plt.tight_layout()
fig3.savefig('PS_extraction_reconstructed_particles_vs_SPS_separatrix.png', dpi=250)

plt.show()
