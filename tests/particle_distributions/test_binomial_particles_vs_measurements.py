"""
Generate binomial particle distribution vs measured profiles at PS extraction and SPS injection
"""
import numpy as np
import matplotlib.pyplot as plt
import fma_ions
import xobjects as xo

# Generate SPS sequence
sps_maker = fma_ions.SPS_sequence_maker()
line, twiss = sps_maker.load_xsuite_line_and_twiss()
context = xo.ContextCpu()

# Test default tracking with space charge on GPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_part=1_000_000, num_turns=100, turn_print_interval=10)
particles = sps.generate_particles(line, context, distribution_type='binomial', use_binomial_dist_after_RF_spill=False)
particles2 = sps.generate_particles(line, context, distribution_type='binomial', use_binomial_dist_after_RF_spill=True)

# Make histograms of particles
bin_heights, bin_borders = np.histogram(particles.zeta, bins=250)
bin_widths = np.diff(bin_borders)
bin_centers = bin_borders[:-1] + bin_widths / 2
bin_heights = bin_heights/max(bin_heights)

bin_heights2, bin_borders2 = np.histogram(particles2.zeta, bins=250)
bin_widths2 = np.diff(bin_borders2)
bin_centers2 = bin_borders2[:-1] + bin_widths2 / 2
bin_heights2 = bin_heights2/max(bin_heights2)


# Load injection data
sps_plot = fma_ions.SPS_Plotting()
zeta_SPS_inj, zeta_SPS_final, zeta_PS_BSM, data_SPS_inj, data_SPS_final, data_PS_BSM = sps_plot.load_longitudinal_profile_data()
zeta_SPS_inj_after_RF_spill, data_SPS_inj_after_RF_spill = sps_plot.load_longitudinal_profile_after_SPS_injection_RF_spill()

# Plot longitudinal phase space, initial and final state
fig = plt.figure(figsize = (8, 7.5))
gs = fig.add_gridspec(2, hspace=0, height_ratios= [1, 1])
ax = gs.subplots(sharex=True, sharey=False)

ax[0].bar(bin_centers, bin_heights, width=bin_widths, color='blue', label='Simulated particles')
ax[0].plot(zeta_SPS_inj, data_SPS_inj, color='orange', marker='v', alpha=0.7,
           ms=5.8, linestyle='None', label='SPS WCM\n2016 data')  

ax[1].bar(bin_centers2, bin_heights2, width=bin_widths2, color='blue', label='Simulated particles')
ax[1].plot(zeta_SPS_inj_after_RF_spill, data_SPS_inj_after_RF_spill, color='orange', alpha=0.8,
           marker='v', ms=5.8, linestyle='None', label='SPS WCM\n2016 data') 

ax[0].legend(loc='upper left', fontsize=12)
ax[1].legend(loc='upper left', fontsize=12)

xlim = 0.95 
ax[0].set_xlim(-xlim, xlim)
ax[1].set_xlim(-xlim, xlim)
ax[1].set_xlabel(r'$\zeta$ [m]')
ax[0].set_ylabel('Amplitude [a.u.]')
ax[1].set_ylabel('Amplitude [a.u.]')

ax[0].text(0.6, 0.91, 'At injection, before RF spill', fontsize=13, transform=ax[0].transAxes)
ax[1].text(0.6, 0.91, 'At injection, after RF spill', fontsize=13, transform=ax[1].transAxes)
fig.tight_layout()
fig.savefig('SPS_simulated_particles_vs_before_and_after_RF_spill.png', dpi=250)
plt.show()
