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
context = xo.ContextCpu(omp_num_threads='auto')

# Test default tracking with space charge on GPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_part=50_000, num_turns=50, turn_print_interval=10)
particles = sps.generate_particles(line, context, distribution_type='qgaussian', matched_for_PS_extraction=True)
particles2 = sps.generate_particles(line, context, distribution_type='qgaussian', matched_for_PS_extraction=False)

# Make histograms of particles
widths, centers, heights = [], [], []
widths2, centers2, heights2 = [], [], []

bin_heights, bin_borders = np.histogram(particles.zeta, bins=70)
bin_widths = np.diff(bin_borders)
bin_centers = bin_borders[:-1] + bin_widths / 2
bin_heights = bin_heights/max(bin_heights)

bin_heights2, bin_borders2 = np.histogram(particles2.zeta, bins=70)
bin_widths2 = np.diff(bin_borders2)
bin_centers2 = bin_borders2[:-1] + bin_widths2 / 2
bin_heights2 = bin_heights2/max(bin_heights2)

widths.append(bin_widths)
centers.append(bin_centers)
heights.append(bin_heights)
widths2.append(bin_widths2)
centers2.append(bin_centers2)
heights2.append(bin_heights2)

# Test tracking 5 turns each, see what happens
print('\nBunch length before spill: {:.3f}'.format(np.std(particles.zeta[particles.state > 0])))
print('Bunch length after spill: {:.3f}'.format(np.std(particles2.zeta[particles2.state > 0])))

# Track particles
for turn in range(1, 50):            
    print('\nTracking turn {}'.format(turn))       
    line.track(particles)
    line.track(particles2)
    
    print('Bunch length before spill: {:.3f}'.format(np.std(particles.zeta[particles.state > 0])))
    print('Bunch length after spill: {:.3f}'.format(np.std(particles2.zeta[particles2.state > 0])))

# Make 2nd batch of histograms of particles, after tracking
bin_heights, bin_borders = np.histogram(particles.zeta, bins=70)
bin_widths = np.diff(bin_borders)
bin_centers = bin_borders[:-1] + bin_widths / 2
bin_heights = bin_heights/max(bin_heights)

bin_heights2, bin_borders2 = np.histogram(particles2.zeta, bins=70)
bin_widths2 = np.diff(bin_borders2)
bin_centers2 = bin_borders2[:-1] + bin_widths2 / 2
bin_heights2 = bin_heights2/max(bin_heights2)

# Append new values
widths.append(bin_widths)
centers.append(bin_centers)
heights.append(bin_heights)
widths2.append(bin_widths2)
centers2.append(bin_centers2)
heights2.append(bin_heights2)
string = ['', 'after_tracking']

# Load injection data
sps_plot = fma_ions.SPS_Plotting()
zeta_SPS_inj, zeta_SPS_final, zeta_PS_BSM, data_SPS_inj, data_SPS_final, data_PS_BSM = sps_plot.load_longitudinal_profile_data()
zeta_SPS_inj_after_RF_spill, data_SPS_inj_after_RF_spill = sps_plot.load_longitudinal_profile_after_SPS_injection_RF_spill()

for i in range(2):

    # Plot longitudinal phase space, initial and final state
    fig = plt.figure(figsize = (8, 7.5))
    gs = fig.add_gridspec(2, hspace=0, height_ratios= [1, 1])
    ax = gs.subplots(sharex=True, sharey=False)
    
    ax[0].bar(centers[i], heights[i], width=widths[i], color='cyan', label='Simulated particles')
    ax[0].plot(zeta_SPS_inj, data_SPS_inj, color='blue', marker='v', alpha=0.7,
               ms=5.8, linestyle='None', label='SPS WCM\n2016 data')  
    
    ax[1].bar(centers2[i], heights2[i], width=widths2[i], color='cyan', label='Simulated particles')
    ax[1].plot(zeta_SPS_inj_after_RF_spill, data_SPS_inj_after_RF_spill, color='blue', alpha=0.8,
               marker='v', ms=5.8, linestyle='None', label='SPS WCM\n2016 data') 
    
    ax[0].legend(loc='upper left', fontsize=12)
    ax[1].legend(loc='upper left', fontsize=12)
    
    xlim = 0.95 
    ax[0].set_xlim(-xlim, xlim)
    ax[1].set_xlim(-xlim, xlim)
    ax[1].set_xlabel(r'$\zeta$ [m]')
    ax[0].set_ylabel('Amplitude [a.u.]')
    ax[1].set_ylabel('Amplitude [a.u.]')
    
    ax[0].text(0.6, 0.91, 'At injection, before RF capture', fontsize=13, transform=ax[0].transAxes)
    ax[1].text(0.6, 0.91, 'At injection, after RF capture', fontsize=13, transform=ax[1].transAxes)
    fig.tight_layout()
    fig.savefig('SPS_simulated_particles_vs_before_and_after_RF_spill{}.png'.format(string[i]), dpi=250)
    plt.show()
    del fig
