"""
Generate longitudinally binomial particle distribution from 
"""
import fma_ions
import numpy as np
import matplotlib.pyplot as plt

# SPS line and Pb nominal parameters
sps = fma_ions.SPS_sequence_maker()
sps_line, twiss = sps.load_xsuite_line_and_twiss()
max_zeta = 0.74269 # assuming h=4653

# Beam parameters
beamParams = fma_ions.BeamParameters_SPS
num_particles = 1_000_000
m = 3.0 # test a relatively parabolic bunch

particles = fma_ions.generate_binomial_distribution_from_PS_extr(num_particles=num_particles,
                                                                 nemitt_x=beamParams.exn, nemitt_y=beamParams.eyn,
                                                                 sigma_z=beamParams.sigma_z, total_intensity_particles=beamParams.Nb,
                                                                 line=sps_line, m=m)


# Make histograms in all planes to inspect distribution
bin_heights, bin_borders = np.histogram(particles.zeta, bins=300)
bin_widths = np.diff(bin_borders)
bin_centers = bin_borders[:-1] + bin_widths / 2


# Generate the plots
plt.close('all')

fig1 = plt.figure(1, figsize=(6.4, 7))
ax21 = fig1.add_subplot(3,1,1)
ax22 = fig1.add_subplot(3,1,2)
ax23 = fig1.add_subplot(3,1,3)
ax21.plot(particles.x*1000, particles.px, '.', markersize=1)
ax21.set_xlabel(r'x [mm]')
ax21.set_ylabel(r'px [-]')
ax22.plot(particles.y*1000, particles.py, '.', markersize=1)
ax22.set_xlabel(r'y [mm]')
ax22.set_ylabel(r'py [-]')
ax23.plot(particles.zeta, particles.delta*1000, '.', markersize=1)
ax23.axvline(x=max_zeta, color='r', linestyle='dashed', label='Max SPS $\zeta$ for RF bucket')
ax23.axvline(x=-max_zeta, color='r', linestyle='dashed', label=None)
ax23.set_xlabel(r'$\zeta$ [m]')
ax23.set_ylabel(r'$\delta$ [1e-3]')
fig1.subplots_adjust(bottom=.08, top=.93, hspace=.33, left=.18,
                     right=.96, wspace=.33)
                     
fig2, ax2 = plt.subplots(1, 1, figsize = (6,5))
ax2.bar(bin_centers, bin_heights, width=bin_widths)
ax2.set_ylabel('Counts')
ax2.set_xlabel(r'$\zeta$ [m]')
                     
plt.show()