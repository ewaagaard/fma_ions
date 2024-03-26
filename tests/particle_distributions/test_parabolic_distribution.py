"""
Test script of parabolic particle distribution
"""
import numpy as np
import fma_ions
import xpart as xp
import xobjects as xo
import xtrack as xt
import xfields as xf
import matplotlib.pyplot as plt
import xplt

# Initial parameters - 
num_particles = 5000
sigma_z_RMS = 0.284
sigma_z0 = 0.225

# Import SPS line
sps = fma_ions.SPS_sequence_maker()
line, twiss = sps.load_xsuite_line_and_twiss()

######## Parabolic distribution ########
particles = fma_ions.generate_parabolic_distribution(num_particles, 
                                                      fma_ions.BeamParameters_SPS.exn, 
                                                      fma_ions.BeamParameters_SPS.eyn, 
                                                      fma_ions.BeamParameters_SPS.Nb, 
                                                      sigma_z_RMS, 
                                                      line)

######## Gaussian distribution ########
particles2 = xp.generate_matched_gaussian_bunch(
        num_particles=num_particles, total_intensity_particles=fma_ions.BeamParameters_SPS.Nb,
        nemitt_x=fma_ions.BeamParameters_SPS.exn, nemitt_y=fma_ions.BeamParameters_SPS.eyn, 
        sigma_z = sigma_z0,
        particle_ref=line.particle_ref, line=line)

# Make a histogram - parabolic
bin_heights, bin_borders = np.histogram(particles.zeta, bins=300)
bin_widths = np.diff(bin_borders)
bin_centers = bin_borders[:-1] + bin_widths / 2

# Make a histogram - Gaussian
bin_heights2, bin_borders2 = np.histogram(particles2.zeta, bins=300)
bin_widths2 = np.diff(bin_borders2)
bin_centers2 = bin_borders2[:-1] + bin_widths2 / 2


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
ax23.set_xlabel(r'z [-]')
ax23.set_ylabel(r'$\delta$ [1e-3]')
fig1.subplots_adjust(bottom=.08, top=.93, hspace=.33, left=.18,
                     right=.96, wspace=.33)
                     
fig2, ax2 = plt.subplots(1, 1, figsize = (6,5))
ax2.bar(bin_centers, bin_heights, width=bin_widths, label='Parabolic')
ax2.bar(bin_centers2, bin_heights2, width=bin_widths2, label='Gaussian', alpha=0.7)
ax2.set_ylabel('Counts')
ax2.set_xlabel(r'z [-]')
ax2.legend(fontsize=10)
plt.tight_layout()
plt.show()