"""
Example script to track an SPS Pb beam in the Q26 lattice with higher intensity, and observe where particles are lost
"""
import matplotlib.pyplot as plt
import fma_ions

# Plot available aperture 
sps_plot = fma_ions.SPS_Plotting()
sps_plot.plot_beam_envelope_and_aperture()