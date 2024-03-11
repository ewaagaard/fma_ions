"""
Remove aperture elements smaller than a certain size for SPS
"""
import fma_ions

# Import SPS line
sps = fma_ions.SPS_sequence_maker()

# Finally test the function
print('\nTesting removal of aperture smaller than 0.05 m!')
line, twiss = sps.load_xsuite_line_and_twiss()
line = sps.remove_aperture_below_threshold(line, 0.05)