"""
Tester script to generate SPS sequence with aperture, remove unnecessarily large aperture at unknown element
"""
import fma_ions
import numpy as np
import matplotlib.pyplot as plt


# Generate SPS sequence maker, make sure aperture exists 
sps = fma_ions.SPS_sequence_maker()
line = sps.generate_xsuite_seq(add_aperture=True)
x_ap, y_ap, a = sps.print_smallest_aperture(line)