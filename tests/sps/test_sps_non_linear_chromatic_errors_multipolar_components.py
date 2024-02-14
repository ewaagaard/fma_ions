"""
Test differences between SPS lines with and without non-linear magnet errors - check magnet components in both cases
"""
import fma_ions
import matplotlib.pyplot as plt
import numpy as np


# Load lines with and without magnet errors
sps = fma_ions.SPS_sequence_maker()
line, twiss = sps.load_xsuite_line_and_twiss(add_non_linear_magnet_errors=False)

# With magnet errors 
sps2 = fma_ions.SPS_sequence_maker()
line2, twiss2 = sps2.load_xsuite_line_and_twiss(add_non_linear_magnet_errors=True)

my_dict = line.to_dict()
my_dict2 = line2.to_dict()
d =  my_dict["elements"]
d2 =  my_dict2["elements"]

# Check if these are different
for key, value in d.items():
    if value['__class__'] == 'Multipole':
        if not np.array_equal(value['knl'][-1], d2[key]['knl'][-1]):
            print('{}: ideal: {} vs error: {}'.format(key, value['knl'][-1], d2[key]['knl'][-1]))
