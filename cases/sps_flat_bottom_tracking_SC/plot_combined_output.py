"""
Script to plot emittance and intensity evolution for different scenarios
"""

output_str_array = ['output_ideal_lattice', 'output_BB', 'output_BB_and_magnet_errors', 'output_ideal_lattice_ibs']
string_array = ['SC ideal lattice', 'SC with BB', 'SC BB + magnet errors', 'SC + IBS ideal lattice']

import fma_ions
sps = fma_ions.SPS_Flat_Bottom_Tracker()
sps.plot_multiple_sets_of_tracking_data(output_str_array=output_str_array, string_array=string_array)