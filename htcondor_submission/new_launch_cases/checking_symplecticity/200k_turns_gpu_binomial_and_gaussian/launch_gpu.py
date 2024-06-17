"""
Launcher script to HTCondor for SPS cases with GPUs
"""
import fma_ions
import os
import pathlib

# Find path of script being run
dir_path = pathlib.Path(__file__).parent.absolute()
script_names = ['sps_BB_sc_binomial_Q20.py', 'sps_BB_sc_binomial_Q26.py', 'sps_BB_sc_gaussian_Q20.py', 'sps_BB_sc_gaussian_Q26.py',
                'sps_ideal_lattice_sc_binomial_Q20.py', 'sps_ideal_lattice_sc_binomial_Q26.py', 'sps_ideal_lattice_sc_gaussian_Q20.py',
                'sps_ideal_lattice_sc_gaussian_Q26.py']
folder_names = ['sps_BB_sc_binomial_Q20', 'sps_BB_sc_binomial_Q26', 'sps_BB_sc_gaussian_Q20', 'sps_BB_sc_gaussian_Q26',
                'sps_ideal_lattice_sc_binomial_Q20', 'sps_ideal_lattice_sc_binomial_Q26', 'sps_ideal_lattice_sc_gaussian_Q20.py',
                'sps_ideal_lattice_sc_gaussian_Q26.py']
string_array = ['BB: SC binomial Q20', 'BB: SC binomial Q26', 'BB: SC Gaussian Q20', 'BB: SC Gaussian Q26',
                'Ideal lattice: SC binomial Q20', 'Ideal lattice: SC binomial Q26', 'Ideal lattice: SC Gaussian Q20', 'Ideal lattice: SC Gaussian Q26']    

# Instantiate the submitter class and launch the two jobs
sub = fma_ions.Submitter() 

# Launch the Python scripts in this folder
for i, script in enumerate(script_names):
    file_name = os.path.join(dir_path, script)
    print(f"Submitting {file_name}")
    sub.submit_GPU(file_name, extra_output_name_str=folder_names[i], number_of_turn_string='200k_turns', job_flavour='nextweek')
sub.copy_master_plot_script(folder_names, string_array)

