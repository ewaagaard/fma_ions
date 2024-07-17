"""
Launcher script to HTCondor for SPS cases with GPUs
"""
import fma_ions
import os
import pathlib

# Find path of script being run
dir_path = pathlib.Path(__file__).parent.absolute()
script_names = ['sps_ideal_lattice_gaussian_only_ibs_q20_cycled_dpx.py', 'sps_ideal_lattice_gaussian_only_ibs_q26_cycled_dpx.py']
folder_names = ['sps_ideal_lattice_gaussian_only_ibs_q20_cycled_dpx', 'sps_ideal_lattice_gaussian_only_ibs_q26_cycled_dpx']
string_array = ['sps_ideal_lattice_gaussian_only_ibs_q20_cycled_dpx', 'sps_ideal_lattice_gaussian_only_ibs_q26_cycled_dpx']    

# Instantiate the submitter class and launch the two jobs
sub = fma_ions.Submitter() 

# Launch the Python scripts in this folder
for i, script in enumerate(script_names):
    file_name = os.path.join(dir_path, script)
    print(f"Submitting {file_name}")
    sub.submit_GPU(file_name, extra_output_name_str=folder_names[i], number_of_turn_string='60k_turns', job_flavour='nextweek')
sub.copy_master_plot_script(folder_names, string_array)

