"""
Launcher script to HTCondor for three SPS cases - with GPUs for 2M turns
"""
import fma_ions
import os
import pathlib

# Find path of script being run
dir_path = pathlib.Path(__file__).parent.absolute()
script_names = ['sps_ideal_lattice_sc.py']
folder_names = ['sps_ideal_lattice_sc_500k_turns']
string_array = ['SC ideal lattice linear in zeta']    

# Instantiate the submitter class and launch the two jobs
sub = fma_ions.Submitter() 

# Launch the Python scripts in this folder
for i, script in enumerate(script_names):
    file_name = os.path.join(dir_path, script)
    print(f"Submitting {file_name}")
    sub.submit_CPU(file_name, extra_output_name_str=folder_names[i], number_of_turn_string='500k_turns', job_flavour='nextweek', 
                   output_format='json')
sub.copy_master_plot_script(folder_names, string_array)

