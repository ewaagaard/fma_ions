"""
Launcher script to HTCondor for three SPS cases - with GPUs for 2M turns
"""
import fma_ions
import os
import pathlib

# Find path of script being run
dir_path = pathlib.Path(__file__).parent.absolute()
script_names = ['sps_ideal_lattice.py', 'sps_ideal_lattice_sc.py', 
                'sps_BB.py', 'sps_BB_sc.py']
folder_names = ['sps_ideal_lattice_2M_turns', 'sps_ideal_lattice_sc_2M_turns',
                'sps_BB_2M_turns', 'sps_BB_sc_2M_turns']
string_array = ['Ideal lattice, no SC', 'SC ideal lattice', 
                'BB, no SC', 'SC with BB']    

# Instantiate the submitter class and launch the two jobs
sub = fma_ions.Submitter() 

# Launch the Python scripts in this folder
for i, script in enumerate(script_names):
    file_name = os.path.join(dir_path, script)
    print(f"Submitting {file_name}")
    sub.submit_GPU(file_name, extra_output_name_str=folder_names[i], number_of_turn_string='2M_turns', job_flavour='nextweek')
sub.copy_master_plot_script(folder_names, string_array)

