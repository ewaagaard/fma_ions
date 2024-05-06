"""
Launcher script to HTCondor for three SPS cases - with GPUs
"""
import fma_ions
import os
import pathlib

# Find path of script being run
dir_path = pathlib.Path(__file__).parent.absolute()
script_names = ['sps_ideal_lattice_sc.py', 'sps_ideal_lattice_sc_Qs_factor_0dot1.py', 'sps_ideal_lattice_sc_Qs_factor_0dot01.py', 'sps_ideal_lattice_sc_Qs_factor_0dot5.py',
		'sps_ideal_lattice_sc_Qs_factor_0dot25.py', 'sps_ideal_lattice_sc_Qs_factor_2.py', 'sps_ideal_lattice_sc_Qs_factor_4.py']
folder_names = ['sps_ideal_lattice_sc_100k_turns', 'sps_ideal_lattice_sc_Qs_factor_0dot1_100k_turns', 'sps_ideal_lattice_sc_Qs_factor_0dot01_100k_turns', 'sps_ideal_lattice_sc_Qs_factor_0dot5_100k_turns',
		'sps_ideal_lattice_sc_Qs_factor_0dot25_100k_turns', 'sps_ideal_lattice_sc_Qs_factor_2_100k_turns', 'sps_ideal_lattice_sc_Qs_factor_4_100k_turns']
string_array = ['Qs factor 1', 'Qs factor 0.1', 'Qs factor  0.01', 'Qs factor 0.5', 'Qs factor 0.25', 'Qs factor 2', 'Qs factor 4']    

# Instantiate the submitter class and launch the two jobs
sub = fma_ions.Submitter() 

# Launch the Python scripts in this folder
for i, script in enumerate(script_names):
    file_name = os.path.join(dir_path, script)
    print(f"Submitting {file_name}")
    sub.submit_GPU(file_name, extra_output_name_str=folder_names[i], number_of_turn_string='100k_turns', job_flavour='nextweek', 
                   output_format='json')
sub.copy_master_plot_script(folder_names, string_array)

