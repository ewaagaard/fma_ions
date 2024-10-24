"""
Launcher script to HTCondor for SPS cases with GPUs
"""
import fma_ions
import os
import pathlib

# Find path of script being run
dir_path = pathlib.Path(__file__).parent.absolute()
script_names = ['sps_0dot1_BB_X_and_Y_sc_binomial_Q26.py', 'sps_0dot1_BB_X_and_Y_sc_gaussian_Q26.py', 'sps_0dot02_BB_X_and_Y_sc_binomial_Q26.py',
                'sps_0dot02_BB_X_and_Y_sc_gaussian_Q26.py', 'sps_0dot05_BB_X_and_Y_sc_binomial_Q26.py', 'sps_0dot05_BB_X_and_Y_sc_gaussian_Q26.py']
folder_names = ['sps_0dot1_BB_X_and_Y_sc_binomial_Q26', 'sps_0dot1_BB_X_and_Y_sc_gaussian_Q26', 'sps_0dot02_BB_X_and_Y_sc_binomial_Q26',
                'sps_0dot02_BB_X_and_Y_sc_gaussian_Q26', 'sps_0dot05_BB_X_and_Y_sc_binomial_Q26', 'sps_0dot05_BB_X_and_Y_sc_gaussian_Q26']
string_array = ['sps_0dot1_BB_X_and_Y_sc_binomial_Q26', 'sps_0dot1_BB_X_and_Y_sc_gaussian_Q26', 'sps_0dot02_BB_X_and_Y_sc_binomial_Q26',
                'sps_0dot02_BB_X_and_Y_sc_gaussian_Q26', 'sps_0dot05_BB_X_and_Y_sc_binomial_Q26', 'sps_0dot05_BB_X_and_Y_sc_gaussian_Q26']    

# Instantiate the submitter class and launch the two jobs
sub = fma_ions.Submitter() 

# Launch the Python scripts in this folder
for i, script in enumerate(script_names):
    file_name = os.path.join(dir_path, script)
    print(f"Submitting {file_name}")
    sub.submit_GPU(file_name, extra_output_name_str=folder_names[i], number_of_turn_string='100k_turns', job_flavour='nextweek')
sub.copy_master_plot_script(folder_names, string_array)

