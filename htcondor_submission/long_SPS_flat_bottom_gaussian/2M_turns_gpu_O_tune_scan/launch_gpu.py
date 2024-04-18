"""
Launcher script to HTCondor for SPS cases with tune scan - with GPUs
"""
import fma_ions
import os
import pathlib

# Find path of script being run
dir_path = pathlib.Path(__file__).parent.absolute()
script_names = ['sps_BB_sc_Qy17.py', 'sps_BB_sc_Qy18.py', 'sps_BB_sc_Qy19.py', 'sps_BB_sc_Qy20.py', 'sps_BB_sc_Qy21.py',
                'sps_BB_sc_Qy22.py', 'sps_BB_sc_Qy23.py', 'sps_BB_sc_Qy24.py', 'sps_BB_sc_Qy25.py']
folder_names = ['sps_BB_sc_2M_turns_Qy17', 'sps_BB_sc_2M_turns_Qy18', 'sps_BB_sc_2M_turns_Qy19', 'sps_BB_sc_2M_turns_Qy20',
                'sps_BB_sc_2M_turns_Qy21', 'sps_BB_sc_2M_turns_Qy22', 'sps_BB_sc_2M_turns_Qy23', 'sps_BB_sc_2M_turns_Qy24', 'sps_BB_sc_2M_turns_Qy25']
string_array = ['SC + BB: Qy = 17', 'SC + BB: Qy = 18', 'SC + BB: Qy = 19', 'SC + BB: Qy = 20', 'SC + BB: Qy = 21',
                'SC + BB: Qy = 22', 'SC + BB: Qy = 23', 'SC + BB: Qy = 24', 'SC + BB: Qy = 25']    

# Instantiate the submitter class and launch the two jobs
sub = fma_ions.Submitter() 

# Launch the Python scripts in this folder
for i, script in enumerate(script_names):
    file_name = os.path.join(dir_path, script)
    print(f"Submitting {file_name}")
    sub.submit_GPU(file_name, extra_output_name_str=folder_names[i], number_of_turn_string='2M_turns', job_flavour='nextweek')
sub.copy_master_plot_script(folder_names, string_array)