"""
Launcher script to HTCondor for three SPS cases - with GPUs for 2000 turns - test PIC vs other for oxygen
"""
import fma_ions
import os
import pathlib

# Find path of script being run
dir_path = pathlib.Path(__file__).parent.absolute()
script_names = ['sps_BB_sc_Qydot15_frozen.py', 'sps_BB_sc_Qydot15_quasifrozen.py', 'sps_BB_sc_Qydot15_PIC.py']
folder_names = ['sps_BB_sc_Qydot15_frozen_30K_turns', 'sps_BB_sc_Qydot15_quasifrozen_30K_turns', 'sps_BB_sc_Qydot15_PIC_30K_turns']
string_array = ['Frozen SC, Qy = .15', 'Quasi-frozen SC, Qy = .15', 'PIC SC, Qy = .15']

# Instantiate the submitter class and launch the jobs
sub = fma_ions.Submitter() 

# Launch the Python scripts in this folder
for i, script in enumerate(script_names):
    file_name = os.path.join(dir_path, script)
    print(f"Submitting {file_name}")
    sub.submit_GPU(file_name, extra_output_name_str=folder_names[i], job_flavour='nextweek', number_of_turn_string='30k_turns')
sub.copy_master_plot_script(folder_names, string_array)

