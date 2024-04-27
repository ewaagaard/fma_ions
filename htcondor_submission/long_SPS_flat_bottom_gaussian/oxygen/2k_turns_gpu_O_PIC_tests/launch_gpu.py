"""
Launcher script to HTCondor for three SPS cases - with GPUs for 2000 turns - test PIC vs other for oxygen
"""
import fma_ions
import os
import pathlib

# Find path of script being run
dir_path = pathlib.Path(__file__).parent.absolute()
script_names = ['sps_BB_sc_Qydot19_frozen.py', 'sps_BB_sc_Qydot19_quasifrozen.py', 'sps_BB_sc_Qydot19_PIC.py',
                'sps_BB_sc_Qydot25_frozen.py', 'sps_BB_sc_Qydot25_quasifrozen.py', 'sps_BB_sc_Qydot25_PIC.py']
folder_names = ['sps_BB_sc_Qydot19_frozen_2K_turns', 'sps_BB_sc_Qydot19_quasifrozen_2K_turns', 'sps_BB_sc_Qydot19_PIC_2K_turns',
                'sps_BB_sc_Qydot25_frozen_2K_turns', 'sps_BB_sc_Qydot25_quasifrozen_2K_turns', 'sps_BB_sc_Qydot25_PIC_2K_turns']
string_array = ['Frozen SC, Qy = .19', 'Quasi-frozen SC, Qy = .19', 'PIC SC, Qy = .19',
                'Frozen SC, Qy = .25', 'Quasi-frozen SC, Qy = .25', 'PIC SC, Qy = .25']

# Instantiate the submitter class and launch the two jobs
sub = fma_ions.Submitter() 

# Launch the Python scripts in this folder
for i, script in enumerate(script_names):
    file_name = os.path.join(dir_path, script)
    print(f"Submitting {file_name}")
    sub.submit_GPU(file_name, extra_output_name_str=folder_names[i], job_flavour='nextweek', number_of_turn_string='2k_turns')
sub.copy_master_plot_script(folder_names, string_array)

