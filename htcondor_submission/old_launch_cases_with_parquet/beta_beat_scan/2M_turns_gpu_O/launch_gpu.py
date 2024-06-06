"""
Launcher script to HTCondor for three SPS cases - with GPUs for 2M turns
"""
import fma_ions
import os
import pathlib

# Find path of script being run
dir_path = pathlib.Path(__file__).parent.absolute()
script_names = ['sps_sc_BB_5perc_X.py', 'sps_sc_BB_10perc_X.py', 'sps_sc_BB_20perc_X.py',
                'sps_sc_BB_5perc_Y.py', 'sps_sc_BB_10perc_Y.py', 'sps_sc_BB_20perc_Y.py',
                'sps_sc_BB_5perc_both.py', 'sps_sc_BB_10perc_both.py', 'sps_sc_BB_20perc_both.py']
folder_names = ['sps_sc_BB_5perc_X_2M_turns', 'sps_sc_BB_10perc_X_2M_turns', 'sps_sc_BB_20perc_X_2M_turns',
                'sps_sc_BB_5perc_Y_2M_turns', 'sps_sc_BB_10perc_Y_2M_turns', 'sps_sc_BB_20perc_Y_2M_turns',
                'sps_sc_BB_5perc_both_2M_turns', 'sps_sc_BB_10perc_both_2M_turns', 'sps_sc_BB_20perc_both_2M_turns']
string_array = ['BB 5 perc in X', 'BB 10 perc in X', 'BB 20 perc in X',
                'BB 5 perc in Y', 'BB 10 perc in Y', 'BB 20 perc in Y',
                'BB 5 perc in X and Y', 'BB 10 perc in X and Y', 'BB 20 perc in X and Y']    

# Instantiate the submitter class and launch the two jobs
sub = fma_ions.Submitter() 

# Launch the Python scripts in this folder
for i, script in enumerate(script_names):
    file_name = os.path.join(dir_path, script)
    print(f"Submitting {file_name}")
    sub.submit_GPU(file_name, extra_output_name_str=folder_names[i], number_of_turn_string='2M_turns', job_flavour='nextweek')
sub.copy_master_plot_script(folder_names, string_array)

