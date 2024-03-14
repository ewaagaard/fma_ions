"""
Launcher script to HTCondor for three SPS cases - with GPUs for 800 000 turns
"""
import fma_ions
import os
import pathlib

# Find path of script being run
dir_path = pathlib.Path(__file__).parent.absolute()
script_names = ['sps_ideal_lattice_sc.py', 'sps_BB_sc.py', 'sps_BB_sc_ibs.py']
folder_names = ['sps_ideal_lattice_800K_turns', 'sps_BB_800K_turns', 'sps_BB_sc_ibs_800K_turns']

# Instantiate the submitter class and launch the two jobs
sub = fma_ions.Submitter() 

# Launch the Python scripts in this folder
for i, script in enumerate(script_names):
    file_name = os.path.join(dir_path, script)
    print(f"Submitting {file_name}")
    sub.submit_GPU(file_name, extra_output_name_str=folder_names[i], job_flavour='nextweek')

