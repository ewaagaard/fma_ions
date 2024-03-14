"""
Launcher script to HTCondor for two SPS cases
"""
import fma_ions
import os
import pathlib

# Find path of script being run
dir_path = pathlib.Path(__file__).parent.absolute()
script_names = ['sps_ideal_lattice_sc.py', 'sps_BB_sc.py']
folder_names = ['sps_ideal_lattice', 'sps_BB']

# Instantiate the submitter class and launch the two jobs
sub = fma_ions.Submitter() 

# Launch the Python scripts in this folder
os.system('myschedd bump')
for i, script in enumerate(script_names):
    file_name = os.path.join(dir_path, script)
    print(f"Submitting {file_name}")
    sub.submit_CPU(file_name, extra_output_name_str=folder_names[i])

