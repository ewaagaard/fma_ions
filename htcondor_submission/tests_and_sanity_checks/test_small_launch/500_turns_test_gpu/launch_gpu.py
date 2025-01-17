"""
Launcher script to HTCondor for GPU
"""
import fma_ions
import os
import pathlib
import datetime

# Find path of script being run
dir_path = pathlib.Path(__file__).parent.absolute()
script_names = ['sps_BB_only_ibs.py']
folder_names = ['sps_BB_only_ibs_500_turns']
string_array = ['IBS with BB']    

# Instantiate the submitter class and launch the two jobs
sub = fma_ions.Submitter() 
master_job_name = '{:%Y_%m_%d__%H_%M_%S}_{}'.format(datetime.datetime.now(), master_name)

# Launch the Python scripts in this folder
for i, script in enumerate(script_names):
    file_name = os.path.join(dir_path, script)
    print(f"Submitting {file_name}")
    sub.submit_GPU(file_name, master_job_name=master_job_name, job_name=folder_names[i])
sub.copy_master_plot_script(folder_names, string_array)
