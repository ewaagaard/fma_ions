"""
Launcher script to HTCondor for GPU
"""
import fma_ions
import os
import pathlib
import numpy as np
import datetime

# Find path of script being run
dir_path = pathlib.Path(__file__).parent.absolute()

# Define run files and which parameters to change
master_name = '_SPS_Pb_SC_all_errors_IBS'
num_turns = 100_000 # corresponds to about 3.46 s for protons
run_files = ['sps_run_1_tbt.py']

# Define script and folder names
script_names = run_files.copy()
folder_names = ['SPS_Pb_SC_all_errors_IBS']
string_array = ['IBS, SC, Pb with all magnet errors']    

# Generate the scripts to be submitted
for i, run_file in enumerate(run_files):
    
    # Write run file for given tune
    print('Generating launch script {}\n'.format(run_file))
    run_file = open(run_file, 'w')
    run_file.truncate(0)  # remove existing content, if any
    run_file.write(
    '''import fma_ions
output_dir = './'

n_turns = {}
num_part = 20_000


# Tracking on GPU context
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns, num_part=num_part)
tbt = sps.track_SPS(which_context='gpu', install_SC_on_line=True, beta_beat=0.1, add_sextupolar_errors=True, 
                add_non_linear_magnet_errors=True, add_tune_ripple=False, apply_kinetic_IBS_kicks=True)
tbt.to_json(output_dir)
    '''.format(num_turns)
    )
    run_file.close()
    
    
# Instantiate the submitter class and launch the jobs
sub = fma_ions.Submitter() 
master_job_name = '{:%Y_%m_%d__%H_%M_%S}_{}'.format(datetime.datetime.now(), master_name)

# Launch the Python scripts in this folder
for i, script in enumerate(script_names):
    file_name = os.path.join(dir_path, script)
    print(f"Submitting {file_name}")
    sub.submit_GPU(file_name, master_job_name=master_job_name, job_name=folder_names[i])
sub.copy_master_plot_script(folder_names, string_array)