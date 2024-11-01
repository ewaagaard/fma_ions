"""
Launcher script to HTCondor for GPU - generate python scripts for tune scan
"""
import fma_ions
import os
import pathlib
import numpy as np
import datetime

# Find path of script being run
dir_path = pathlib.Path(__file__).parent.absolute()

# Define run files and which parameters to change
master_name = 'SPS_Q26_2b_Pb_ibs_only_ideal_lattice'
num_turns = 2_000_000 # corresponds to about 3.46 s for protons
Qx = 26.30
Qy = 26.19
run_files = ['sps_run_1_tbt.py']

# Define script and folder names
script_names = run_files.copy()
folder_names = ['sps_Qx_{:.2f}_Qy_{:.2f}'.format(Qx, Qy)]
string_array = ['Qy = {:.2f}, Qx = {:.2f} IBS'.format(Qy, Qx)]    

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

# Import beam parameters from 2024-10-30 test with high intensity
beamParams_2b = fma_ions.BeamParameters_SPS_2024_2b()

# Tracking on GPU context
sps = fma_ions.SPS_Flat_Bottom_Tracker(qx0={:.3f}, qy0={:.3f}, num_turns=n_turns, num_part=num_part)
tbt = sps.track_SPS(which_context='gpu', beamParams=beamParams_2b, distribution_type='qgaussian', install_SC_on_line=False, beta_beat=None, 
                add_non_linear_magnet_errors=False, apply_kinetic_IBS_kicks=True)
tbt.to_json(output_dir)
    '''.format(num_turns, Qx, Qy)
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