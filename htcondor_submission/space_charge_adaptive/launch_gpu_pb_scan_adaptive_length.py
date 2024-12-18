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
master_name = 'Q26_Pb_ions_adaptive_SC_length_scan'
num_turns = 15_000 # corresponds to 3s for SPS ions at flat bottom
Qy = 26.10
Qx = 26.31
SC_adaptive_interval = [100, 200, 500, 1000, 2000, 5000]

run_files = ['sps_run_sc_interval_{}.py'.format(i+1) for i in range(len(SC_adaptive_interval))]

# Define script and folder names
script_names = run_files.copy()
folder_names = ['sps_Qy_{:.2f}_SC_interval_{:.2f}'.format(Qy, SC_adaptive_interval[i]) for i in range(len(SC_adaptive_interval))]
string_array = ['Qx = {:.2f}, Qy = {:.2f}. Adaptive space charge element length interval = {}'.format(Qx, Qy, 
                                                                                                      SC_adaptive_interval[i]) for i in range(len(SC_adaptive_interval))]    

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

# Higher intensity to see effects faster
beamParams=fma_ions.BeamParameters_SPS()
beamParams.Nb = 3.5e9

# Tracking on GPU context
sps = fma_ions.SPS_Flat_Bottom_Tracker(qx0={:.3f}, qy0={:.3f}, num_turns=n_turns, num_part=num_part)
tbt = sps.track_SPS(which_context='gpu', distribution_type='qgaussian', beamParams=beamParams, install_SC_on_line=True, add_beta_beat=True,
                add_non_linear_magnet_errors=True, apply_kinetic_IBS_kicks=False, SC_adaptive_interval_during_tracking={})
tbt.to_json(output_dir)
    '''.format(num_turns, Qx, Qy, SC_adaptive_interval[i])
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