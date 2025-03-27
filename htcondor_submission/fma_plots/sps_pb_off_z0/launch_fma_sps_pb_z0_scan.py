"""
Launcher script to HTCondor for GPU - generate python scripts for FMA with initial z0 scan
"""
import fma_ions
import os
import pathlib
import numpy as np
import datetime

# Find path of script being run
dir_path = pathlib.Path(__file__).parent.absolute()

# Define run files and which parameters to change
master_name = 'Q26_Pb_FMA_on_momentum_ideal_lattice_z0_scan'

z0_range = np.array([0.0, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.])
Qx = 26.31
Qy = 26.25

run_files, folder_names, string_array = [], [] , []
for i in range(len(z0_range)):
    run_files.append('sps_run{}_z0.py'.format(i+1))
    folder_names.append('sps_Qx_{:.2f}_Qy_{:.2f}_z0_{:.3e}'.format(Qx, Qy, z0_range[i]))
    string_array.append('Qx = {:.2f}, Qy = {:.2f}, z0 = {:.3e}'.format(Qx, Qy, z0_range[i]))

# Generate the scripts to be submitted
for i in range(len(z0_range)):

    run_file = run_files[i]

    # Write run file for given tune
    print('Generating launch script {}\n'.format(run_file))
    run_file = open(run_file, 'w')
    run_file.truncate(0)  # remove existing content, if any
    run_file.write(
    '''import fma_ions
output_dir = './'

# Tracking on GPU context
fma_sps = fma_ions.FMA(n_linear=200, z0={})
tbt = fma_sps.run_SPS(qx0={:.3f}, qy0={:.3f}, which_context = 'gpu')
tbt.to_json(output_dir)
    '''.format(z0_range[i], Qx, Qy)
    )
    run_file.close()
    
# Instantiate the submitter class and launch the jobs
sub = fma_ions.Submitter() 
master_job_name = '{:%Y_%m_%d__%H_%M_%S}_{}'.format(datetime.datetime.now(), master_name)

# Launch the Python scripts in this folder
for k, script in enumerate(run_files):
    file_name = os.path.join(dir_path, script)
    print(f"Submitting {file_name}")
    sub.submit_GPU(file_name, master_job_name=master_job_name, job_name=folder_names[k])
sub.copy_master_fma_plot_script(folder_names)
