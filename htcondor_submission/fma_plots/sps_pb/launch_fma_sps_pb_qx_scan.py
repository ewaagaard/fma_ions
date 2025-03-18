"""
Launcher script to HTCondor for GPU - generate python scripts for FMA with Qx tune scan
"""
import fma_ions
import os
import pathlib
import numpy as np
import datetime

# Find path of script being run
dir_path = pathlib.Path(__file__).parent.absolute()

# Define run files and which parameters to change
master_name = 'Q26_Pb_FMA_on_momentum_ideal_lattice'

Qx_range = np.arange(26.28, 26.4, 0.02)
Qy_range = np.array([26.19, 26.25, 26.27])
run_files, folder_names, string_array = [], [] , []
for i in range(len(Qx_range)):
    run_files0 = []
    for j in range(len(Qy_range)):
        run_files0.append('sps_run_qx_{}_tbt_qy_{}_ripple.py'.format(i+1, j+1))
        folder_names.append('sps_Qx_{:.2f}_Qy_{:.2f}'.format(Qx_range[i], Qy_range[j]))
        string_array.append('Qx = {:.2f}, Qy = {:.2f} space charge'.format(Qx_range[i], Qy_range[j]))
    run_files.append(run_files0)

# Generate the scripts to be submitted
for i in range(len(Qx_range)):
    for j in range(len(Qy_range)):
        run_file = run_files[i][j]

        # Write run file for given tune
        print('Generating launch script {}\n'.format(run_file))
        run_file = open(run_file, 'w')
        run_file.truncate(0)  # remove existing content, if any
        run_file.write(
        '''import fma_ions
output_dir = './'

# Tracking on GPU context
fma_sps = fma_ions.FMA(n_linear=200)
fma_sps.run_SPS(qx0={:.3f}, qy0={:.3f}, which_context = 'gpu')
        '''.format(Qx_range[i], Qy_range[j])
        )
        run_file.close()
    
# Flatten list to define script names
script_names = [x for xs in run_files for x in xs]

# Instantiate the submitter class and launch the jobs
sub = fma_ions.Submitter() 
master_job_name = '{:%Y_%m_%d__%H_%M_%S}_{}'.format(datetime.datetime.now(), master_name)

# Launch the Python scripts in this folder
for k, script in enumerate(script_names):
    file_name = os.path.join(dir_path, script)
    print(f"Submitting {file_name}")
    sub.submit_GPU(file_name, master_job_name=master_job_name, job_name=folder_names[k])
sub.copy_master_fma_plot_script(folder_names)
