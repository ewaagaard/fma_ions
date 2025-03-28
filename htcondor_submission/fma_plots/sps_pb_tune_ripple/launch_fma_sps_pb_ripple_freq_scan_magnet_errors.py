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
master_name = 'Q26_Pb_FMA_on_momentum_ideal_lattice_ripple_scan'
ripple_freq = [10., 50., 150., 300., 600., 1200.]
Qx = 26.31
Qy = 26.25

run_files, folder_names, string_array = [], [] , []
for i in range(len(ripple_freq)):
    run_files.append('sps_run{}_ripple_magnet_errors.py'.format(i+1))
    folder_names.append('sps_Qx_{:.2f}_Qy_{:.2f}_ripple_{:.3e}_Hz_ripple_magnet_errors'.format(Qx, Qy, ripple_freq[i]))
    string_array.append('Qx = {:.2f}, Qy = {:.2f}, ripple = {:.3e}'.format(Qx, Qy, ripple_freq[i]))

# Generate the scripts to be submitted
for i in range(len(ripple_freq)):

    run_file = run_files[i]

    # Write run file for given tune
    print('Generating launch script {}\n'.format(run_file))
    run_file = open(run_file, 'w')
    run_file.truncate(0)  # remove existing content, if any
    run_file.write(
    '''import fma_ions
import numpy as np
output_dir = './'

# Tracking on GPU context
fma_sps = fma_ions.FMA(n_linear=200)
tbt = fma_sps.run_SPS(qx0={:.3f}, qy0={:.3f}, which_context = 'gpu', add_non_linear_magnet_errors=True, add_tune_ripple=True, ripple_freqs=np.array([{}]))
tbt.to_json(output_dir)
    '''.format(Qx, Qy, ripple_freq[i])
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
