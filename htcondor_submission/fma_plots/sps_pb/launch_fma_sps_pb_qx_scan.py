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
master_name = 'Q26_Pb_FMA_LSE_excitation_3_on_momentum_ideal_lattice'
num_turns = 130_000 # corresponds to 3s for SPS ions at flat bottom
Qy = 26.25
Qx_range = np.arange(26.28, 26.4, 0.02)
run_files = ['sps_run_qx_{}_tbt_qy_26dot19_ripple.py'.format(i+1) for i in range(len(Qx_range))]

# Define script and folder names
script_names = run_files.copy()
folder_names = ['sps_Qx_{:.2f}_Qy_{:.2f}'.format(Qx_range[i], Qy) for i in range(len(Qx_range))]
string_array = ['Qx = {:.2f}, Qy = {:.2f} space charge'.format(Qx_range[i], Qy) for i in range(len(Qx_range))]    

# Generate the scripts to be submitted
for i, run_file in enumerate(run_files):
    
    # Write run file for given tune
    print('Generating launch script {}\n'.format(run_file))
    run_file = open(run_file, 'w')
    run_file.truncate(0)  # remove existing content, if any
    run_file.write(
    '''import fma_ions
import numpy as np
output_dir = './'

n_turns = 1200
num_part = 20_000

# Tracking on GPU context
fma_sps = fma_ions.FMA(output_folder=output_dir, z0=0., n_linear=200)
fma_sps.run_SPS()
    '''.format(num_turns, Qx_range[i], Qy)
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
sub.copy_plot_script_emittances_for_scan(master_name, folder_names, scan_array_for_x_axis='np.arange(26.28, 26.42, 0.01)',
                                             label_for_x_axis='$Q_{x}$', 
                                             extra_text_string='$Q_{y}$ = 26.19 - q-Gaussian beam\\nAdaptive SC, IBS, ~10% $\\beta$-beat + non-linear magnet errors\\nLSE excitation')