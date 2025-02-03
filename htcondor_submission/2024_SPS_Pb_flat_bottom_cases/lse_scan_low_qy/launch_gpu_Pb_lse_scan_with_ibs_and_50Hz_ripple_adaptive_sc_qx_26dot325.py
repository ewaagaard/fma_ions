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
master_name = 'launch_gpu_Pb_lse_scan_with_ibs_and_50Hz_ripple_adaptive_sc_qx_26dot325'
LSE_strengths = np.arange(0, 11.5, 2.0)
num_turns = 130_000 # corresponds to 3s for SPS ions at flat bottom
Qx = 26.325
Qy = 26.10
run_files = ['sps_run_lse_{}_tbt_ripple.py'.format(i+1) for i in range(len(LSE_strengths))]

# Define script and folder names
script_names = run_files.copy()
folder_names = ['sps_LSE_{:.3e}_{:.2f}_Qy_{:.2f}'.format(LSE_strengths[i], Qx, Qy) for i in range(len(LSE_strengths))]
string_array = ['LSE = {:.3e}, Qx = {:.2f}, Qy = {:.2f} space charge'.format(LSE_strengths[i], Qx, Qy) for i in range(len(LSE_strengths))]    

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
sps = fma_ions.SPS_Flat_Bottom_Tracker(qx0={:.3f}, qy0={:.3f}, num_turns=n_turns, num_part=num_part)
tbt = sps.track_SPS(which_context='gpu', distribution_type='qgaussian', install_SC_on_line=True, add_beta_beat=True, add_non_linear_magnet_errors=True, 
                    I_LSE={}, apply_kinetic_IBS_kicks=True, ibs_step = 2000, add_tune_ripple=True, SC_adaptive_interval_during_tracking=100)
tbt.to_json(output_dir)
    '''.format(num_turns, Qx, Qy, LSE_strengths[i])
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
sub.copy_plot_script_emittances_for_scan(master_name, folder_names, scan_array_for_x_axis='np.arange(0, 11.5, 2.0)',
                                             label_for_x_axis='LSE [I]', 
                                             extra_text_string='$Q_{x, y}$ = 26.325, 26.10 - q-Gaussian beam\n Frozen SC, 10% $\\beta$-beat + non-linear magnet errors\nLSE excitation')