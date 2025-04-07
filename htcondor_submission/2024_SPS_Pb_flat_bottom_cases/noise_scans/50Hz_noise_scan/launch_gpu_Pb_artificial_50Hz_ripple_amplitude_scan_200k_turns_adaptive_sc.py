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
master_name = 'Q26_Pb_ions_artificial_50Hz_amplitude_scan_200k_turns_adaptive_SC'
num_turns = 200_000 
Qy = 26.19
Qx = 26.31
k_amp_array = np.array([1e-7, 5e-7, 1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6])

run_files = ['sps_run_artificial_50hz_amp_{}_tbt_ripple_scan.py'.format(i+1) for i in range(len(k_amp_array))]

# Define script and folder names
script_names = run_files.copy()
folder_names = ['sps_artificial_50hz_amp_{:.2f}'.format(k_amp_array[i]) for i in range(len(k_amp_array))]
string_array = ['Summed norm. FFT 50 Hz amplitude = {:.2f}'.format(k_amp_array[i]) for i in range(len(k_amp_array))]    

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

n_turns = {}
num_part = 20_000

# Tracking on GPU context
sps = fma_ions.SPS_Flat_Bottom_Tracker(qx0={:.3f}, qy0={:.3f}, num_turns=n_turns, num_part=num_part)
tbt = sps.track_SPS(which_context='gpu', distribution_type='qgaussian', install_SC_on_line=True, add_beta_beat=True, add_non_linear_magnet_errors=True, 
                    apply_kinetic_IBS_kicks=True, ibs_step = 2000, add_tune_ripple=True,
                    kqf_amplitudes = np.array([{}]), kqd_amplitudes = np.array([{}]), SC_adaptive_interval_during_tracking=20)
tbt.to_json(output_dir)
    '''.format(num_turns, Qx, Qy, k_amp_array[i], k_amp_array[i])
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
sub.copy_plot_script_emittances_for_scan(master_name, folder_names, scan_array_for_x_axis=k_amp_array,
                                             label_for_x_axis='Summed 50 Hz norm. FFT noise amplitude', 
                                             extra_text_string='$Q_{x, y}$ = 26.31, 26.19 - q-Gaussian beam\n Frozen SC, IBS, 15% $\\beta$-beat + non-linear magnet errors')