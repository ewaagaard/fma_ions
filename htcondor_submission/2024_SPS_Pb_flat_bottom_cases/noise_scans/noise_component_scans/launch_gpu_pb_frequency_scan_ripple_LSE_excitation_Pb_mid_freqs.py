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
master_name = 'Q26_Pb_ions_frequency_ripple_scan_adaptive_SC_excited_LSE_effective_aperture_mid_freqs'
num_turns = 60_000
Qx = 26.36
Qy = 26.19
frequencies = np.arange(1400., 3601., 200)
run_files = ['sps_run_{}_tbt.py'.format(i+1) for i in range(len(frequencies))]

# Define script and folder names
script_names = run_files.copy()
folder_names = ['sps_Qx_{:.2f}_Qy_{:.2f}_{:.1f}_Hz'.format(Qx, Qy, frequencies[i]) for i in range(len(frequencies))]
string_array = ['Qx = {:.2f}, Qy = {:.2f} {:.1f} Hz'.format(Qx, Qy, frequencies[i]) for i in range(len(frequencies))]    

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

ripple_freqs = np.array([{}])

# Tracking on GPU context
sps = fma_ions.SPS_Flat_Bottom_Tracker(qx0={:.3f}, qy0={:.3f}, num_turns=n_turns, num_part=num_part)
tbt = sps.track_SPS(which_context='gpu', distribution_type='qgaussian', install_SC_on_line=True, add_beta_beat=True,
                add_non_linear_magnet_errors=True, add_tune_ripple=False if ripple_freqs[0] == 0.0 else True,
                ripple_freqs = ripple_freqs, SC_adaptive_interval_during_tracking=100, I_LSE=-3.)
tbt.to_json(output_dir)
    '''.format(num_turns, frequencies[i], Qx, Qy)
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
sub.copy_plot_script_emittances_for_scan(master_name, folder_names, 
                                         scan_array_for_x_axis='np.arange(1400., 3601., 200)',
                                             label_for_x_axis='Ripple frequency [Hz]', 
                                             extra_text_string='$Q_{x, y}$ = 26.36, 26.19 - q-Gaussian Pb beam\\nAdaptive SC, ~10% $\\beta$-beat + non-linear magnet errors\\nExcited LSE')