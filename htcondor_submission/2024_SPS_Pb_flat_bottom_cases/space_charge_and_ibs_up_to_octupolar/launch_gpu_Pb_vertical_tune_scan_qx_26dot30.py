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
master_name = 'Q26_Pb_ions_IBS_and_SC_frozen_up_to_octupolar_errors_Qxdot30'
num_turns = 2_000_000 # corresponds to about 48 s for SPS ions at flat bottom
Qx = 26.30
Qy_range = np.arange(26.13, 26.30, 0.01)
run_files = ['sps_run_{}_tbt_qx_26dot30.py'.format(i+1) for i in range(len(Qy_range))]

# Define script and folder names
script_names = run_files.copy()
folder_names = ['sps_Qx_{:.2f}_Qy_{:.2f}'.format(Qx, Qy_range[i]) for i in range(len(Qy_range))]
string_array = ['Qy = {:.2f}, Qx = {:.2f} space charge'.format(Qy_range[i], Qx) for i in range(len(Qy_range))]    

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
tbt = sps.track_SPS(which_context='gpu',  distribution_type='qgaussian', install_SC_on_line=True, beta_beat=0.1, add_sextupolar_errors=True, add_octupolar_errors=True,
                add_non_linear_magnet_errors=True, add_tune_ripple=False, apply_kinetic_IBS_kicks=True)
tbt.to_json(output_dir)
    '''.format(num_turns, Qx, Qy_range[i])
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
sub.copy_plot_script_emittances_for_scan(master_name, folder_names, scan_array_for_x_axis=Qy_range,
                                             label_for_x_axis='$Q_{y}$', 
                                             extra_text_string='$Q_{x}$ = 26.30 - q-Gaussian beam\n IBS + Frozen SC, 10% $\\beta$-beat + up to octupolar\nmagnet errors')