"""
Launcher script to HTCondor for GPU - generate python scripts for tune and tune ripple scan
"""
import fma_ions
import os
import pathlib
import numpy as np
import datetime

# Find path of script being run
dir_path = pathlib.Path(__file__).parent.absolute()

# Define run files and which parameters to change
master_name = 'Q26_protons_ripple_amplitude_scan_SC_frozen_up_to_octupolar_magnet_errors_Qxdot13'
num_turns = 150_000 # corresponds to about 3.46 s for protons
Qx = 26.13
Qy_range = np.arange(26.15, 26.30, 0.01)
dq_range = np.array([0.01, 0.02, 0.05, 0.1])

# Create empty arrays
run_files = []
folder_names = []
string_arrays = []

# Generate the scripts to be submitted
for j in range(len(dq_range)):
    for i in range(len(Qy_range)):
        run_file = 'sps_run_{}_tbt_qx_26dot13_dq_{}.py'.format(i+1, j+1)
        run_files.append(run_file)
        folder_name = 'sps_Qx_{:.2f}_Qy_{:.2f}_dq_{:.2f}'.format(Qx, Qy_range[i], dq_range[j])
        folder_names.append(folder_name)
        string_array = 'Qy = {:.2f}, Qx = {:.2f} space charge, dq = {:.2f}'.format(Qy_range[i], Qx, dq_range[j]) 
        string_arrays.append(string_array)

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
tbt = sps.track_SPS(ion_type='proton', which_context='gpu', install_SC_on_line=True, beta_beat=0.1, add_sextupolar_errors=True, add_octupolar_errors=True,
                add_non_linear_magnet_errors=True, add_tune_ripple=True, dq={:.2f}, apply_kinetic_IBS_kicks=False)
tbt.to_json(output_dir)
        '''.format(num_turns, Qx, Qy_range[i], dq_range[j])
        )
        run_file.close()


# Define script and folder names
script_names = run_files.copy()
    
# Instantiate the submitter class and launch the jobs
sub = fma_ions.Submitter()
master_job_name = '{:%Y_%m_%d__%H_%M_%S}_{}'.format(datetime.datetime.now(), master_name)

# Launch the Python scripts in this folder
for i, script in enumerate(script_names):
    file_name = os.path.join(dir_path, script)
    print(f"Submitting {file_name}")
    sub.submit_GPU(file_name, master_job_name=master_job_name, job_name=folder_names[i])
sub.copy_master_plot_script(folder_names, string_array)
for j in range(len(dq_range)):
    dq_folder_names = ['sps_Qx_{:.2f}_Qy_{:.2f}_dq_{:.2f}'.format(Qx, Qy_range[k], dq_range[j]) for k in range(len(Qy_range))]
    sub.copy_plot_script_emittances_for_scan('{}_dq_{:.2f}'.format(master_name, dq_range[j]), dq_folder_names, scan_array_for_x_axis=Qy_range,
                                             label_for_x_axis='$Q_{y}$', 
                                             extra_text_string='$Q_{{x}}$ = {:.2f}. Frozen SC, 10% $\\beta$-beat + up to octupolar magnet errors + ripple'.format(Qx))