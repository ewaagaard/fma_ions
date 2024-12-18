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
master_name = 'Q26_Pb_ions_bunch_intensity_scan'
num_turns = 2_000_000 # corresponds to 48s for SPS ions at flat bottom
Qy = 26.19
Qx = 26.31

no_LEIR_inj = [2, 3, 4, 5, 6, 8]
Nb_array = ['1.3e8', '1.75e8', '2.215e8', '2.93e8', '3.202e8', '3.94e8']
exn_array = ['0.73e-6', '0.89e-6', '1.11e-6', '1.4e-6', '1.75e-6', '2.1e-6']
eyn_array = ['0.48e-6', '0.63e-6', '0.74e-6', '0.85e-6', '1.05e-6', '1.175e-6']
run_files = ['sps_run_leir_inj_{}.py'.format(i+1) for i in range(len(no_LEIR_inj))]

# Define script and folder names
script_names = run_files.copy()
folder_names = ['sps_no_leir_inj_{}'.format(no_LEIR_inj[i]) for i in range(len(no_LEIR_inj))]
string_array = ['No. LEIR inj. = {}'.format(no_LEIR_inj[i]) for i in range(len(no_LEIR_inj))]    

# Only re-run case number 4
i = 2
run_file = run_files[i]
    
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

beamParams=fma_ions.BeamParameters_SPS()
beamParams.Nb = {}
beamParams.exn = {}
beamParams.eyn = {}

# Tracking on GPU context
sps = fma_ions.SPS_Flat_Bottom_Tracker(qx0={:.3f}, qy0={:.3f}, num_turns=n_turns, num_part=num_part)
tbt = sps.track_SPS(which_context='gpu', distribution_type='qgaussian', beamParams=beamParams, install_SC_on_line=False, 
                apply_kinetic_IBS_kicks=True, ibs_step = 5000)
tbt.to_json(output_dir)
'''.format(num_turns, Nb_array[i], exn_array[i], eyn_array[i], Qx, Qy)
)
run_file.close()
    
    
# Instantiate the submitter class and launch the jobs
sub = fma_ions.Submitter() 
master_job_name = '{:%Y_%m_%d__%H_%M_%S}_{}'.format(datetime.datetime.now(), master_name)

# Launch the Python scripts in this folder
script = script_names[i]    
file_name = os.path.join(dir_path, script)
print(f"Submitting {file_name}")
sub.submit_GPU(file_name, master_job_name=master_job_name, job_name=folder_names[i])
sub.copy_master_plot_script(folder_names, string_array)
sub.copy_plot_script_emittances_for_scan(master_name, folder_names, scan_array_for_x_axis=Nb_array,
                                             label_for_x_axis='Injected Pb ions per bunch', 
                                             extra_text_string='$Q_{x, y}$ = 26.31, 26.19 - q-Gaussian beam\n IBS')