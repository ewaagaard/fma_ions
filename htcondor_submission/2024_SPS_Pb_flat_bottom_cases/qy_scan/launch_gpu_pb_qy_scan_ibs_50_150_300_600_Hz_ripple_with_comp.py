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
master_name = 'Q26_Pb_ions_Qy_scan_ibs_50_150_300_600_Hz_ripple_adaptive_sc'
num_turns = 130_000 # corresponds to 3s for SPS ions at flat bottom
Qx = 26.31
Qy_range = np.arange(26.10, 26.26, 0.01)
run_files = ['sps_run_{}_tbt_qx_26dot31.py'.format(i+1) for i in range(len(Qy_range))]

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
import numpy as np
output_dir = './'

n_turns = {}
num_part = 20_000

# Desired ripple frequencies and amplitudes
ripple_freqs = np.array([50.0, 150.0, 300.0, 600.0])
kqf_amplitudes = np.array([1.6384433351717334e-08, 2.1158318710898557e-07, 3.2779826135772383e-07, 4.7273849059164697e-07])
kqd_amplitudes = np.array([2.753093584240069e-07, 4.511100472630622e-07, 5.796354631307802e-07, 4.5487568431405856e-07])
kqf_phases = np.array([0.9192671763874849, 0.030176158557178895, 0.5596488397663701, 0.050511945653341016])
kqd_phases = np.array([0.9985112397758237, 3.003827454851132, 0.6369886405485959, -3.1126209931146547])


# Tracking on GPU context
sps = fma_ions.SPS_Flat_Bottom_Tracker(qx0={:.3f}, qy0={:.3f}, num_turns=n_turns, num_part=num_part)
tbt = sps.track_SPS(which_context='gpu', distribution_type='qgaussian', install_SC_on_line=True, add_beta_beat=True,
                add_non_linear_magnet_errors=True, apply_kinetic_IBS_kicks=True, ibs_step = 2000,
                add_tune_ripple=True, ripple_freqs = ripple_freqs, kqf_amplitudes = kqf_amplitudes, 
                kqd_amplitudes = kqd_amplitudes, kqf_phases=kqf_phases, kqd_phases=kqd_phases,
                SC_adaptive_interval_during_tracking=20)
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