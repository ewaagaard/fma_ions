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
master_name = 'Q26_Pb_ions_Qx_scan_LSE_exc_0_with_50_150_300_600_ripple_NO_COMP_TRANSFER_FUNCTION_ibs_adaptive'
num_turns = 130_000 # corresponds to 3s for SPS ions at flat bottom
Qy = 26.19
Qx_range = np.arange(26.28, 26.42, 0.01)
run_files = ['sps_run_qx_{}_tbt_qy_26dot19_no_comp.py'.format(i+1) for i in range(len(Qx_range))]

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

n_turns = {}
num_part = 20_000

# Transfer function factors
a_50 = 1.0 #1.7170
a_150 = 0.5098
a_300 = 0.2360
a_600 = 0.1095

# Desired ripple frequencies and amplitudes
ripple_freqs = np.array([50.0, 150.0, 300.0, 600.0])
kqf_amplitudes = np.array([1.0141062492337905e-06*a_50, 1.9665396648867768e-07*a_150, 3.1027971430227987e-07*a_300, 4.5102937494506313e-07*a_600])
kqd_amplitudes = np.array([1.0344583265981035e-06*a_50, 4.5225494700433166e-07*a_150, 5.492718035100028e-07*a_300, 4.243698659233664e-07*a_600])
kqf_phases = np.array([0.7646995873548973, 2.3435670020522825, -1.1888958255027886, 2.849205512655574])
kqd_phases = np.array([0.6225130389353318, -1.044380492147742, -1.125401419249802, -0.30971750008702853])

# Tracking on GPU context
sps = fma_ions.SPS_Flat_Bottom_Tracker(qx0={:.3f}, qy0={:.3f}, num_turns=n_turns, num_part=num_part)
tbt = sps.track_SPS(which_context='gpu', distribution_type='qgaussian', install_SC_on_line=True, add_beta_beat=True, add_non_linear_magnet_errors=True, 
                    apply_kinetic_IBS_kicks=True, ibs_step = 2000, add_tune_ripple=True, ripple_freqs = ripple_freqs,
                    kqf_amplitudes = kqf_amplitudes, kqd_amplitudes = kqd_amplitudes, kqf_phases=kqf_phases, kqd_phases=kqd_phases,
                    SC_adaptive_interval_during_tracking=20)
tbt.to_json(output_dir)
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
