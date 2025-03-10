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
master_name = 'Q26_Pb_ions_Qx_scan_LSE_excitation_3_with_summed_50_150_300_600_ripple_effective_aperture_WITH_COMP_TRANSFER_FUNCTION_2024_11_25_ibs_adaptive_20'
num_turns = 130_000 # corresponds to 3s for SPS ions at flat bottom
Qy = 26.19
Qx_range = np.arange(26.28, 26.42, 0.01)
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

n_turns = {}
num_part = 20_000

# Desired ripple frequencies and amplitudes
ripple_freqs = np.array([50.0, 150.0, 300.0, 600.0])
kqf_amplitudes = np.array([1.6384433351717334e-08*0.4, 2.1158318710898557e-07*0.07558, 3.2779826135772383e-07*0.0254, 4.7273849059164697e-07*0.0543])
kqd_amplitudes = np.array([1.6331206486868624e-08*0.63, 2.108958328708343e-07*0.0685, 3.2673336803004776e-07*0.08636, 4.7120274094403095e-07*0.04666])
kqf_phases = np.array([0.9192671763874849, 0.030176158557178895, 0.5596488397663701, 0.050511945653341016])
kqd_phases = np.array([-2.2223254772020873, -3.1114164950326417, -2.581943813823403, -3.0910807079364635])

# Tracking on GPU context
sps = fma_ions.SPS_Flat_Bottom_Tracker(qx0={:.3f}, qy0={:.3f}, num_turns=n_turns, num_part=num_part)
tbt = sps.track_SPS(which_context='gpu', distribution_type='qgaussian', install_SC_on_line=True, add_beta_beat=True, add_non_linear_magnet_errors=True, 
                    I_LSE=-3.0, add_tune_ripple=True, apply_kinetic_IBS_kicks=True, ibs_step = 2000, ripple_freqs = ripple_freqs,
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