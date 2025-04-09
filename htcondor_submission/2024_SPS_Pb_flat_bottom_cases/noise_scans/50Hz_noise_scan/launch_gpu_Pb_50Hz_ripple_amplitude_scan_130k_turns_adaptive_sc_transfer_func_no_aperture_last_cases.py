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
master_name = 'Q26_Pb_ions_50Hz_amplitude_scan_130k_turns_adaptive_SC_transfer_func_no_aperture_last_cases'
num_turns = 130_000 
Qy = 26.19
Qx = 26.31
noise_amp = np.array([56, 105, 132.8])

run_files = ['sps_run_50hz_amp_{}_tbt_ripple_scan.py'.format(i+1) for i in range(len(noise_amp))]

# Define script and folder names
script_names = run_files.copy()
folder_names = ['sps_50hz_amp_{:.2f}'.format(noise_amp[i]) for i in range(len(noise_amp))]
string_array = ['Summed norm. FFT 50 Hz amplitude = {:.2f}'.format(noise_amp[i]) for i in range(len(noise_amp))]    

# Define arrays with noise levels and components
kqd_amplitudes_array = ['np.array([1.1304820191071485e-06, 1.932807549565041e-07, 1.785663243936142e-07, 2.2877348015981624e-08])',
                        'np.array([2.1225348518782994e-06, 1.6082104536963016e-07, 3.38113714803967e-08, 2.7701780425104516e-08])',
                        'np.array([2.7151802441949258e-06, 1.5517436224854464e-07, 2.835248658072942e-08, 2.2494477391887813e-08])']

kqf_amplitudes_array = ['np.array([1.196421521854063e-06, 1.929192734451135e-07, 2.0213194318330352e-07, 2.7282455405952533e-08])',
                        'np.array([2.221944441771484e-06, 1.2213288869133976e-07, 1.9126094485955036e-08, 6.1088640812556605e-09])',
                        'np.array([2.7695612061506836e-06, 1.3427566614154784e-07, 3.6387920232527904e-08, 1.4197014230887817e-08])']

kqd_phases_array = ['np.array([1.9029742, -2.7743878, 0.6021359, 1.5541494])',
                    'np.array([-3.1252482, -1.628448, -1.258592, -1.878203])',
                    'np.array([1.1344357, -1.6943372, 2.9972126, -1.3131486])']

kqf_phases_array = ['np.array([0.41851026, 0.7496304, 1.8946044, -1.2702432])',
                    'np.array([-0.20396788, 0.5095066, -2.0499842, 1.3687758])',
                    'np.array([2.5333304, -3.0855007, 1.2785302, 1.9509202])']

Nb_array = ['3.616e8', '3.6e8', '3.718e8']


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
transfer_func_fact = np.array([a_50, a_150, a_300, a_600])

# Desired ripple frequencies and amplitudes
ripple_freqs = np.array([50., 150., 300., 600.])
kqd_amplitudes = {}
kqf_amplitudes = {}
kqd_phases = {}
kqf_phases = {}

beamParams=fma_ions.BeamParameters_SPS()
beamParams.Nb = {}
beamParams.exn = 2.25e-6 # measured on 2024-11-13
beamParams.eyn = 1.18e-6 # measured on 2024-11-13

# Tracking on GPU context
sps = fma_ions.SPS_Flat_Bottom_Tracker(qx0={:.3f}, qy0={:.3f}, num_turns=n_turns, num_part=num_part)
tbt = sps.track_SPS(which_context='gpu', distribution_type='qgaussian', beamParams=beamParams, install_SC_on_line=True, add_beta_beat=True, add_non_linear_magnet_errors=True,
                    add_aperture=False, use_effective_aperture=False,
                    apply_kinetic_IBS_kicks=True, ibs_step = 2000, add_tune_ripple=True, ripple_freqs = ripple_freqs,
                    kqf_amplitudes = transfer_func_fact*kqf_amplitudes, kqd_amplitudes = transfer_func_fact*kqd_amplitudes, 
                    kqf_phases=kqf_phases, kqd_phases=kqd_phases, SC_adaptive_interval_during_tracking=100)
tbt.to_json(output_dir)
    '''.format(num_turns, kqd_amplitudes_array[i], kqf_amplitudes_array[i],
               kqd_phases_array[i], kqf_phases_array[i], Nb_array[i],
               Qx, Qy)
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
sub.copy_plot_script_emittances_for_scan(master_name, folder_names, scan_array_for_x_axis=noise_amp,
                                             label_for_x_axis='Summed 50 Hz norm. FFT noise amplitude', 
                                             extra_text_string='$Q_{x, y}$ = 26.31, 26.19 - q-Gaussian beam\n Frozen SC, IBS, 15% $\\beta$-beat + non-linear magnet errors')