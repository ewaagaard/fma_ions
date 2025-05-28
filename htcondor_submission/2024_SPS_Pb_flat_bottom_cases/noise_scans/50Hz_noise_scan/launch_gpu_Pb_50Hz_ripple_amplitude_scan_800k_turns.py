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
master_name = 'Q26_Pb_ions_50Hz_amplitude_scan_800k_turns_adaptive_SC'
num_turns = 800_000 
Qy = 26.19
Qx = 26.31
noise_amp = np.array([5, 23, 56, 79, 105, 132.8])

run_files = ['sps_run_50hz_amp_{}_tbt_ripple.py'.format(i+1) for i in range(len(noise_amp))]

# Define script and folder names
script_names = run_files.copy()
folder_names = ['sps_50hz_amp_{:.2f}'.format(noise_amp[i]) for i in range(len(noise_amp))]
string_array = ['Summed norm. FFT 50 Hz amplitude = {:.2f}'.format(noise_amp[i]) for i in range(len(noise_amp))]    

# Define arrays with noise levels and components
kqd_amplitudes_array = ['np.array([9.723021321406122e-08, 2.3745188570956088e-07, 2.635005955653469e-07, 2.576444444457593e-07])',
                        'np.array([3.9812456975596433e-07, 2.2934266041829687e-07, 3.0684040552841907e-07, 4.282823340417963e-07])',
                        'np.array([1.1304820191071485e-06, 1.932807549565041e-07, 1.785663243936142e-07, 2.2877348015981624e-08])',
                        'np.array([1.6389600432376028e-06, 2.5543189963173063e-07, 2.6373035666438227e-07, 3.858885406771151e-07])',
                        'np.array([2.1225348518782994e-06, 1.6082104536963016e-07, 3.38113714803967e-08, 2.7701780425104516e-08])',
                        'np.array([2.7151802441949258e-06, 1.5517436224854464e-07, 2.835248658072942e-08, 2.2494477391887813e-08])']

kqf_amplitudes_array = ['np.array([1.120060346693208e-07, 2.205204907568259e-07, 3.0521849225806363e-07, 2.747771361555351e-07])',
                        'np.array([5.696599032489758e-07, 2.0930548316755448e-07, 3.36147508050999e-07, 4.649008644719288e-07])',
                        'np.array([1.196421521854063e-06, 1.929192734451135e-07, 2.0213194318330352e-07, 2.7282455405952533e-08])',
                        'np.array([1.6357727190552396e-06, 2.1367992530940683e-07, 3.4807700899364136e-07, 3.98198835682706e-07])',
                        'np.array([2.221944441771484e-06, 1.2213288869133976e-07, 1.9126094485955036e-08, 6.1088640812556605e-09])',
                        'np.array([2.7695612061506836e-06, 1.3427566614154784e-07, 3.6387920232527904e-08, 1.4197014230887817e-08])']

kqd_phases_array = ['np.array([1.145213, 1.7562652, -1.3936949, 1.5295615])',
                    'np.array([-1.5430037, -2.9268184, 1.636692, -2.0278132])',
                    'np.array([1.9029742, -2.7743878, 0.6021359, 1.5541494])',
                    'np.array([2.9723594, 2.4459786, 2.7609522, 2.6171744])',
                    'np.array([-3.1252482, -1.628448, -1.258592, -1.878203])',
                    'np.array([1.1344357, -1.6943372, 2.9972126, -1.3131486])']

kqf_phases_array = ['np.array([-2.2178884, -2.043975, 1.6096314, -2.003183])',
                    'np.array([1.6630147, -1.3794123, -1.2727182, 0.08756675])',
                    'np.array([0.41851026, 0.7496304, 1.8946044, -1.2702432])',
                    'np.array([0.21719944, -1.7445294, 0.08148171, 0.14667067])',
                    'np.array([-0.20396788, 0.5095066, -2.0499842, 1.3687758])',
                    'np.array([2.5333304, -3.0855007, 1.2785302, 1.9509202])']

Nb_array = ['3.4e8', '3.626e8', '3.616e8', '3.46e8', '3.6e8', '3.718e8']


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
                    apply_kinetic_IBS_kicks=True, ibs_step = 2000, add_tune_ripple=True, ripple_freqs = ripple_freqs,
                    kqf_amplitudes = kqf_amplitudes, kqd_amplitudes = kqd_amplitudes, kqf_phases=kqf_phases, kqd_phases=kqd_phases, SC_adaptive_interval_during_tracking=100)
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