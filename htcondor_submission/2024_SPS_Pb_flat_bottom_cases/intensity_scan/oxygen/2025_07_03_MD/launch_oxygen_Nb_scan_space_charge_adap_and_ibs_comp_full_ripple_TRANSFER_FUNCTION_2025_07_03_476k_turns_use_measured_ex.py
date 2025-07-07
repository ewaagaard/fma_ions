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
master_name = 'SPS_O_ions_Nb_scan_adaptive_sc_ibs_with_comp_50_150_300_600_Hz_ripple_476k_turns_2025_07_03'
num_turns = 476_000
Qx = 26.31
Qy = 26.25

# Measured values from 2025-07-03 MD, reduce emittances by 10% to account for delayed measurement at inj.
exn0 = 0.9*np.array([1.28044905, 1.21522662, 1.09623646, 1.09909263, 1.28992405]) # too big? Maybe assume same as eyn?
eyn0 = 0.9*np.array([0.73679743, 0.84760539, 0.93580759, 1.00084611, 1.20491472])
Nb0 = np.array([16.58413887, 17.75226021, 21.62930679, 26.4480648 , 32.95989609])

# Use measured 2025-06-17 O8+ SPS values
Nb_array = ['{:.2f}e8'.format(Nb) for Nb in Nb0]  
exn_array = ['{:.2f}e-6'.format(exn) for exn in exn0]
eyn_array = ['{:.2f}e-6'.format(eyn) for eyn in eyn0]
run_files = ['sps_oxygen_Nb_scan_{}_longer_ex.py'.format(i+1) for i in range(len(Nb_array))]

# Define script and folder names
script_names = run_files.copy()
folder_names = ['sps_oxygen_Nb_scan_{}'.format(i) for i in range(len(Nb_array))]
string_array = ['SPS oxygen case = {}'.format(i) for i in range(len(Nb_array))]    

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

beamParams=fma_ions.BeamParameters_SPS()
beamParams.Nb = {}
beamParams.exn = {}
beamParams.eyn = {}

# Transfer function factors
a_50 = 1.0 #1.7170
a_150 = 0.5098
a_300 = 0.2360
a_600 = 0.1095

# Desired ripple frequencies and amplitudes - from DCCT measurements 2024-10-30
ripple_freqs = np.array([50.0, 150.0, 300.0, 600.0])
kqf_amplitudes = np.array([1.79381965522221e-07*a_50, 1.7917856960711038e-07*a_150, 1.717715125357188e-07*a_300, 1.0613897587376263e-07*a_600])
kqd_amplitudes = np.array([3.1433458408493135e-07*a_50, 4.125645646596158e-07*a_150, 2.6325770762187453e-07*a_300, 8.302889259074001e-08*a_600])
kqf_phases = np.array([2.5699456856082965, -1.2707524434033985, 1.1509405507521766, -2.3897351868985552])
kqd_phases = np.array([-1.6168418898711074, -1.5349070763197448, -2.145386063404577, 0.7431459693919794])

# Tracking on GPU context
sps = fma_ions.SPS_Flat_Bottom_Tracker(qx0={:.3f}, qy0={:.3f}, num_turns=n_turns, num_part=num_part)
tbt = sps.track_SPS(ion_type='O', which_context='gpu', distribution_type='qgaussian', beamParams=beamParams, install_SC_on_line=True, add_beta_beat=True, 
                    add_non_linear_magnet_errors=True, apply_kinetic_IBS_kicks=True, ibs_step = 2000, add_tune_ripple=True, ripple_freqs = ripple_freqs,
                    kqf_amplitudes = kqf_amplitudes, kqd_amplitudes = kqd_amplitudes, kqf_phases=kqf_phases, kqd_phases=kqd_phases,
                    SC_adaptive_interval_during_tracking=100)
tbt.to_json(output_dir)
    '''.format(num_turns, Nb_array[i], exn_array[i], eyn_array[i], Qx, Qy)
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
sub.copy_plot_script_emittances_for_scan(master_name, folder_names, scan_array_for_x_axis=Nb_array,
                                             label_for_x_axis='Injected Pb ions per bunch', 
                                             extra_text_string='$Q_{x, y}$ = 26.31, 26.25 - q-Gaussian beam\\nAdaptive SC, ~10% $\\beta$-beat + non-linear magnet errors')