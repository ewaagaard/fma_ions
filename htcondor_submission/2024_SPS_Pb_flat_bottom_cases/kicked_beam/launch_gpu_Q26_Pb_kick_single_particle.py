"""
Launcher script to HTCondor for GPU

Kicked beam without non-linearties and collective effects, set chroma knobs to zero
"""
import fma_ions
import os
import pathlib
import numpy as np
import datetime

# Find path of script being run
dir_path = pathlib.Path(__file__).parent.absolute()

# Define run files and which parameters to change
master_name = 'Q26_Pb_frequency_ripples_with_kicked_single_particle'
num_turns = 70_000
Qx = 26.31
Qy = 26.19
run_files = ['sps_run_0_tbt.py']

# Define script and folder names
script_names = run_files.copy()
folder_names = ['sps_run_0_tbt']
string_array = ['Kicked beam with all frequencies']    

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

# Generate spectrum with frequencies, then add same amplitudes and phases as the known 50 Hz component
ripple_freqs = np.hstack((np.arange(10., 100., 10), np.arange(100., 600., 50), np.arange(600., 1200., 100), np.arange(1200., 3601., 600))).ravel()
kqf_amplitudes = 9.7892e-7 * np.ones(len(ripple_freqs))
kqd_amplitudes = 9.6865e-7 * np.ones(len(ripple_freqs))
kqf_phases = 0.5564422 * np.ones(len(ripple_freqs))
kqd_phases = 0.4732764 * np.ones(len(ripple_freqs))

# Tracking on GPU context
sps = fma_ions.SPS_Flat_Bottom_Tracker(qx0={:.3f}, qy0={:.3f}, dqx0=0.0, dqy0=0.0, num_turns=n_turns)
tbt = sps.track_SPS(which_context='gpu', distribution_type='single', install_SC_on_line=False, 
                    add_tune_ripple=True, ripple_freqs = ripple_freqs, kqf_amplitudes = kqf_amplitudes, 
                    kqd_amplitudes = kqd_amplitudes, kqf_phases=kqf_phases, kqd_phases=kqd_phases, kick_beam=True)
tbt.to_json(output_dir)
    '''.format(num_turns, Qx, Qy)
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