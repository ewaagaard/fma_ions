"""
Launcher script to HTCondor for CPU - generate python scripts for tune scan
"""
import fma_ions
import os
import pathlib
import numpy as np

# Find path of script being run
dir_path = pathlib.Path(__file__).parent.absolute()

# Define run files and which parameters to change
Qy_range = np.arange(26.13, 26.25, 0.01)
run_files = ['sps_run_{}_tbt.py'.format(i+1) for i in range(len(Qy_range))]

# Define script and folder names
script_names = run_files.copy()
folder_names = ['sps_Qy_{:.2f}_IBS_BB'.format(Qy_range[i]) for i in range(len(Qy_range))]
string_array = ['Qy = {:.2f}, IBS with BB'.format(Qy_range[i]) for i in range(len(Qy_range))]    

# Generate the scripts to be submitted
for i, run_file in enumerate(run_files):
    
    # Write run file for given tune
    print('Generating launch script {}\n'.format(run_file))
    run_file = open(run_file, 'w')
    run_file.truncate(0)  # remove existing content, if any
    run_file.write(
    '''import fma_ions
output_dir = './'

n_turns = 20
num_part = 20_000


# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(qy0={:.4f}, num_turns=n_turns, num_part=num_part)
tbt = sps.track_SPS(ion_type='proton', which_context='cpu', install_SC_on_line=False, beta_beat=0.1, 
                add_non_linear_magnet_errors=True, apply_kinetic_IBS_kicks=True)
tbt.to_json(output_dir)
    '''.format(Qy_range[i])
    )
    run_file.close()
    
    
# Instantiate the submitter class and launch the jobs
sub = fma_ions.Submitter() 

# Launch the Python scripts in this folder
for i, script in enumerate(script_names):
    file_name = os.path.join(dir_path, script)
    print(f"Submitting {file_name}")
    sub.submit_CPU(file_name, extra_output_name_str=folder_names[i], number_of_turn_string='')
sub.copy_master_plot_script(folder_names, string_array)