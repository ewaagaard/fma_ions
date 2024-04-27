"""
Launcher script to HTCondor for three SPS cases - with GPUs for 2000 turns - test PIC vs other for oxygen
"""
import fma_ions
import os
import pathlib

# Find path of script being run
dir_path = pathlib.Path(__file__).parent.absolute()
script_names = ['sps_BB_sc_Qydot25_PIC_250k_particles.py', 'sps_BB_sc_Qydot25_PIC_500k_particles.py', 'sps_BB_sc_Qydot25_PIC_1M_particles.py',
                'sps_BB_sc_Qydot25_PIC_2M_particles.py', 'sps_BB_sc_Qydot25_PIC_4M_particles.py', 'sps_BB_sc_Qydot25_frozen.py']
folder_names = ['sps_BB_sc_Qydot25_PIC_250k_particles_20k_turns', 'sps_BB_sc_Qydot25_PIC_500k_particles_20k_turn', 'sps_BB_sc_Qydot25_PIC_1M_particles_20k_turn',
                'sps_BB_sc_Qydot25_PIC_2M_particles_20k_turn', 'sps_BB_sc_Qydot25_PIC_4M_particles_20k_turn', 'sps_BB_sc_Qydot25_frozen_20k_turn']
string_array = ['PIC, 250k particles', 'PIC, 500k particles', 'PIC, 1M particles', 'PIC, 2M particles', 'PIC, 4M particles', 'Frozen']

# Instantiate the submitter class and launch the jobs
sub = fma_ions.Submitter() 

# Launch the Python scripts in this folder
for i, script in enumerate(script_names):
    file_name = os.path.join(dir_path, script)
    print(f"Submitting {file_name}")
    sub.submit_GPU(file_name, extra_output_name_str=folder_names[i], job_flavour='nextweek', number_of_turn_string='20k_turns')
sub.copy_master_plot_script(folder_names, string_array)

