"""
Launcher script to HTCondor for SPS cases with GPUs
"""
import fma_ions
import os
import pathlib

# Find path of script being run
dir_path = pathlib.Path(__file__).parent.absolute()
script_names = ['sps_ideal_lattice.py', 'sps_ideal_lattice_space_charge.py', 'sps_ideal_lattice_only_ibs.py', 
                'sps_BB_space_charge.py', 'sps_BB_only_ibs.py', 'sps_BB_space_charge_ibs.py']
folder_names = ['sps_ideal_lattice', 'sps_ideal_lattice_space_charge', 'sps_ideal_lattice_only_ibs', 
                'sps_BB_space_charge', 'sps_BB_only_ibs', 'sps_BB_space_charge_ibs']
string_array = ['Ideal lattice, no SC', 'Ideal lattice with SC', 'Ideal lattice with IBS', 
                'SC with $\beta$-beat', 'IBS with $\beta$-beat', 'SC + IBS with $\beta$-beat']    

# Instantiate the submitter class and launch the two jobs
sub = fma_ions.Submitter() 

# Launch the Python scripts in this folder
for i, script in enumerate(script_names):
    file_name = os.path.join(dir_path, script)
    print(f"Submitting {file_name}")
    sub.submit_GPU(file_name, extra_output_name_str=folder_names[i], number_of_turn_string='2M_turns', job_flavour='nextweek')
sub.copy_master_plot_script(folder_names, string_array)

