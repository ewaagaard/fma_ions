"""
Launcher script to HTCondor for SPS cases with GPUs
"""
import fma_ions
import os
import pathlib

# Find path of script being run
dir_path = pathlib.Path(__file__).parent.absolute()
script_names = ['sps_BB_only_ibs_binomial.py', 'sps_ideal_lattice_only_ibs_binomial.py', 'sps_BB_only_ibs_binomial_before_RF_spill.py',
                'sps_ideal_lattice_only_ibs_binomial_before_RF_spill.py']
folder_names = ['sps_BB_only_ibs_binomial', 'sps_ideal_lattice_only_ibs_binomial', 'sps_BB_only_ibs_binomial_before_RF_spill',
                'sps_ideal_lattice_only_ibs_binomial_before_RF_spill']
string_array = ['IBS with BB, binomial', 'IBS with ideal lattice, binomial', 'IBS with BB before RF spill, binomial', 
                'IBS with ideal lattice before RF spill, binomial']    

# Instantiate the submitter class and launch the two jobs
sub = fma_ions.Submitter() 

# Launch the Python scripts in this folder
for i, script in enumerate(script_names):
    file_name = os.path.join(dir_path, script)
    print(f"Submitting {file_name}")
    sub.submit_GPU(file_name, extra_output_name_str=folder_names[i], number_of_turn_string='955k_turns', job_flavour='nextweek')
sub.copy_master_plot_script(folder_names, string_array)

