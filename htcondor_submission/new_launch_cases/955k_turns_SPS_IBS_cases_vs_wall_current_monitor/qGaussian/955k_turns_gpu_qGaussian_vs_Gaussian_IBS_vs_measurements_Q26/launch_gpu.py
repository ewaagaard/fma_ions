"""
Launcher script to HTCondor for SPS cases with GPUs
"""
import fma_ions
import os
import pathlib

# Find path of script being run
dir_path = pathlib.Path(__file__).parent.absolute()
script_names = ['sps_ideal_lattice_only_ibs_gaussian.py', 'sps_ideal_lattice_only_ibs_binomial_before_RF_spill.py',
                'sps_ideal_lattice_only_ibs_binomial.py', 'sps_10percBB_sc_ibs_binomial.py',
                'sps_10percBB_sc_binomial.py', 'sps_10percBB_only_ibs_binomial.py', 
                'sps_5percBB_sc_ibs_gaussian.py', 'sps_5percBB_sc_ibs_binomial.py',
                'sps_5percBB_sc_binomial.py', 'sps_5percBB_only_ibs_binomial.py',
                'sps_2percBB_sc_ibs_binomial.py', 'sps_2percBB_sc_binomial.py',
                'sps_2percBB_only_ibs_binomial.py']
folder_names = ['sps_ideal_lattice_only_ibs_gaussian', 'sps_ideal_lattice_only_ibs_binomial_before_RF_spill',
                'sps_ideal_lattice_only_ibs_binomial', 'sps_10percBB_sc_ibs_binomial',
                'sps_10percBB_sc_binomial', 'sps_10percBB_only_ibs_binomial', 
                'sps_5percBB_sc_ibs_gaussian', 'sps_5percBB_sc_ibs_binomial',
                'sps_5percBB_sc_binomial', 'sps_5percBB_only_ibs_binomial',
                'sps_2percBB_sc_ibs_binomial', 'sps_2percBB_sc_binomial',
                'sps_2percBB_only_ibs_binomial']
string_array = ['Ideal lattice' 'Ideal lattice, SC', 'Ideal lattice, IBS', 
                '5% beta-beat', '5% beta-beat, SC', '5% beta-beat, IBS', '5% beta-beat, SC + IBS',
                'Ideal lattice, IBS, before RF spill', '5% beta-beat, IBS, before RF spill',
                '5% beta-beat, SC and IBS, before RF spill']    

# Instantiate the submitter class and launch the two jobs
sub = fma_ions.Submitter() 

# Launch the Python scripts in this folder
for i, script in enumerate(script_names):
    file_name = os.path.join(dir_path, script)
    print(f"Submitting {file_name}")
    sub.submit_GPU(file_name, extra_output_name_str=folder_names[i], number_of_turn_string='955k_turns', job_flavour='nextweek')
sub.copy_master_plot_script(folder_names, string_array)

