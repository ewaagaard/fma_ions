"""
Launcher script to HTCondor for SPS cases with GPUs
"""
import fma_ions
import os
import pathlib

# Find path of script being run
dir_path = pathlib.Path(__file__).parent.absolute()
script_names = ['sps_ideal_lattice_only_ibs_qgaussian.py', 'sps_10percBB_sc_qgaussian.py', 'sps_10percBB_sc_ibs_qgaussian.py',
                'sps_10percBB_only_ibs_qgaussian.py', 'sps_5percBB_sc_qgaussian.py', 'sps_5percBB_sc_ibs_qgaussian.py',
                'sps_5percBB_only_ibs_qgaussian.py', 'sps_2percBB_sc_qgaussian.py',
                'sps_2percBB_sc_ibs_qgaussian.py', 'sps_2percBB_only_ibs_qgaussian.py']
folder_names = ['sps_ideal_lattice_only_ibs_qgaussian', 'sps_10percBB_sc_qgaussian', 'sps_10percBB_sc_ibs_qgaussian',
                'sps_10percBB_only_ibs_qgaussian', 'sps_5percBB_sc_qgaussian', 'sps_5percBB_sc_ibs_qgaussian',
                'sps_5percBB_only_ibs_qgaussian', 'sps_2percBB_sc_qgaussian',
                'sps_2percBB_sc_ibs_qgaussian', 'sps_2percBB_only_ibs_qgaussian']
string_array = ['sps_ideal_lattice_only_ibs_qgaussian', 'sps_10percBB_sc_qgaussian', 'sps_10percBB_sc_ibs_qgaussian',
                'sps_10percBB_only_ibs_qgaussian', 'sps_5percBB_sc_qgaussian', 'sps_5percBB_sc_ibs_qgaussian',
                'sps_5percBB_only_ibs_qgaussian', 'sps_2percBB_sc_qgaussian',
                'sps_2percBB_sc_ibs_qgaussian', 'sps_2percBB_only_ibs_qgaussian']    

# Instantiate the submitter class and launch the two jobs
sub = fma_ions.Submitter() 

# Launch the Python scripts in this folder
for i, script in enumerate(script_names):
    file_name = os.path.join(dir_path, script)
    print(f"Submitting {file_name}")
    sub.submit_GPU(file_name, extra_output_name_str=folder_names[i], number_of_turn_string='2M_turns', job_flavour='nextweek')
sub.copy_master_plot_script(folder_names, string_array)

