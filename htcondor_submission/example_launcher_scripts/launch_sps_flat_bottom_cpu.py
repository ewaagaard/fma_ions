"""
Main launch script to create shell script and HTCondor submit file for Xsuite - using CPUs
"""
import os, json, shutil, stat, sys
import numpy as np

# Specify which line from fma_ions
python_script_source_path = '/afs/cern.ch/user/e/elwaagaa/public/sps_flat_bottom_tracking/tests/test_launch_script/sc_tests_comparing_time.py'
python_script_name = os.path.basename(python_script_source_path)

#sequence_file_path = '/afs/cern.ch/work/e/elwaagaa/public/space_charge/first_space_charge_gpu_example/test_sequence/SPS_2021_Pb_ions_matched_with_RF.json'

# initiate settings for output
settings = {}
settings['output_directory_afs'] = '/afs/cern.ch/user/e/elwaagaa/public/sps_flat_bottom_tracking/output_logs'
settings['output_directory_eos'] = '/eos/user/e/elwaagaa/PhD/Projects/fma_ions/htcondor_submission/output'
os.makedirs(settings['output_directory_afs'], exist_ok=True)
os.makedirs(settings['output_directory_afs'], exist_ok=True)
turnbyturn_file_name = 'tbt.parquet'
turnbyturn_path_eos = os.path.join(settings['output_directory_eos'], turnbyturn_file_name)

job_file_name = os.path.join(settings['output_directory_afs'], 'SPS_flat_bottom.job')


bash_script_path = os.path.join(settings['output_directory_afs'],'SPS_flat_bottom.sh')
bash_script_name = os.path.basename(bash_script_path)
bash_script = open(bash_script_path,'w')
bash_script.write(
f"""#!/bin/bash\n
echo 'sourcing environment'
source /afs/cern.ch/work/e/elwaagaa/public/venvs/miniconda/bin/activate
date
echo 'Running job'
python {python_script_name} 1> out.txt 2> err.txt
echo 'Done'
date
xrdcp -f {turnbyturn_file_name} {turnbyturn_path_eos}
xrdcp -f out.txt {os.path.join(settings['output_directory_eos'],"out.txt")}
xrdcp -f err.txt {os.path.join(settings['output_directory_eos'],"err.txt")}
xrdcp -f abort.txt {os.path.join(settings['output_directory_eos'],"abort.txt")}
""")

bash_script.close()
os.chmod(bash_script_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IXOTH)

job_file = open(job_file_name,'w')

job_file.write(
f'''executable        = {bash_script_path}
transfer_input_files  = {python_script_source_path}
output                = {os.path.join(settings['output_directory_afs'],"$(ClusterId).$(ProcId).out")}
error                 = {os.path.join(settings['output_directory_afs'],"$(ClusterId).$(ProcId).err")}
log                   = {os.path.join(settings['output_directory_afs'],"$(ClusterId).$(ProcId).log")}
request_CPUs = 4
+JobFlavour = "espresso"
queue''')

job_file.close()

#### IF EXTERNAL SEQUENCE FILE NEEDED, make sure to replace in the job_file above
# transfer_input_files  = {python_script_source_path},{sequence_file_path}

os.system('myschedd bump') # find the least loaded cluster
os.system(f'condor_submit {job_file_name}')
