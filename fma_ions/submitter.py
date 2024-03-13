"""
Class container for batch submission to HTCondor
"""
import os, stat, datetime

class Submitter:
    """Class to submit jobs to HTCondor more easliy"""
    
    def submit_CPU(self, 
                   python_script_source_path,
                   output_folder_eos='/eos/user/e/elwaagaa/PhD/Projects/fma_ions/htcondor_submission/output',
                   job_flavour="espresso",
                   nr_of_CPUs_to_request=8
                   ):
        """Method to submit .py script to HTCondor using CPUs"""
        
        # Specify which line from fma_ions
        python_script_name = os.path.basename(python_script_source_path)

        # Initiate settings for output
        settings = {}
        settings['output_directory_afs'] = '/afs/cern.ch/user/e/elwaagaa/public/sps_flat_bottom_tracking/output_logs'
        settings['output_directory_eos'] = '{}/{date:%Y_%m_%d__%H_%M}'.format(output_folder_eos, date=datetime.datetime.now() )
        os.makedirs(settings['output_directory_afs'], exist_ok=True)
        os.makedirs(settings['output_directory_afs'], exist_ok=True)
        turnbyturn_file_name = 'tbt.parquet'
        turnbyturn_path_eos = os.path.join(settings['output_directory_eos'], turnbyturn_file_name)

        # Create bash script and make executable
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

        # Make job file
        job_file = open(job_file_name,'w')
        job_file.write(
        f'''executable        = {bash_script_path}
        transfer_input_files  = {python_script_source_path}
        output                = {os.path.join(settings['output_directory_afs'],"$(ClusterId).$(ProcId).out")}
        error                 = {os.path.join(settings['output_directory_afs'],"$(ClusterId).$(ProcId).err")}
        log                   = {os.path.join(settings['output_directory_afs'],"$(ClusterId).$(ProcId).log")}
        request_CPUs = {nr_of_CPUs_to_request}
        +JobFlavour = {job_flavour}
        queue''')
        job_file.close()

        # Find the least loaded cluster and submit job
        os.system('myschedd bump')
        os.system(f'condor_submit {job_file_name}')
    
    
    def submit_GPU(self, 
                   python_script_source_path,
                   output_folder_eos='/eos/user/e/elwaagaa/PhD/Projects/fma_ions/htcondor_submission/output',
                   job_flavour="testmatch",
                   ):
        """Method to submit .py script to HTCondor with GPUs"""
        
        # Specify which line from fma_ions
        python_script_name = os.path.basename(python_script_source_path)
        
        # Initiate settings for output
        settings = {}
        settings['output_directory_afs'] = '/afs/cern.ch/user/e/elwaagaa/public/sps_flat_bottom_tracking/output_logs'
        settings['output_directory_eos'] = '{}/{date:%Y_%m_%d__%H_%M}'.format(output_folder_eos, date=datetime.datetime.now() )
        os.makedirs(settings['output_directory_afs'], exist_ok=True)
        os.makedirs(settings['output_directory_afs'], exist_ok=True)
        turnbyturn_file_name = 'tbt.parquet'
        turnbyturn_path_eos = os.path.join(settings['output_directory_eos'], turnbyturn_file_name)
        
        # Create bash script and make executable
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
        request_GPUs = 1
        +JobFlavour = {job_flavour}
        requirements = regexp("V100", TARGET.CUDADeviceName) || regexp("A100", TARGET.CUDADeviceName)
        queue''')
        
        job_file.close()
        
        # Find the least loaded cluster and submit job
        os.system('myschedd bump')
        os.system(f'condor_submit {job_file_name}')
