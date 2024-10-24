"""
Class container for batch submission to HTCondor
"""
import os, stat, datetime, pathlib

class Submitter:
    """
    Class to submit jobs to HTCondor more easliy
    """
    
    def submit_CPU(self, 
                   python_script_source_path : str,
                   output_folder_eos: str = '/eos/user/e/elwaagaa/PhD/Projects/fma_ions/htcondor_submission/output',
                   master_job_name = None,
                   job_name = None,
                   nr_of_CPUs_to_request : int = 8,
                   change_to_best_node : bool = True,
                   copy_plot_scripts_to_output : bool = True,
                   output_format : str = 'json'
                   ):
        """
        Method to submit .py script to HTCondor using CPUs
        
        Instead of +JobFlavour = 'nextweek', use +MaxRuntime = 1209500 (two weeks)
        """

        # Specify which line from fma_ions
        python_script_name = os.path.basename(python_script_source_path)

        #### MASTER JOB NAME #### - if many jobs are run together, which folder should be located in
        if master_job_name is None:
            master_job_name = '{:%Y_%m_%d__%H_%M}'.format(datetime.datetime.now())

        #### JOB NAME #### -  Whether to create extra subfolder or not for job
        extra_str = '/{}'.format(job_name) if job_name is not None else ''

        # Initiate settings for output
        settings = {}
        settings['output_directory_afs'] = '/afs/cern.ch/work/e/elwaagaa/public/output_logs/{}{}'.format(master_job_name, extra_str)
        settings['output_directory_eos'] = '{}/{}_cpu_{}'.format(output_folder_eos, master_job_name, extra_str)
        self.output_folder_eos = '{}/{}_cpu'.format(output_folder_eos, master_job_name)
        
        # Make folder for output
        os.makedirs(settings['output_directory_afs'], exist_ok=True)
        os.makedirs(settings['output_directory_eos'], exist_ok=True)
        print('\nSaving EOS data to {}'.format(settings['output_directory_eos']))
        print('\nSaving AFS data to {}'.format(settings['output_directory_afs']))

        turnbyturn_file_name = f'tbt.{output_format}'
        turnbyturn_path_eos = os.path.join(settings['output_directory_eos'], turnbyturn_file_name)
        print('File: {}'.format(turnbyturn_path_eos))

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
+MaxRuntime = 1209500
queue'''
        )
        job_file.close()

        # Find the least loaded cluster and submit job - better to do it before launching many jobs
        if change_to_best_node:
            os.system('myschedd bump')
        os.system(f'condor_submit {job_file_name}')

        # Copy plot scripts to output folder
        if copy_plot_scripts_to_output:
            self.copy_plot_script(settings['output_directory_eos'])
    
    
    def submit_GPU(self, 
                   python_script_source_path : str,
                   output_folder_eos : str = '/eos/user/e/elwaagaa/PhD/Projects/fma_ions/htcondor_submission/output',
                   master_job_name = None,
                   job_name = None,
                   change_to_best_node : bool = True,
                   copy_plot_scripts_to_output : bool = True,
                   output_format : str = 'json'
                   ):
        """
        Method to submit .py script to HTCondor with GPUs
        
        Instead of +JobFlavour = 'nextweek', use +MaxRuntime = 1209500 (two weeks)
        """        

        # Specify which line from fma_ions
        python_script_name = os.path.basename(python_script_source_path)
        
        #### MASTER JOB NAME #### - if many jobs are run together, which folder should be located in
        if master_job_name is None:
            master_job_name = '{:%Y_%m_%d__%H_%M}'.format(datetime.datetime.now())

        #### JOB NAME #### -  Whether to create extra subfolder or not for job
        extra_str = '/{}'.format(job_name) if job_name is not None else ''

        # Initiate settings for output
        settings = {}
        settings['output_directory_afs'] = '/afs/cern.ch/work/e/elwaagaa/public/output_logs/{}{}'.format(master_job_name, extra_str)
        settings['output_directory_eos'] = '{}/{}_cpu_{}'.format(output_folder_eos, master_job_name, extra_str)
        self.output_folder_eos = '{}/{}_cpu'.format(output_folder_eos, master_job_name)
        
        # Make settings for output
        os.makedirs(settings['output_directory_afs'], exist_ok=True)
        os.makedirs(settings['output_directory_eos'], exist_ok=True)
        print('\nSaving EOS data to {}'.format(settings['output_directory_eos']))
        print('\nSaving AFS data to {}'.format(settings['output_directory_afs']))

        turnbyturn_file_name = f'tbt.{output_format}'
        turnbyturn_path_eos = os.path.join(settings['output_directory_eos'], turnbyturn_file_name)
        print('File: {}'.format(turnbyturn_path_eos))
        
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
+MaxRuntime = 1209500
queue'''
        )
        # previously also included "requirements = regexp("V100", TARGET.CUDADeviceName) || regexp("A100", TARGET.CUDADeviceName)"
        job_file.close()
        
        # Find the least loaded cluster and submit job - better to do it before launching many jobs
        if change_to_best_node:
            os.system('myschedd bump')
        os.system(f'condor_submit {job_file_name}')

        # Copy plot scripts to output folder
        if copy_plot_scripts_to_output:
            self.copy_plot_script(settings['output_directory_eos'])


    def copy_plot_script(self, eos_output_directory):
        """Create a simple TBT data plot script in the output directory"""
        plot_file = open('plot_tbt.py','w')
        plot_file.write(
        '''import fma_ions
# Load data and plot
sps_plot = fma_ions.SPS_Plotting()
sps_plot.plot_tracking_data()
sps_plot.plot_longitudinal_monitor_data()
sps_plot.plot_WS_profile_monitor_data()
        '''
        )
        plot_file.close()
        os.system(f'cp plot_tbt.py {os.path.join(eos_output_directory,"plot_tbt.py")}')


    def copy_master_plot_script(self, folder_names, string_names):
        """After all jobs have been submitted, create master plot script in the output directory with folders and legend strings specified"""

        plot_file = open('plot_combined_output.py','w')
        plot_file.write(
        f'''import fma_ions\nfolder_names = {folder_names}\nstring_array = {string_names}\n\nsps = fma_ions.SPS_Plotting()
sps.plot_multiple_sets_of_tracking_data(output_str_array=folder_names, string_array=string_array)
        '''
        )
        plot_file.close()
        os.system(f'cp plot_combined_output.py {os.path.join(self.output_folder_eos,"plot_combined_output.py")}')
        print(f'Successfully copied plot file to {self.output_folder_eos}')
