import fma_ions
import numpy as np
sps = fma_ions.SPS_Plotting()
folder_names = [None]

sps.fit_and_plot_transverse_profiles(scan_array_for_x_axis=[26.10], 
                                        label_for_x_axis='$Q_{y}$',
                                        extra_text_string='SPS Pb simulation\n$Q_{x}$ = 26.31. Adaptive space charge, IBS, ~10% $\\beta$-beat\n50 Hz ripple',
                                        emittance_range = [0.0, 4.1],
                                        transmission_range=[80.0, 105],
                                        output_str_array=folder_names,
                                        master_job_name='SPS_Q26_Pb_Qy_scan',
                                        load_measured_profiles=True,
                                        x_axis_quantity='Qy')
        
