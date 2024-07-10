"""
Simple test script to launch SPS tracking - check that longitudinal transverse losses are plotted correctly

Want to ensure that particles that are lost are not included in plots
"""
import fma_ions

sps_plot = fma_ions.SPS_Plotting()
sps_plot.plot_longitudinal_monitor_data()
sps_plot.plot_WS_profile_monitor_data()