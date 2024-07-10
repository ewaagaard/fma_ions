"""
Script to plot TBT data
"""
import fma_ions

sps = fma_ions.SPS_Flat_Bottom_Tracker()
sps.load_tbt_data_and_plot(show_plot=True)