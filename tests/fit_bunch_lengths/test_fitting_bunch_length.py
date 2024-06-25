"""
Test fitting bunch length
"""
import fma_ions

sps = fma_ions.SPS_Plotting()
sps.fit_bunch_lengths_to_data(show_final_profile=True)
sps.plot_tracking_data()