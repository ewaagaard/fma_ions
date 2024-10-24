"""
Small test to see what happens 
"""

"""
SPS lattice with beta-beat and SC and IBS - with GPUs
"""
import fma_ions

n_turns = 80 
num_part = 3000

# Test default tracking with space charge on CPU context - then test plotting
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns, num_part=num_part, qx0=20.3, qy0=20.25, proton_optics='q20', turn_print_interval=10)
tbt = sps.track_SPS(which_context='cpu', install_SC_on_line=False, distribution_type='qgaussian', matched_for_PS_extraction=True,
                    nturns_profile_accumulation_interval=20)

sps_plot = fma_ions.SPS_Plotting()
sps_plot.plot_tracking_data(tbt_dict=tbt.to_dict(convert_to_numpy=True))
