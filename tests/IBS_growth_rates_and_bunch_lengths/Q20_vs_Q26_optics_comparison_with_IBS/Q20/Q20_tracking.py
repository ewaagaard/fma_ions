"""
Check emittance growth after 2000 turns for Q20 vs Q26 optics
"""
import fma_ions

n_turns = 2000  
num_part = 10_000

sps = fma_ions.SPS_Flat_Bottom_Tracker(num_turns=n_turns, num_part=num_part, qx0=20.3, qy0=20.25, proton_optics='q20')
tbt = sps.track_SPS(which_context='cpu', install_SC_on_line=False, apply_kinetic_IBS_kicks=True, distribution_type='qgaussian')
tbt.to_json()

sps_plot = fma_ions.SPS_Plotting()
sps_plot.plot_tracking_data()