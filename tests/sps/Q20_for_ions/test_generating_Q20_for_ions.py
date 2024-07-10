"""
Small script to generate Q20 optics for ions
"""
import fma_ions

# First try normal sequence
sps_maker = fma_ions.SPS_sequence_maker(qx0=20.3, qy0=20.25, proton_optics='q20')
line, twiss = sps_maker.load_xsuite_line_and_twiss()

# Then generate with beta-beat
line2, twiss2 = sps_maker.load_xsuite_line_and_twiss(beta_beat=0.1, add_non_linear_magnet_errors=True)

# Then see if we can run the flat bottom tracker
sps = fma_ions.SPS_Flat_Bottom_Tracker(num_part=1000, num_turns=20, qx0=20.3, qy0=20.25, proton_optics='q20')
tbt = sps.track_SPS(add_non_linear_magnet_errors=True, beta_beat=0.1, Qy_frac=25)

# Then retry default, make sure it is Q26
sps_maker_q26 = fma_ions.SPS_sequence_maker()
line_q26, twiss_q26 = sps_maker_q26.load_xsuite_line_and_twiss()

print('Qx = {:.3f}, Qy = {:.3f}'.format(twiss['qx'], twiss['qy']))
print('With neta-beat: Qx = {:.3f}, Qy = {:.3f}'.format(twiss2['qx'], twiss2['qy']))
print('Default Qx = {:.3f}, Qy = {:.3f}'.format(twiss_q26['qx'], twiss_q26['qy']))