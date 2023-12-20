"""
Test adding beta-beat to SPS line with space charge interactions - different vertical tunes
"""
import fma_ions

sps_fma = fma_ions.FMA(num_turns=120, n_linear=25, output_folder='Output_beta_beat_and_SC')

# Test generating a sequence with beta-beat
sps_seq = fma_ions.SPS_sequence_maker()


# First check tune at Qy = 26.25
sps_line_beat, twiss_beat = sps_seq.load_xsuite_line_and_twiss(Qy_frac=25, beta_beat=0.02)
line_SC_beat = sps_fma.install_SC_and_get_line(sps_line_beat, fma_ions.BeamParameters_SPS)
twiss = line_SC_beat.twiss()
print('Twiss for Qy = .25 succeeded!')

# Then check Qy = 26.19
sps_line_beat2, twiss_beat2 = sps_seq.load_xsuite_line_and_twiss(Qy_frac=19, beta_beat=0.02)
line_SC_beat_2 = sps_fma.install_SC_and_get_line(sps_line_beat2, fma_ions.BeamParameters_SPS)
twiss2 = line_SC_beat_2.twiss()
print('Twiss for Qy = .19 succeeded!')


# Then check Qy = 26.19 - higher beta beat! 
sps_line_beat3, twiss_beat3 = sps_seq.load_xsuite_line_and_twiss(Qy_frac=19, beta_beat=0.05)
line_SC_beat_3 = sps_fma.install_SC_and_get_line(sps_line_beat3, fma_ions.BeamParameters_SPS)
twiss3 = line_SC_beat_3.twiss()
print('Twiss for Qy = .19 with higher beta-beat succeeded!')