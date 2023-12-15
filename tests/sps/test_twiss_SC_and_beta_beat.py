"""
Test adding beta-beat to SPS line with space charge interactions 
"""
import fma_ions
import numpy as np

sps_fma = fma_ions.FMA(num_turns=120, n_linear=25, output_folder='Output_beta_beat_and_SC')
Qy_fracs=[19, 25]


for Qy_frac in Qy_fracs:
    # Test generating a sequence with beta-beat
    sps_seq = fma_ions.SPS_sequence_maker()
    sps_line, twiss = sps_seq.load_xsuite_line_and_twiss(Qy_frac=Qy_frac)
    
    # Test installing space charge on this line
    line_SC = sps_fma.install_SC_and_get_line(sps_line, fma_ions.BeamParameters_SPS)
    twiss2 = line_SC.twiss()
    
    # Test adding beta-beat to line with space charge
    line_SC_beat = sps_seq.generate_xsuite_seq_with_beta_beat(beta_beat=0.05, line=line_SC)
    particles = sps_fma.generate_particles(line_SC_beat, fma_ions.BeamParameters_SPS)
    twiss3 = line_SC_beat.twiss()
    
    # As a fourth option, try to generate sequence with beta-beat and then installing space charge - DOES NOT SEEM TO WORK! 
    #sps_line_beat, twiss_beat = sps_seq.load_xsuite_line_and_twiss(Qy_frac=Qy_frac, beta_beat=0.05)
    #line_SC_beat_2 = sps_fma.install_SC_and_get_line(sps_line_beat, fma_ions.BeamParameters_SPS)
    #twiss4 = line_SC_beat_2.twiss()