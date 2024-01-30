"""
Real-case scenario of 50 Hz tune ripple in SPS with aiming for 10% beta-beat in Y-plane
- inspired by Hannes tune ripple studies: https://indico.cern.ch/event/828559/contributions/3528378/attachments/1938914/3214441/2019.11.05_SC_and_ripple.pdf
"""
import fma_ions
import xtrack as xt

# Find Twiss from SPS Pb ions at injection
sps = fma_ions.SPS_sequence_maker()
line, twiss = sps.load_xsuite_line_and_twiss()

# Calculate tune ripple period
freq = 50. # Hz
T_ripple = 1./freq
ripples_per_turn = T_ripple/twiss['T_rev0'] #ripple time / time per turn --> ~860

# Beta-beat of 10%
sps_ripple = fma_ions.Tune_Ripple_SPS(ripple_period=ripples_per_turn, num_turns=20_000, n_linear=30, output_folder='output_tune_ripple_bb_0dot1')
sps_ripple.run_ripple_and_analysis(load_tbt_data=True, use_symmetric_lattice=False, install_SC_on_line=True, sextupolar_value_to_add=0.0, plot_random_colors=True,
                               also_show_plot=True, phase_sweep_up_to_turn=20_000, phase_space_sweep_interval=200, beta_beat=0.1,
                               plane_beta_beat='Y')