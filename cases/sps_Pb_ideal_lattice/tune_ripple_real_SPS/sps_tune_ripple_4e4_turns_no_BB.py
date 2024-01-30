"""
Real-case scenario of 50 Hz tune ripple in SPS - no-beta-beat
"""

import fma_ions
import xtrack as xt

# Find Twiss from SPS Pb ions at injection
sps = fma_ions.SPS_sequence_maker()
line, twiss = sps.load_xsuite_line_and_twiss()

# Calculate tune ripple period
freq = 50. # Hz
T_ripple = 1./freq
ripples_per_turn = int(T_ripple/twiss['T_rev0']) #ripple time / time per turn --> ~860
print('\nNr of ripples per turn: {}\n'.format(ripples_per_turn))

# Beta-beat of 10%
sps_ripple = fma_ions.Tune_Ripple_SPS(ripple_period=ripples_per_turn, num_turns=20_000, n_linear=30, output_folder='output_tune_ripple')
sps_ripple.run_ripple_and_analysis(load_tbt_data=True, use_symmetric_lattice=False, install_SC_on_line=True, sextupolar_value_to_add=0.0, plot_random_colors=True,
                               also_show_plot=True, phase_sweep_up_to_turn=20_000, phase_space_sweep_interval=200, beta_beat=0.0,
                               plane_beta_beat='Y')