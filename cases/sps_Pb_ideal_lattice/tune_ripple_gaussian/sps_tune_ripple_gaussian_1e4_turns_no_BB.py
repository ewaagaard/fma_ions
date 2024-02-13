"""
Real-case scenario of 50 Hz tune ripple in SPS 
- no-beta-beat
- Gaussian beam
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

# Run ripple with Gaussian beam
sps_ripple = fma_ions.Tune_Ripple_SPS(ripple_period=ripples_per_turn, num_turns=10_000, n_part=10_000, output_folder='output_tune_ripple_860_Hz__gaussian_10_000_particles_1e4_turns_no_BB')
sps_ripple.run_ripple_with_Gaussian_beam(dq=0.05, load_tbt_data=False, use_symmetric_lattice=False, install_SC_on_line=True, sextupolar_value_to_add=0.0, beta_beat=None)
