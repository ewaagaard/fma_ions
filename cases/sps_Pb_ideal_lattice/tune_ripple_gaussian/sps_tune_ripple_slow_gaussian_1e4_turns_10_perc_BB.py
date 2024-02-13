"""
Real-case scenario of slow tune ripple in SPS 
- with 10% beta-beat
- Gaussian beam
"""

import fma_ions
import xtrack as xt

# Find Twiss from SPS Pb ions at injection
sps = fma_ions.SPS_sequence_maker()
line, twiss = sps.load_xsuite_line_and_twiss()

# Run ripple with Gaussian beam - slow ripples
sps_ripple = fma_ions.Tune_Ripple_SPS(ripple_period=10_000, num_turns=20_000, n_part=10_000, output_folder='output_tune_ripple_slow_gaussian_10_000_particles_1e4_turns_10_perc_BB')
sps_ripple.run_ripple_with_Gaussian_beam(dq=0.05, load_tbt_data=False, use_symmetric_lattice=False, install_SC_on_line=True, sextupolar_value_to_add=0.0, beta_beat=0.1)
