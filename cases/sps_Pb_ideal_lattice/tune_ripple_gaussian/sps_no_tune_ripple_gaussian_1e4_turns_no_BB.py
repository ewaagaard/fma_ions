"""
Real-case scenario of SPS tracking distribution, but without tune ripple
- no-beta-beat
- Gaussian beam
"""

import fma_ions
import xtrack as xt

# Find Twiss from SPS Pb ions at injection
sps = fma_ions.SPS_sequence_maker()
line, twiss = sps.load_xsuite_line_and_twiss()

# Run ripple with Gaussian beam and no tune ripple
sps_ripple = fma_ions.Tune_Ripple_SPS(num_turns=10_000, n_part=10_000, output_folder='output_no_tune_ripple_gaussian_10_000_particles_1e4_turns_no_BB')
sps_ripple.run_ripple_with_Gaussian_beam(load_tbt_data=False, use_symmetric_lattice=False, install_SC_on_line=True, sextupolar_value_to_add=0.0, beta_beat=None, vary_tune=False)
