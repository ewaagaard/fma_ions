"""
SPS Pb ions: plot tune over action from generated turn-by-turn data
- On-momentum, symmetric lattice

- Generate particle distribution with fixed Jy and varying Jx - "trace"" rather than "grid""
- Track particles and find tunes and corresponding action (and normalized beam size)
- Plot tune over action, and normalized phase space 
"""
import fma_ions
import numpy as np

d_min = -11.5

# On-momentum case - Qy = 0.25
fma_sps = fma_ions.FMA(output_folder='output_Pb_on_momentum_symmetric_lattice_Jy_trace', num_turns=1200, 
                       n_linear=5000, z0=0., plot_order=4)
fma_sps.run_SPS(load_tbt_data=True, make_single_Jy_trace=True, use_symmetric_lattice=True)

# Load Twiss and plot normalized phase space
sps_seq = fma_ions.SPS_sequence_maker()
sps_line, twiss_sps = sps_seq.load_xsuite_line_and_twiss(Qy_frac=25, use_symmetric_lattice=True)

Jx, Jy, Qx, Qy, d = fma_sps.plot_tune_over_action(twiss_sps, load_tune_data=True, also_show_plot=False, case_name='z0 = 0')

# Select index with a diffusion coefficient higher than a certain threshold
ind = np.where(d > d_min)[0]

fma_sps.plot_normalized_phase_space(twiss_sps, particle_index=ind, also_show_plot=True,
                                    case_name='z0 = 0')
