"""
SPS Pb ions: plot tune over action from generated turn-by-turn data
- On-momentum, ideal lattice

- Generate particle distribution with fixed Jy and varying Jx - "trace"" rather than "grid""
- Track particles and find tunes and corresponding action (and normalized beam size)
- Plot tune over action, and normalized phase space 
"""
import fma_ions
import numpy as np

d_min = -11.0

# On-momentum case - Qy = 0.25
fma_sps = fma_ions.FMA(output_folder='output_Pb_on_momentum_ideal_lattice_0dot25', num_turns=1200, n_linear=5000, z0=0.)
fma_sps.run_SPS(load_tbt_data=False, make_single_Jy_trace=True)

# Load Twiss and plot normalized phase space
sps_seq = fma_ions.SPS_sequence_maker()
sps_line, twiss_sps = sps_seq.load_xsuite_line_and_twiss(Qy_frac=25)

Jx, Jy, Qx, Qy, d = fma_sps.plot_tune_over_action(twiss_sps, also_show_plot=False, case_name='z0 = 0')

# Select index with a diffusion coefficient higher than a certain threshold
ind = np.where(d > d_min)[0]

fma_sps.plot_normalized_phase_space(twiss_sps, particle_index=ind, also_show_plot=True,
                                    case_name='z0 = 0')

"""
# On-momentum case - Qy = 0.19
fma_sps2 = fma_ions.FMA(output_folder='output_Pb_on_momentum_ideal_lattice_0dot19', z0=0., n_linear=1500)
fma_sps2.run_SPS(load_tbt_data=False, make_single_Jy_trace=True, Qy_frac=19)

# Load Twiss and plot normalized phase space
sps_line2, twiss_sps2 = sps_seq.load_xsuite_line_and_twiss(Qy_frac=19)

Jx2, Jy2, Qx2, Qy2, d2 = fma_sps.plot_tune_over_action(twiss_sps2, also_show_plot=True, case_name='z0 = 0')

fma_sps.plot_normalized_phase_space(twiss_sps, start_particle=1, plot_up_to_particle=200, also_show_plot=True,
                                    case_name='z0 = 0')
"""