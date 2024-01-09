"""
Generate SPS phase space plot from generated turn-by-turn data
"""
import fma_ions
import numpy as np

# On-momentum case

fma_sps = fma_ions.FMA(output_folder='output_Pb_on_momentum_ideal_lattice')

# Load Twiss and plot normalized phase space
sps_seq = fma_ions.SPS_sequence_maker()
sps_line, twiss_sps = sps_seq.load_xsuite_line_and_twiss(Qy_frac=25)

fma_sps.plot_normalized_phase_space(twiss_sps, particle_index=np.arange(1, 60), also_show_plot=False,
                                    case_name='z0 = 0')

# Off-momentum case

fma_sps2 = fma_ions.FMA(output_folder='output_Pb_off_momentum_ideal_lattice_z0_0dot1')
fma_sps2.plot_normalized_phase_space(twiss_sps, case_name='z0 = 0.1', particle_index=np.arange(1, 60))